#!/usr/bin/env python3
"""Recipe for training a Transformer ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/transformer.yaml
> python train.py hparams/conformer.yaml

With the default hyperparameters, the system employs a convolutional frontend and a transformer.
The decoder is based on a Transformer decoder. Beamsearch coupled with a Transformer
language model is used  on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

The best model is the average of the checkpoints from last 5 epochs.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.


Authors
 * Jianyuan Zhong 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020, 2021, 2022
 * Titouan Parcollet 2021, 2022
"""

import os
import sys
import numpy as np
import gc
import torch
from torch.utils.data import DataLoader
import logging
import time
from enum import Enum, auto
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import LoopedLoader
from tqdm.contrib import tqdm
import wandb

logger = logging.getLogger(__name__)

# class Stage(Enum):
#     """Simple enum to track stage of experiments."""

#     TRAIN = auto()
#     VALID = auto()
#     TEST = auto()

# Define training procedure
class ASR(sb.core.Brain):

    def _fit_train(self, train_set, epoch, enable, cotrain = False, asr_scale = 1):
        # train_set here is list: [ asr_dataset, lm_dataset]

        # Training stage
        self.on_stage_start(sb.Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # these two aren't saved as self.avg_train_loss
        asr_losses = 0
        lm_losses = 0
        seq_losses = 0

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        with tqdm(
            train_set[0],
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            count = 0
            for asr_batch in t:
                count += 1
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                
                # If true, change to ASC training instead of co-train
                if cotrain == False:
                    batch = asr_batch
                else:
                    # How much we learn from LM depends on asr 
                    try:
                        lm_batch = next(iter(train_set[1]))
                    except:
                        # use backup dataset to create the lm dataset
                        train_set[1] = self.make_dataloader(
                            train_set[2], stage=sb.Stage.TRAIN, **train_loader_kwargs[1]
                        )
                        lm_batch = next(iter(train_set[1]))
                    batch = [ asr_batch, lm_batch ]
                
                loss, asr_loss, lm_loss, seq_loss = self.fit_batch(batch, asr_scale = asr_scale)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                asr_losses = self.update_average(
                    asr_loss, asr_losses)
                if lm_losses > 0 and lm_loss == 0:
                    # previous got LM training and then exit training.
                    # set logg to zero.
                    lm_losses == 0
                else:
                    # normal updates
                    lm_losses = self.update_average(
                        lm_loss, lm_losses)
                # sanity check
                assert not( 0 < lm_loss < 1 and lm_losses == 0 ), f"{lm_loss} {lm_losses}"
                seq_losses = self.update_average(
                    seq_loss, seq_losses)

                # set loss print to screen
                t.set_postfix( 
                    train_loss=self.avg_train_loss, 
                    scaled_asr_loss=asr_losses * asr_scale,  
                    lm_loss=lm_losses,  
                    asr_loss = asr_losses, asr_scale=asr_scale, seq_loss=seq_losses, )
                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if (
                    self.checkpointer is not None
                    and self.ckpt_interval_minutes > 0
                    and time.time() - last_ckpt_time
                    >= self.ckpt_interval_minutes * 60.0
                ):
                    # This should not use run_on_main, because that
                    # includes a DDP barrier. That eventually leads to a
                    # crash when the processes'
                    # time.time() - last_ckpt_time differ and some
                    # processes enter this block while others don't,
                    # missing the barrier.
                    if sb.utils.distributed.if_main_process():
                        self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()

        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch, asr_losses, lm_losses, seq_losses)
        self.avg_train_loss = 0.0
        self.step = 0

        # return the LM datset to reuse in the next epochs
        return lm_losses, seq_losses, train_set[1]

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs=[],
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.

        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:

        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``

        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.

        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : list of Dataset, DataLoader
            Each item is a set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly. train_set = [ASR_data, LM_data]
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : list of dict
            Each dict (Kwargs) passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """
        if not (isinstance(train_set[0], DataLoader) or isinstance(train_set[0], LoopedLoader)) \
                 and not ( isinstance(train_set[1], DataLoader or isinstance(train_set[1], LoopedLoader) )
        ):
            train_asr_set = self.make_dataloader(
                train_set[0], stage=sb.Stage.TRAIN, **train_loader_kwargs[0]
            )
            train_lm_set = self.make_dataloader(
                train_set[1], stage=sb.Stage.TRAIN, **train_loader_kwargs[1]
            )
            # for LM, creat a backup version !!
            # so that if next() is not happy, create a new one for it!
            train_set = [train_asr_set, train_lm_set, train_set[1]]
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        # whether shift from ASR train to co training
        cotrain = True

        x = list( np.linspace(0.01, 0.05,  4) )
        y = list( np.logspace(-4, 0, base=2, num=16) )
        asr_scales = x + y

        for epoch in epoch_counter:
            if len(asr_scales) > 1:
                asr_scale = asr_scales.pop(0)
            else:
                asr_scale = asr_scale[0]
            unscaled_lm_loss, seq_loss, lm_dataset = self._fit_train(train_set=train_set, epoch=epoch, enable=enable, cotrain = cotrain, asr_scale = asr_scale)
            # we could reuse the last lm_dataset (if not finished in the last epoch) in the next epoch.
            train_set[1] = lm_dataset

            # Change cotrain to asr only if lm has been small.
            cotrain = True
            if unscaled_lm_loss < 1:
                cotrain = False

            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

    def compute_forward(self, batch, stage):
        
        """Forward computations from the waveform batches to the output probabilities."""
        if type(batch) == list:
            # co-training
            asr_batch = batch[0].to(self.device)
            lm_batch = batch[1].to(self.device)
            lm_tokens_bos, _ = lm_batch.tokens_bos
        else:
            asr_batch = batch.to(self.device)
        asr_tokens_bos, _ = asr_batch.tokens_bos

        if hasattr(self.hparams, "train_mode"):
            mode = self.hparams.train_mode
            # we only do text forward during training
            if ( (mode == 0 or mode == 2) and stage == sb.Stage.TRAIN ) and type(batch) == list:
                lm_pred = self.hparams.Transformer.lm_forward(lm_tokens_bos)
                # # output layer for seq2seq log-probabilities
                pred = self.modules.seq_lin(lm_pred)
                p_seq_lm = self.hparams.log_softmax(pred)
                p_ctc = None
                wav_lens = None
            else:
                p_seq_lm = None

            if mode == 2:
                ################## ASR objective ###############
                wavs, wav_lens = asr_batch.sig
                # Add augmentation if specified
                if stage == sb.Stage.TRAIN:
                    if hasattr(self.modules, "env_corrupt"):
                        wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                        wavs = torch.cat([wavs, wavs_noise], dim=0)
                        wav_lens = torch.cat([wav_lens, wav_lens])
                        asr_tokens_bos = torch.cat([asr_tokens_bos, asr_tokens_bos], dim=0)

                # compute features
                feats = self.hparams.compute_features(wavs)
                current_epoch = self.hparams.epoch_counter.current
                feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)
                # print('ASR batch:', feats.shape, end = '\n\n')
                if stage == sb.Stage.TRAIN:
                    if hasattr(self.hparams, "augmentation"):
                        feats = self.hparams.augmentation(feats)

                # forward modules
                src = self.modules.CNN(feats)

                _, audio_max_len, _, _ = src.shape
                _, text_max_len = asr_tokens_bos.shape
                seg_stats = [audio_max_len, text_max_len ]
                enc_out, asr_pred = self.modules.Transformer(
                    src, asr_tokens_bos, wav_lens, seg_stats = seg_stats, pad_idx=self.hparams.pad_index,
                )

                # output layer for ctc log-probabilities
                logits = self.modules.ctc_lin(enc_out)
                p_ctc = self.hparams.log_softmax(logits)

                # output layer for seq2seq log-probabilities
                pred = self.modules.seq_lin(asr_pred)
                p_seq_asr = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens, seg_stats)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens, seg_stats)

        return p_ctc, [p_seq_asr, p_seq_lm ], wav_lens, hyps

    def compute_lm_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        batch = batch.to(self.device)
        tokens_eos, tokens_len = batch.tokens_eos
        loss = self.hparams.compute_lm_cost(
            predictions, tokens_eos, length=tokens_len
        )
        # validation is asr only, thus, we don't have acc_metric here
        # if stage != sb.Stage.TRAIN:
        #     # compute the accuracy of the one-step-forward prediction
        #     self.acc_metric.append(predictions, tokens_eos, tokens_len)
        return loss

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        # now as training progresses we use real prediction from the prev step instead of teacher forcing

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )
        seq2seq_loss = (1 - self.hparams.ctc_weight) * loss_seq.detach().cpu()
        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss, seq2seq_loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            # ASR Losses are excluded from mixed precision to avoid instabilities
            asr_predictions = ( predictions[0], predictions[1][0], predictions[2], predictions[3], )
            loss, seq_loss = self.compute_objectives(asr_predictions, batch, stage = stage)
        
        gc.collect()
        torch.cuda.empty_cache()
        del batch, seq_loss

        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss,  epoch, asr_loss = 0, lm_loss = 0, seq_loss=0):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        
        if stage == sb.Stage.TRAIN:
            stage_stats = {"loss": stage_loss, "asr_loss": asr_loss, "asr_seq_loss": seq_loss, "lm_loss": lm_loss}
            self.train_stats = stage_stats
        else:
            stage_stats = {"loss": stage_loss}
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def fit_batch(self, batch, asr_scale = 1):

        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):

                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            
            if len(outputs) == 4:

                 # ASR losses 
                asr_outputs = ( outputs[0], outputs[1][0], outputs[2], outputs[3], )
                if type(batch) == list:
                    asr_loss, seq_loss = self.compute_objectives(asr_outputs, batch[0], sb.Stage.TRAIN)
                else:
                    asr_loss, seq_loss = self.compute_objectives(asr_outputs, batch, sb.Stage.TRAIN)
                # LM losses
                if outputs[1][1] != None:
                    lm_loss = self.compute_lm_objectives(outputs[1][1], batch[1], sb.Stage.TRAIN)
                else:
                    lm_loss = torch.tensor(0)
                
                loss = asr_loss *  asr_scale + lm_loss 
            else:
                assert False, f"Output shape must be len 2, with asr and lm result"
            if self.log_to_wandb:
                wandb.log({
                    "total_loss": loss.item(), 
                    "scaled_asr_loss": asr_loss.item() * asr_scale, 
                    'unscaled_seq_loss': seq_loss,
                    "nonfinite_count": self.nonfinite_count,
                    })
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)
        else:
            assert False, f"I haven't implement this one"
        
        self.on_fit_batch_end(batch, outputs, loss, should_step)
        
        gc.collect()
        torch.cuda.empty_cache()
        del batch

        return loss.detach().cpu(), asr_loss.detach().cpu(), lm_loss.detach().cpu(), seq_loss


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    if int(hparams["train_mode"]) == 2:
        new_file = hparams["train_csv"].replace('train.csv', 'train-clean-100.csv')
        hparams["train_csv"] = new_file
    else:
        assert False, f"This cotrain.py only supports mode 2"
    
    train_asr_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder} )
    train_lm_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_lm_csv"], replacements={"data_root": data_folder} )
    if len(train_asr_data) < len(train_lm_data):
        batch_len_scale = int( round( len(train_lm_data) / len(train_asr_data)) )
        print( '*'*40, f"\nbatch_len_scale of LM is: {batch_len_scale}, set it to 1 NOW\n", '*'*40 ) 
    else:
        assert False, f"LM should larger than ASR size"

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    if int(hparams["train_mode"]) in [1,2]:
        test_datasets = {}
        for csv_file in hparams["test_csv"]:
            name = Path(csv_file).stem
            test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_file, replacements={"data_root": data_folder}
            )
            test_datasets[name] = test_datasets[name].filtered_sorted(
                sort_key="duration"
            )
    else:
        test_datasets = None
    
    datasets = [train_asr_data, valid_data]
    valtest_datasets = [valid_data]
    if int(hparams["train_mode"]) in [1,2]:
        datasets += [i for k, i in test_datasets.items()]
        valtest_datasets += [i for k, i in test_datasets.items()]
    
    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig
    if int(hparams["train_mode"]) in [1,2]:
        sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if hparams["speed_perturb"]:
            sig = sb.dataio.dataio.read_audio(wav)

            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig
    
    if int(hparams["train_mode"]) in [1,2]:
        sb.dataio.dataset.add_dynamic_item([train_asr_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets + [ train_lm_data ], text_pipeline)

    # 4. Set output:
    if int(hparams["train_mode"]) in [1,2]:
        # datsets: all sets except train_lm
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
        )
        # dataset that is text only
        # assert False, f"{type(train_lm_data)} {type(datasets)}"
        sb.dataio.dataset.set_output_keys(
            [ train_lm_data ], ["id", "wrd", "tokens_bos", "tokens_eos", "tokens"],
        )
    else:
        assert False, f"Seems you are at mode 0? Please use lmtrain.py instead of cotrain.py"
        # sb.dataio.dataset.set_output_keys(
        #     datasets, ["id", "wrd", "tokens_bos", "tokens_eos", "tokens"],
        # )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_asr_batch_sampler = DynamicBatchSampler(
            train_asr_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
            max_batch_ex=dynamic_hparams["max_batch_ex"],
        )

        train_lm_batch_sampler = DynamicBatchSampler(
            train_lm_data,
            dynamic_hparams["max_batch_len"]*batch_len_scale,
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
            max_batch_ex=dynamic_hparams["max_batch_ex"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )
    
    return (
        train_asr_data,
        train_lm_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_asr_batch_sampler,
        train_lm_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from cotraining_prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    log_to_wandb = hparams['wandb_log']
    if log_to_wandb:
        wandb.init(
            name="With_LM" , # hard coded
            group= "debug_gradient",
            project= "ASR_LM",
            config={ "learning_rate": hparams['lr_adam'] }
        )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "mode": int(hparams["train_mode"]),
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "lm_splits": hparams["lm_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train_lm.csv",
            "skip_prep": hparams["skip_prep"],
        },
    ) 
        
    (
        train_asr_data,
        train_lm_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_asr_bsampler,
        train_lm_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        log_to_wandb = log_to_wandb,
    )
    
    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_asr_bsampler is not None:
        train_asr_dataloader_opts = {
            "batch_sampler": train_asr_bsampler,
            "num_workers": hparams["num_workers"],
        }

    if train_lm_bsampler is not None:
        train_lm_dataloader_opts = {
            "batch_sampler": train_lm_bsampler,
            "num_workers": hparams["num_workers"],
        }

    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        [ train_asr_data, train_lm_data],
        valid_data,
        train_loader_kwargs=[ train_asr_dataloader_opts, train_lm_dataloader_opts ],
        valid_loader_kwargs=valid_dataloader_opts,
    )
    
    if int(hparams["train_mode"]) in [1,2]:
        # Testing
        for k in test_datasets.keys():  # keys are test_clean, test_other etc
            asr_brain.hparams.wer_file = os.path.join(
                hparams["output_folder"], "wer_{}.txt".format(k)
            )
            asr_brain.evaluate(
                test_datasets[k],
                max_key="ACC",
                test_loader_kwargs=hparams["test_dataloader_opts"],
            )