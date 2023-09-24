"""Transformer for ASR in the SpeechBrain sytle.
Authors
* Yiqi Wang, 2022
* Jianyu Mao, 2022
"""
import copy
import torch  # noqa 42
from torch import nn
from typing import Optional
import re
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.containers import ModuleList
from speechbrain.lobes.models.transformer.InterleaveFormerLM import InterleaveFormerLM
from speechbrain.lobes.models.transformer.InterleaveFormer import (
    InterleaveFormerInterface,
    get_lookahead_mask,
    get_lookahead_hopping_mask,
    get_key_padding_mask,
    NormalizedEmbedding,
)
from speechbrain.nnet.activations import Swish
from speechbrain.dataio.dataio import length_to_mask


class InterleaveFormerASR(InterleaveFormerInterface):
    """This is an implementation of InterleaveFormer model for ASR.
    The architecture is based on the paper "PLACE HODLER":
    arxiv PLACE HODLER
    Arguments
    ----------
    tgt_vocab: int
        Size of vocabulary.
    input_size: int
        Input feature size.
    d_model : int, optional
        Embedding dimension size.
        (default=512).
    nhead : int, optional
        The number of heads in the multi-head attention models (default=8).
    num_encoder_layers : int, optional
        The number of causal-encoder-layers in the InterleaveFormer (default=6).
    num_decoder_layers : int, optional
        The number of sub-decoder-layers in the decoder (default=0).
    dim_ffn : int, optional
        The dimension of the feedforward network model (default=2048).
    dropout : int, optional
        The dropout value (default=0.1).
    activation : torch.nn.Module, optional
        The activation function of FFN layers.
        Recommended: relu or gelu (default=relu).
    positional_encoding: str, optional
        Type of positional encoding used. e.g. 'fixed_abs_sine' for fixed absolute positional encodings.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    kernel_size: int, optional
        Kernel size in convolutional layers when Conformer is used.
    bias: bool, optional
        Whether to use bias in Conformer convolutional layers.
    encoder_module: str, optional
        InterleaveFormer as a causal encoder. No other option!
    conformer_activation: torch.nn.Module, optional
        NOT USED
        Activation module used after Conformer convolutional layers. E.g. Swish, ReLU etc. it has to be a torch Module.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    max_length: int, optional
        Max length for the target and source sequence in input.
        Used for positional encodings.
    causal: bool, optional
        Whether the encoder should be causal or not (InterleaveFormer is always causal).
        If causal the Conformer convolutional layer is causal.
    Example
    -------
    >>> src = torch.rand([8, 200, 512]) # 200 is the padded total length including many bi-modality segments
    >>> tgt = torch.randint(0, 720, [8, 200])
    >>> net = TransformerASR(
    ...     720, 512, 512, 8, 1, 1, 1024, activation=torch.nn.GELU
    ... )
    >>> enc_out = net.forward(src, tgt) # not that enc_out actually contains both audio and text
    >>> enc_out.shape
    torch.Size([8, 200, 512])
    """

    def __init__(
        self,
        tgt_vocab,
        input_size,
        d_model=768,
        nhead=12,
        num_encoder_layers=12,
        num_decoder_layers=0,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=True,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "InterleaveFormer",
        conformer_activation: Optional[nn.Module] = Swish,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        init_path: Optional[str] = "", 
        audio_expert_init_path: Optional[str] = "", 
        init_mode=0,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
        )

        self.custom_src_module = ModuleList(
            Linear(
                input_size=input_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        print("\n\tPlease initialize custom_tgt / lm _module  by LM's src_module!")
        # used by ASR decoding
        self.custom_tgt_module = ModuleList(
            NormalizedEmbedding(d_model, tgt_vocab)
        )
        print("\n\tThey are all random init now")
        # used by LM 
        self.custom_lm_module = self.custom_tgt_module
        # NormalizedEmbedding(d_model, tgt_vocab)

        # modality embedding
        self.modality_emb = nn.Embedding(2, d_model)
        self.audio = torch.tensor([0]).long()
        self.text = torch.tensor([1]).long()
        self.init_mode = init_mode
        # reset parameters using xavier_normal_ and load weights from pretrained GPT
        if len(init_path) > 1:
            print("\n\nLM init\n\n")
            if len(audio_expert_init_path) < 1:
                audio_expert_init_path = None
            self._init_params_with_LM(init_path, audio_expert_init_path, init_mode)
        else:
            print("\n\nNormal init\n\n")
            self._init_params()
        # layers to decode is the encoder layers (decoder layers is 0)
        self.decode_layers = num_encoder_layers
        self.decode_dim = d_model

    def lm_forward(self, src):
        """
        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        """
        # make mask first
        src_mask = get_lookahead_mask(src)
        # assume padding id is 0
        src_key_padding_mask = get_key_padding_mask(src, 0)

        src = self.custom_lm_module(src)
        src = src + self.positional_encoding(src) + self.modality_emb(self.text.to(src.device))

        encoder_out, last_attn, weight_cache = self.encoder(
            mode="encode",
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        return encoder_out

    def forward(self, src, tgt, wave_len, seg_stats, pad_idx=0):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wave_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        seg_stats : list
            has audio and text length
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """
        # assert type(seg_stats) == type(dict()), f"Need seg_stats to be a valid dictionary for modality expert!"
        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if src.ndim == 4:
            bz, t, ch1, ch2 = src.shape
            src = src.reshape(bz, t, ch1 * ch2)

        (
            src_key_padding_mask,
            tgt_key_padding_mask,
            src_mask,
            tgt_mask, # this one is the hopping causal mask! 
        ) = self.make_masks(src, tgt, wave_len, seg_stats, pad_idx=pad_idx)

        src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        if self.attention_type == "RelPosMHAXL":
            pos_embs_encoder = self.positional_encoding(src) + self.modality_emb(self.audio.to(src.device))
        elif self.positional_encoding_type == "fixed_abs_sine":
            src = src + self.positional_encoding(src) + self.modality_emb(self.audio.to(src.device))
            pos_embs_encoder = None

        tgt = self.custom_tgt_module(tgt)
        if self.attention_type == "RelPosMHAXL":
            tgt = tgt + self.positional_encoding(tgt) + self.modality_emb(self.text.to(tgt.device))
            pos_embs_target = None
            pos_embs_encoder = None
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt) + self.modality_emb(self.text.to(tgt.device))
            pos_embs_target = None
            pos_embs_encoder = None

        # first encoder audio
        encoded_output, _, _ = self.encoder(
            mode="encode",
            src=src,
            src_mask=src_mask, # should be None
            src_key_padding_mask=src_key_padding_mask,
            tgt=None,
            tgt_mask=None,
            tgt_key_padding_mask=None,
            pos_embs=pos_embs_encoder,
        )
        
        final_src = torch.cat([encoded_output, tgt], dim = 1)
        final_padding_mask = torch.cat([src_key_padding_mask, tgt_key_padding_mask], dim = 1)

        # Decoding 
        decoded_output, _, _ = self.encoder(
            mode="decode",
            src=final_src,
            src_mask=None,
            src_key_padding_mask=None,
            tgt=tgt,
            tgt_mask=tgt_mask, # this must be a semi-causal mask, hopping style
            tgt_key_padding_mask=final_padding_mask,
            pos_embs=pos_embs_encoder,
        )

        return encoded_output, decoded_output

    def make_masks(self, src, tgt, wave_len = None, seg_stats = None, pad_idx=0):
        """This method generates the masks for training the transformer model.
        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        wave_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        seg_stats:
            an array contains max len of src and max len of tgt
            # dictionary for milestones 2
            # The sum of each key's array is the total true length of a sequence where each element in the array indicates segment len
            # Format: { key_i: array for i in range(batch)} used by modality expert.
            # Key_i: key to index a sample's modality stats.
            # Value_of_key_i: an array. Even element is audio segment true length. Odd element is text tokens true length.
        pad_idx : int
            The index for <pad> token (default=0).
        """
        src_key_padding_mask = None
        if  wave_len is not None:
            abs_len = torch.round(wave_len * src.shape[1])
            src_key_padding_mask = ~length_to_mask(abs_len).bool()
        tgt_key_padding_mask = get_key_padding_mask(tgt, pad_idx=pad_idx)

        src_mask = None
        tgt_mask = get_lookahead_hopping_mask(tgt, seg_stats) # hopping causal mask implemented in InterleaveFormer.py
        return src_key_padding_mask, tgt_key_padding_mask, src_mask, tgt_mask

    @torch.no_grad()
    def decode_use_cache(self, tgt, encoder_out, cache, enc_len=None, seg_stats=None):
        """This method implements a decoding step for the transformer model.
        It use weight_caching to improve decoding speed.
        Arguments
        ---------
        tgt : torch.Tensor
            The sequence to the decoder.
        encoder_out : torch.Tensor
            Hidden output of the encoder.
        cache : Dictionary of torch.Tensor
            Cached weight for previous decoding step. A dictionry where key is layer index
            Each value of the dictionry is 3d tensor: batch x horizon x dim 
        enc_len : torch.LongTensor
            The actual length of encoder states.
        seg_stats : list
            has audio and text length
        """
        # text length  == latest token
        seg_stats[1] = 1
        # With weight caching, no need for a mask.
        tgt_mask = None
        # tgt_mask = get_lookahead_hopping_mask(tgt[:, -1:], seg_stats)
        # print("\n\nCreate tgt mask:", tgt_mask.shape, end="\n\n")
        src_key_padding_mask = None
        if enc_len is not None:
            src_key_padding_mask = (1 - length_to_mask(enc_len)).bool()
        tgt_key_padding_mask = torch.zeros_like(tgt, device=tgt.device).bool()
        tgt = self.custom_tgt_module(tgt)
        
        if self.attention_type == "RelPosMHAXL":
            # use standard sinusoidal pos encoding in decoder
            tgt = tgt + self.positional_encoding_decoder(tgt) + self.modality_emb(self.text.to(tgt.device))
            pos_embs_encoder = None  # self.positional_encoding(src)
            pos_embs_target = None
        elif self.positional_encoding_type == "fixed_abs_sine":
            tgt = tgt + self.positional_encoding(tgt) + + self.modality_emb(self.text.to(tgt.device))
            pos_embs_target = None
            pos_embs_encoder = None

        final_src = torch.cat([encoder_out, tgt], dim = 1)
        # assert False, f"{src_key_padding_mask.shape} {tgt_key_padding_mask.shape}"
        final_padding_mask = torch.cat([src_key_padding_mask, tgt_key_padding_mask], dim = 1)
        # Decoding 
        decoded_output, multihead_attns, cache = self.encoder(
            mode="decode",
            src=final_src,
            src_mask=None,
            src_key_padding_mask=None,
            tgt=tgt,
            tgt_mask=tgt_mask, # this must be a semi-causal mask, hopping style
            tgt_key_padding_mask=final_padding_mask,
            pos_embs=pos_embs_encoder,
            cache = cache
        )

        return decoded_output, multihead_attns[-1], cache
    
    # @torch.no_grad()
    # def decode(self, tgt, encoder_out, enc_len=None):
    #     """This method implements a decoding step for the InterleaveFormer model.
    #     Arguments
    #     ---------
    #     tgt : torch.Tensor
    #         The sequence to the decoder.
    #     src : torch.Tensor
    #         Raw audio instead of encoded audio.
    #     enc_len : torch.LongTensor
    #         Not used. 
    #         The actual length of encoder states.
    #     """
    #     # length of audio and latest text token)
    #     seg_stats=[encoder_out.shape[1], 1]
    #     # make mask 
    #     src_key_padding_mask = None
    #     if wav_len is not None:
    #         abs_len = torch.round(wav_len * src.shape[1])
    #         src_key_padding_mask = ~length_to_mask(abs_len).bool()
    #     tgt_mask = get_lookahead_hopping_mask(tgt, seg_stats)
    #     tgt_key_padding_mask = torch.zeros_like(tgt).bool().to(tgt_mask.device)
    #     # embed text with position + modality embedding
    #     tgt = self.custom_tgt_module(tgt)
    #     if self.attention_type == "RelPosMHAXL":
    #         assert False, f"Don't support RelPosMHAXL yet"
    #     elif self.positional_encoding_type == "fixed_abs_sine":
    #         tgt = tgt + self.positional_encoding(tgt) + self.modality_emb(self.text.to(tgt.device))
    #         pos_embs_target = None
    #         pos_embs_encoder = None

    #     final_src = torch.cat([encoded_output, tgt], dim = 1)
    #     final_padding_mask = torch.cat([src_key_padding_mask, tgt_key_padding_mask], dim = 1)
    #     # Decoding 
    #     decoded_output, multihead_attns = self.encoder(
    #         mode="decode",
    #         src=final_src,
    #         src_mask=None,
    #         src_key_padding_mask=None,
    #         tgt=tgt,
    #         tgt_mask=tgt_mask, # this must be a semi-causal mask, hopping style
    #         tgt_key_padding_mask=final_padding_mask,
    #         pos_embs=pos_embs_encoder,
    #     )

    #     return prediction, multihead_attns[-1]


    def _init_params(self):
        # Init parameters
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
        
        # set audio expert parameters to trainable, 
        # all other layers in transformer frozen
        # print("\nComment out me and below to make everyone except audio_expert unfrozen!\n")
        # for name, p in self.named_parameters():
            # if 'encoder' in name and 'audio_expert' not in name:
            #     p.requires_grad = False

    def _init_params_with_LM(self, init_path, audio_expert = None, init_mode=2):
        '''
        If 0:
            Frozen all self-attn stuff + text expert, random init audio expert and train.
        If 1:
            Init by an ASR where output/input models + audio expert are trained.
            The rest are LM init. Train them all.

        '''
        assert init_mode < 3, f"No mode larger than 2"
        # initialize the asr model entirely
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
        print('#'*20, 'Compare key', '#'*20,)
        if audio_expert is not None:
            print("\tLoad audio checkpoint from ", audio_expert)
            ref_model_state_dic = torch.load(audio_expert, map_location=torch.device('cuda:1'))
        else:
            print("\tLoad LM checkpoint from ", init_path)
            ref_model_state_dic  = torch.load(init_path, map_location=torch.device('cuda:1') )
        for key1, in zip( ref_model_state_dic):
            print('reference key: ', key1, )

        state_dict = self.state_dict()
        for key2, in zip( state_dict):
            print('asr key: ', key2, )
        print('#'*20, 'Start init with Init mode', init_mode, '#'*20,)

        # ASR weights init first
        state_dict = self.state_dict()
        # initialization
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
        
        # set current model weight with ref model weight
        count = 0
        for param_key in self.state_dict():
            count += 1
            # LM only have "audio expert" trained
            # Initialize ASR's text epxert with LM's audio exper if audio_expert init path is None.

            if "text_expert" in param_key:
                if(audio_expert is None) and ( init_mode == 0 or init_mode == 2):
                    # in init_mode 0 and 2, LM init, text expert of LM is not trained ,use "audio" expert.
                    state_dict[param_key] = ref_model_state_dic["1." + param_key.replace('text', 'audio')]
                elif init_mode == 1:
                    # in init mode 1, asr checkpoint is used, text expert is already initialized (by LM audio expert)
                    # no need for replacement.
                    state_dict[param_key] = ref_model_state_dic["1." + param_key]
                else:
                    assert False
            elif "audio_expert" in param_key:
                # try to init the audio expert with audio_expert checkpoint separtely trained
                if audio_expert is not None:
                    _check_key = "1." + param_key
                    state_dict[param_key] = ref_model_state_dic[_check_key]
                    # print("\t\tInit model audio expert with ref key:", _check_key)
                else:
                    # random init audio expert if no path provided
                    assert init_mode == 0 or init_mode == 2
                    # we we only encouter this if in init_mode 0.
                    # where audio expert is random init 
                    # since LM's audio expert is actually text expert.
            
            else:
                try:
                    state_dict[param_key] = ref_model_state_dic["1." + param_key]
                    print('Loading key:', param_key)
                except:
                    if init_mode == 0 or init_mode == 2:
                        print("Skip a different param:", param_key, state_dict[param_key].shape)
                    elif init_mode == 1:
                        assert False, f"This shouldn't happend in init_mode 1"
                    else:
                        assert False, f"unknown init_mode. Valid mode include 0,1"

        # overwrite weight
        self.load_state_dict(state_dict=state_dict)
        if ref_model_state_dic is not None:
            del ref_model_state_dic
        # mode mode 0 has frozen layers, mode 2 don't.
        if init_mode == 0:
            for name, p in self.named_parameters():
                if 'encoder' in name and ( 'audio_expert' not in name):
                # if 'encoder' in name and ( 'audio_expert' not in name) and ( 'norm' not in name):
                    p.requires_grad = False
                    print('\tSet', name, 'to NOT required gradient')
        