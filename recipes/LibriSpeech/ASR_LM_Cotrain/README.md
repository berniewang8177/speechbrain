## csvs files
wiki-103 related files are symbolic links since they're too large. 
Below are the processed wiki-103, mimicing Speechbrain style csv.

#### Donwload linke
- [wiki-test.csv](https://drive.google.com/file/d/1XfaEGsahy8N2Wqmiu22IMBgw7M8Q7Vsr/view?usp=sharing)
- [wiki-train.csv](https://drive.google.com/file/d/1o68Tlc8VrCltEAqJUIMuDz7-nRhBstna/view?usp=sharing)
- [wiki-valid.csv](https://drive.google.com/file/d/1SoMKMhKLyH6-TGtCk04f-G2N8OmGXvKF/view?usp=sharing)

#### Usage in GPT_Style
1. cotrain.py co-train an ASR model with Librispeech 960 text (lm) + 100h audio-text (asr) together
2. lmtrain.py (Staled), train an LM model with 960h text
3. lmtrain_wiki.py train an LM model on wiki-103


