## emotional speech recognition 

- recipe is based on LibriSpeech recipe
- train a model with Librispeech clean_100 data
- test data is from IEMOCAP
- test data consists of 4emotions (ang, hap, neu, sad)


### 01_libri_build.py
- this file makes essential files for run.sh
- text, spk2gender, utt2spk, wav.scp of 4 emotions in IEMOCAP

### 02_transcriptions_emotion.py
- this file makes transcriptions from corpus
- original transcription files have all the utterances together
- this file separates utterances to each emotions