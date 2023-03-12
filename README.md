# Robo-Shaul

This repository contains all you need in order to use the train your own robo-shaul or use pre-trained models.

The system consists of the SASPEECH dataset, which is a collection of recordings of Shaul Amsterdamski's unedited recordings for the podcast 'Hayot Kis', and a Text-to-Speech system trained on the dataset, implemented in the Coqui AI TTS framework.

The text-to-speech system consists of two parts:

1. A text-to-mel spectrogram model called OverFlow
2. A mel spectrogram-to-wav model called HiFi-GAN

To download the dataset for training, go to [TODO]

To download the trained models, go to [TODO] for overflow, [TODO] for hifi-gan

The model expects diacritized hebrew (עברית מנוקדת), we recommend [Nakdimon](https://nakdimon.org) by Elazar Gershuni and Yuval Pinter. The link is to a free online tool, the code and model are available on GitHub at [https://github.com/elazarg/nakdimon](https://github.com/elazarg/nakdimon)
## Installation

Here are the installation instructions necessary to use our trained models or to train your own models.

It is recommended to do these steps inside a virtual environment, conda env, or similar.

Steps:

1. Clone our fork of coqui-tts: `git clone https://github.com/shenberg/TTS`
2. Install it as an editable install: `pip install -e TTS`
3. Download our trained models & extract the archives: [Overflow TTS](TODO: link), [HiFi-GAN](TODO: link)
4. Test that it works: `CUDA_VISIBLE_DEVICES=0 tts --text "עַכְשָׁיו, לְאַט לְאַט, נָסוּ לְדַמְיֵין סוּפֶּרְמַרְקֶט." --model_path /path/to/saspeech_overflow.pth  --config_path /path/to/config_saspeech_overflow.json  --vocoder_path /path/to/saspeech_hifigan.pth --vocoder_config_path /path/to/config_saspeech_hifigan.json --out_path test.wav`

You should now have a file named `test.wav` which has the model's TTS output.

Note: Many thanks to 
### Training

Once you have successfully created `test.wav` using our trained models, the time has come to set up training your own models.

**NOTE**: we are assuming you are in a python environment where you successfully installed TTS and generated audio with it.

#### Training OverFlow

1. Download the SASPEECH dataset and extract it to `TTS/recipes/saspeech/data/`
2. Resample the audio from 44100Hz to 22050Hz: `TODO`
3. Run training script: `TODO`
4. Look at metrics as the script is running: `TODO: tensorboard`

Outputs, meaning model checkpoints and logs, are stored in TODO

#### Training HiFi-GAN

We assume here that you have already resampled the dataset as in step 2 of training OverFlow

1. Preprocess SASPEECH for hifigan training\*: `TODO`
2. Run training script: `TODO`
3. Look at metrics as it's running: `TODO: Tensorboard`

Fine-tuning an existing run is simple: add the argument `--restore_path /path/to/saspeech_hifigan`, so, if I downloaded saspeech_hifigan.tar.gz and extracted it to `~/saspeech_hifigan_trained`, in order to continue training from that point, my full command would be `TODO --restore_path ~/saspeech_hifigan_trained`

Outputs, meaning model checkpoints and logs, are stored in TODO
