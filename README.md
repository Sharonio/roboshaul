# Robo-Shaul

Welcome to the Robo-Shaul repository! Here, you'll find everything you need to train your own Robo-Shaul or use pre-trained models. Robo-Shaul is a text-to-speech system that converts diacritized Hebrew text into speech using the Coqui AI TTS framework.

#### For a website look [here](https://www.roboshaul.com)

#### For a demo look [here](https://anonymous19283746.github.io/saspeech/)

#### For a quick start look at [roboshaul_usage.ipynb](https://github.com/Sharonio/roboshaul/blob/main/roboshaul_usage.ipynb) or <a target="_blank" href="https://colab.research.google.com/github/Sharonio/roboshaul/blob/main/roboshaul_usage_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#### For the חיות כיס podcast documenting the project listen [here](https://open.spotify.com/episode/7eM8KcpUGMxOk6X5WQYdh5?si=3xf0TNzwRTSHaCo8jIozOg)


The system consists of the SASPEECH dataset, which is a collection of recordings of Shaul Amsterdamski's unedited recordings for the podcast 'Hayot Kis', and a Text-to-Speech system trained on the dataset, implemented in the Coqui AI TTS framework.

The text-to-speech system consists of two parts:

1. A text-to-mel spectrogram model called OverFlow
2. A mel spectrogram-to-wav model called HiFi-GAN

To download the dataset for training, go to [Open SLR](https://www.openslr.org/134/)

To download the trained models, go to [this link](https://drive.google.com/drive/folders/1C7xfx8p8iTaF73bvfvIdkGDPv01wvjmx?usp=share_link) for OverFlow, [this link](https://drive.google.com/drive/folders/1SC6IQtdXH1SjHSgLGY1iZtl9nwDGQ072?usp=share_link) for HiFi-GAN.

The model expects diacritized hebrew (עברית מנוקדת), we recommend [Nakdimon](https://nakdimon.org) by Elazar Gershuni and Yuval Pinter. The link is to a free online tool, the code and model are also available on GitHub at [https://github.com/elazarg/nakdimon](https://github.com/elazarg/nakdimon)
## Installation

Here are the installation instructions necessary to use our trained models or to train your own models. They have been tested on Ubuntu 22.04, and should work as-is on Mac except for there being no CUDA.
If you're on Windows, running `pip install numpy==1.23.5 numba==0.56.4` after installation has been reported to make it work on python 3.10, however not thoroughly tested.

It is recommended to do these steps inside a virtual environment, conda env, or similar.

Steps:

1. Clone our fork of coqui-tts: `git clone https://github.com/shenberg/TTS`
2. Install it as an editable install: `pip install -e TTS`
3. Download our trained models: [Overflow TTS](https://drive.google.com/drive/folders/1C7xfx8p8iTaF73bvfvIdkGDPv01wvjmx?usp=share_link), [HiFi-GAN](https://drive.google.com/drive/folders/1SC6IQtdXH1SjHSgLGY1iZtl9nwDGQ072?usp=share_link)
4. Test that it works: `CUDA_VISIBLE_DEVICES=0 tts --text "עַכְשָׁיו, לְאַט לְאַט, נָסוּ לְדַמְיֵין סוּפֶּרְמַרְקֶט." --model_path /path/to/saspeech_overflow.pth  --config_path /path/to/config_saspeech_overflow.json  --vocoder_path /path/to/saspeech_hifigan.pth --vocoder_config_path /path/to/config_saspeech_hifigan.json --out_path test.wav`

You should now have a file named `test.wav` which has the model's TTS output.

**NOTE**: By now our modifications for Coqui-TTS have been upstreamed so a regular installation should work.

#### Example windows installation using conda
```
conda create --name "roboshaul" python=3.10
conda activate roboshaul
conda install -c anaconda cython
conda install -c conda-forge jupyterlab
git clone https://github.com/shenberg/TTS
pip install -e TTS
pip install numpy==1.23.5 numba==0.56.4
```


## Training

Once you have successfully created `test.wav` using our trained models, the time has come to set up training your own models.

**NOTE**: we are assuming you are in a python environment where you successfully installed TTS and generated audio with it.

### Data Preparation

The sequence of actions necessary to extract the dataset (replace `/path/to` with the real path of course):

```
$ unzip Roboshaul.zip
$ mkdir /path/to/TTS/recipes/saspeech/data
$ mv saspeech_*tar.gz /path/to/TTS/recipes/saspeech/data
$ cd /path/to/TTS/recipes/saspeech/data
$ tar zxvf saspeech_automatic_data_v1.0.tar.gz
$ tar zxvf saspeech_gold_standard_v1.0.tar.gz
$ rm saspeech_automatic_data_v1.0.tar.gz saspeech_gold_standard_v1.0.tar.gz
```

Now `data/` should have two sub-directories, `data/saspeech_automatic_data/`, `data/saspeech_gold_standard/`

#### Resampling

The files are provided at a sampling rate of 44100Hz, but we train models on audio at 22050Hz in order to decrease computational loads while still providing decent quality.

```bash
$ python -m TTS.bin.resample --input_dir data/saspeech_gold_standard/ --output_dir data/saspeech_gold_standard_resampled --output_sr 22050
$ python -m TTS.bin.resample --input_dir data/saspeech_automatic_data/ --output_dir data/saspeech_automatic_data_resampled --output_sr 22050
```

Advanced users may want to resample using sox vhq algorithm with intermediate phase, as it is higher-quality than what these scripts provide (they use librosa 0.8, which uses resampy behind the scenes and is okay quality).
#### Windowing for HiFi-GAN
Some of the audio files are long, especially in the automatically tagged portion of the data where some exceed a minute in length.

HiFi-GAN training expects short files to take random windows out of, so we have a script that will break down long files into shorter segments. We will use this script to also gather both the gold-standard and automatic subsets of the data into one directory.

```bash
$ cd hifigan
$ python prepare_dataset_for_hifigan.py --input_dir ../data/saspeech_gold_standard_resampled/wavs/ ../data/saspeech_automatic_data_resampled/wavs/ --output_dir ../data/saspeech_all_windowed
```

### Training OverFlow

Run training script from `TTS/recipes/saspeech/overflow/`:
```bash
$ CUDA_VISIBLE_DEVICES=0 python train_overflow.py
```

#### Metrics and Saved Models
When you run the training script, it will print out a bunch of logs, and one of the rows will say something like:

```
> Start Tensorboard: tensorboard --logdir=/path/to/TTS/recipes/saspeech/overflow/overflow_saspeech_gold-March-14-2023_08+46AM-91fd5654
```
In order to look at metrics, open another terminal in the same python virtual environment and run the command from the logs, e.g.

```bash 
$ tensorboard --logdir=/path/to/TTS/recipes/saspeech/overflow/overflow_saspeech_gold-March-14-2023_08+46AM-91fd5654
```

Now that tensorboard is running, you can go to [http://localhost:6006](http://localhost:6006) in your browser to view metrics as training evolves.

Model checkpoints are saved to the same directory, so in our above example, the directory `/path/to/TTS/recipes/saspeech/overflow/overflow_saspeech_gold-March-14-2023_08+46AM-91fd5654` will also contain model checkpoints

### Training HiFi-GAN
Run training script from `TTS/recipes/saspeech/hifigan/`: 
```bash
$ CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python train_hifigan.py
```

#### Fine-Tuning existing checkpoint
Fine-tuning an existing run is simple: add the argument `--restore_path /path/to/saspeech_hifigan`, so, if I downloaded saspeech_hifigan.tar.gz and extracted it to `~/saspeech_hifigan_trained`, in order to continue training from that point, my full command would be `TODO --restore_path ~/saspeech_hifigan_trained`.


#### Metrics and Saved Models
When you run the training script, it will print out a bunch of logs, and one of the rows will say something like:

```
> Start Tensorboard: tensorboard --logdir=/path/to/TTS/recipes/saspeech/hifigan/run-March-14-2023_07+07AM-91fd5654
```

In order to look at metrics, open another terminal in the same python virtual environment and run the command from the logs, e.g.

```bash 
$ tensorboard --logdir=/path/to/TTS/recipes/saspeech/hifigan/run-March-14-2023_07+07AM-91fd5654
```

Now that tensorboard is running, you can go to [http://localhost:6006](http://localhost:6006) in your browser to view metrics as training evolves.

Model checkpoints are saved to the same directory, so in our above example, the directory `/path/to/TTS/recipes/saspeech/hifigan/run-March-14-2023_07+07AM-91fd5654` will also contain model checkpoints

### Contact Us

We are Roee Shenberg and Orian Sharoni or in other words Up·AI. If you have any questions or comments, please feel free to contact us using the information below.

| **Orian Sharoni**          | **Roee Shenberg**         |
| ------------------------- | ------------------------- |
| <a href="https://www.linkedin.com/in/orian-sharoni/" target="_blank">Connect on LinkedIn</a> | <a href="https://www.linkedin.com/in/roeeshenberg/" target="_blank">Connect on LinkedIn</a> |
| <a href="https://twitter.com/OrianSharoni" target="_blank">Follow on Twitter</a>       | <a href="https://twitter.com/roeeshenberg" target="_blank">Follow on Twitter</a>       |
| <orian.sharoni@upai.dev>  | <roee.shenberg@upai.dev>  |

The project's <a href="https://discord.gg/t5d2N9gUZH" target="_blank">discord</a> is also available for communication and collaboration.

### Citation
If you use our work, cite us as: Sharoni, O., Shenberg, R., Cooper, E. (2023) SASPEECH: A Hebrew Single Speaker Dataset for Text To Speech and Voice Conversion. Proc. INTERSPEECH 2023, 5566-5570, doi: 10.21437/Interspeech.2023-430

```
@inproceedings{sharoni23_interspeech,
  author={Orian Sharoni and Roee Shenberg and Erica Cooper},
  title={{SASPEECH: A Hebrew Single Speaker Dataset for Text To Speech and Voice Conversion}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={5566--5570},
  doi={10.21437/Interspeech.2023-430}
}
```

