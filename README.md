# HiFiGAN

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/CITATION.cff">
   <img src="https://img.shields.io/badge/cite-this%20repo-purple">
</a>
</p>

## About

This repository contains HW3 from HSE University DLA course - HiFiGAN


## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=baseline.yaml HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (generate wav from text or audio(or rather spectrogram generated from it) and save it):

```bash
python3 inference.py -cn=inference.yaml datasets.test.data_dir="your/path/to/folder" #+datasets.test.is_audio=True for wav files
```
==
To download best model, run the following commmand:
```bash
python3 -m src.utils.download_best_model [path to folder] [link to best model][link to config (may be equal to "")]
```
where link to the best model is [this](https://drive.google.com/uc?export=download&id=1PuVlDWUziIW_w02Xq5E62KrOX1AGb0AQ)

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
