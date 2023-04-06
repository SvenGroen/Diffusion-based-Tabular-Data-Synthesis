(WORK IN PROGRESS)

# Diffusion based Tabular data synthesis
This repository contains the software code for my [master thesis](https://github.com/SvenGroen/Masterarbeit)
The code is based upon the implementation of [TabDDPM](https://github.com/rotot0/tab-ddpm) and expands their code.

Make sure to have a look at the paper "TabDDPM: Modelling Tabular Data with Diffusion Models" ([paper](https://arxiv.org/abs/2209.15421)).

Additionally, this code makes use of [TabSynDex](https://github.com/vikram2000b/tabsyndex) implementation of the corresponding paper "TabSynDex: A Universal Metric for Robust Evaluation of Synthetic Tabular Data" ([paper](https://arxiv.org/abs/2207.05295)).

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Project Structure](#project-structure)
3. [Running Experiments](#running-experiments)
   - [Datasets](#datasets)
   - [File Structure](#file-structure)
   - [Examples](#examples)
   
## Environment Setup
### Locally
1. Install [anaconda](https://www.anaconda.com/) (just to manage the env).

2. clone git repository:

```bash
cd path/to/where/code/will/be/saved

git clone https://github.com/SvenGroen/Diffusion-based-Tabular-Data-Synthesis.git
```

2. Create the conda environment, please run the following as administrator:

```bash
cd path/to/the/github_repo

conda env create -f environment.yml
```

3. activate conda environment and install package locally:


```bash
cd path/to/the/github_repo

conda activate tabsynth

pip install -e .
```
this will install the `tabsynth` code locally to the conda environment.
Please note, that this code is not meant to be a fully finished pip package.
Instead, it is used to be fully visible within the code and is used to avoid
adding the project folder to the `PYTHONPATH` manually, like in the [original implementation](https://github.com/rotot0/tab-ddpm).

Installing it through pip locally allows easy installation for Windows and Linux users.
Any changes that you make to the code (which is encouraged) will automatically discovered any used as well.

4. "I want to use the code locally with Microsoft Azure" (OPTIONAL)
If you want to use Azure to run the code (or use Azure.ipynb), the environment needs some additional packages:

run: 
```bash
# install azure
pip install azure-core, azureml-core

# you might also have to add Microsoft to you conda channels
conda config --env --add channels Microsoft
```
Do not use `environment_azure.yml` for installation, as it is just use to create the environment for azure jobs.

## Project Structure
The repository has the following structure:

- tab-ddpm/: Implementation of the proposed method
- tuned_models/: Tuned hyperparameters of evaluation model (CatBoost or MLP)
- scripts/: Contains main scripts for training, sampling, and evaluation
- exp/: Contains experiment configurations, results, and synthetic data
- smote/: Contains the SMOTE baseline
- CTGAN/: Contains the CTGAN (TVAE) baseline
- CTAB-GAN/: Contains the CTAB-GAN baseline
- CTAB-GAN-Plus/: Contains the CTAB-GAN-Plus baseline

For more details on the project structure, refer to the Project Structure section below.

## Running Experiments
### Datasets
Download the datasets used in the paper with the provided train/val/test splits:

```bash
conda activate tddpm
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
tar -xvf data.tar
```

## File Structure

Refer to the Project Structure section for a detailed explanation of the file structure.

## Examples
<ins>Run TabDDPM tuning.</ins>

Template and example (--eval_seeds is optional):