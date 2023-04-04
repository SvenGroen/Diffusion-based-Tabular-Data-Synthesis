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
### Windows
1. Install [anaconda](https://www.anaconda.com/) (just to manage the env).

'''
"env" : {
    "PYTHONPATH": "${workspaceFolder}"
},
'''

### Linux
1. Install [anaconda](https://www.anaconda.com/) (just to manage the env).
2. Run the following commands to set up the environment:

```bash
export REPO_DIR=/path/to/the/code
cd $REPO_DIR

conda create -n tddpm python=3.9.7
conda activate tddpm

pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
conda env config vars set PROJECT_DIR=${REPO_DIR}

conda deactivate
conda activate tddpm
```

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