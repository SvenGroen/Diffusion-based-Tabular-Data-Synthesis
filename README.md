![Tests](https://github.com/SvenGroen/Diffusion-based-Tabular-Data-Synthesis/actions/workflows/tests.yml/badge.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fJS9lTVgiaBZq1VlE1SN4aFUZ50GpFAm?usp=sharing)

# Diffusion based Tabular Data Synthesis
This repository allows you to create synthetic tabular data using [TabDDPM](https://github.com/rotot0/tab-ddpm), [CTABGAN](https://github.com/Team-TUD/CTAB-GAN), [CTABGAN+](https://github.com/Team-TUD/CTAB-GAN-Plus), TVAE and SMOTE.

It contains the software code for my [master thesis](https://github.com/SvenGroen/Masterarbeit).
The code is based upon the implementation of [TabDDPM](https://github.com/rotot0/tab-ddpm) and expands their code.

Make sure to have a look at the paper "TabDDPM: Modelling Tabular Data with Diffusion Models" ([paper](https://arxiv.org/abs/2209.15421)).

Additionally, this code makes use of [TabSynDex](https://github.com/vikram2000b/tabsyndex) implementation of the corresponding paper "TabSynDex: A Universal Metric for Robust Evaluation of Synthetic Tabular Data" ([paper](https://arxiv.org/abs/2207.05295)).

## Limitations and known issues:
- So far, I only tested on the "adult" dataset (binary classification).  
Regression or Multiclass-classifications datasets should also work at the current state but I have not tested it yet. It might be the case that this would require some debugging. Let me know if you find some issues.
## Table of Contents
- [Setup](#setup)
  * [Environment and Dataset Setup](#environment-and-dataset-setup)
  * [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
  * [Folder Structure](#folder-structure)
  * [Scripts](#scripts)
- [How to run Experiments](#how-to-run-experiments)
  * [Apply to own Datasets](#apply-to-own-dataset)
- [Tabular Processor](#tabular-processor)
  * [What is a Tabular Processor?](#what-is-a-tabular-processor-)
  * [Adding new Tabular Processing mechanisms](#adding-new-tabular-processing-mechanisms)
- [Appendix](#appendix)
  * [Dataset info.json](#dataset-infojson)
  * [Visualizations](#visualizations)
    + [Pipeline.py](#pipelinepy)
    + [Train, Sample, Eval](#train--sample--eval)
    + [Tune and Eval_seeds.py](#tune-and-eval-seedspy)
  * [Changes made compared to the TabDDPM repository](#changes-made-compared-to-the-tabddpm-repository)
___
Make sure to have a look at the [Google Colab](https://colab.research.google.com/drive/1fJS9lTVgiaBZq1VlE1SN4aFUZ50GpFAm?usp=sharing) for an minimal setup and experiment running example!
___
# Setup

## Environment and Dataset Setup
1. Install [anaconda](https://www.anaconda.com/) (just to manage the environment).
2. clone git repository:

```bash
cd path/to/where/code/will/be/saved

git clone https://github.com/SvenGroen/Diffusion-based-Tabular-Data-Synthesis.git
```
3. Create the conda environment, please run the following as administrator:

```bash
cd path/to/the/github_repo

conda env create -f environment.yml
```

4. activate conda environment and install the package locally:

```bash
cd path/to/the/github_repo

conda activate tabsynth

pip install -e .
```
This will install the `tabsynth` code locally to the conda environment.
Please note, that this code is not meant to be a fully finished pip package.
Instead, it is used to be fully visible within the code and is used to avoid
adding the project folder to the `PYTHONPATH` manually, like in the [original implementation](https://github.com/rotot0/tab-ddpm).

Installing it through pip locally allows easy installation for Windows and Linux users.
Any changes that you make to the code (which is encouraged) will automatically discovered and used as well.

5. Download and Setup the dataset:

The authors of [TabDDPM](https://github.com/rotot0/tab-ddpm) provided some dataset.  
Download them at https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0 and unpack it into `src/tabsynth/data`

Each dataset, must contain the data separated into training, validation and testing dataset, as well as a separation of categorical, numerical and target columns (`X_[cat|num]_[train|val|test].npy` and `y_[train|val|test].npy`)

Additionally, each dataset folder is required to have an `info.json`, for which you have to add a "dataset_config" entry. Have a look at `src/tabsynth/data/adult/info.json` for an example of the "dataset_config" and have a look at [an explanation of the file](#dataset-infojson) below.

If you want to use a dataset other than "adult" and don't know the "dataset_config", have a look at `src/tabsynth/CTABGAN_Plus/columns.json`, which is a good starting point.
___

6. Running the code in Microsoft Azure (OPTIONAL):

If you want to use Azure to run the code, the environment needs some additional packages:

run: 
```bash
# install azure
pip install azure-core, azureml-core

# you might also have to add Microsoft to you conda channels
conda config --env --add channels Microsoft
```

Have a look at `Azure.ipynb`, which contains example code to setup an environment inside azure (`environment_azure.yml`) and shows how to run the different scripts inside azure.
___

## Troubleshooting
If you have any struggles running some of the script make sure to check the following:

1. Run `pip list` and check if all library's are installed correctly (with correct version numbers). 
Also check if `tabsynth` is installed.

2. If you have struggles with path finding, e.g. finding the `data` folder, check `src/tabsynth/lib/variables.py` and may change the ROOT_DIR variable. When running the script locally, I used visual studio code with `path/to/the/github_repo` as my current working directory (cwd) and `ROOT_DIR` pointing to the root directory of the tabsynth library (i.e. the `path/to/the/github_repo/src` folder)


# Project Structure

## Folder Structure
The repository has the following folder structure:
```bash
+---📁outputs                             # will be created in the scripts to save all results 
|   +---📁src 
|   |   +---📁tabsynth
|   |   |   +---📁exp
|   |   |   |   +---📁[dataset_name]        # All tuning expirements for the same dataset will be saved here
|   |   |   |   |   +---📁[experiment_name]  # individual experiments 
+---📁processor_state
+---📁src
|   +---📁tabsynth
|   |   +---📁CTABGAN                     # code for the CTABGAN model
|   |   +---📁CTABGAN_Plus                # code for the CTABGAN_Plus model
|   |   +---📁CTGAN                       # code for the TVAE model (belongs to the "CTGAN" code)
|   |   +---📁data                        # data folder
|   |   |   +---📁[dataset_name]          # individual dataset
|   |   +---📁evaluation                  # contains code for evaluation
|   |   +---📁exp
|   |   |   +---📁[dataset_name]          # contains the exp config.toml for each dataset
|   |   |   +---📁original_exp            # stores the original experiment results from the "TabDDPM" repo
|   |   +---📁lib                         # various utility functions
|   |   +---📁processor_state             # tabular processing states will be saved here (will be created in the script)
|   |   +---📁scripts                     # (Most important!) all scripts for training, sampling, evaluation, etc.
|   |   +---📁smote                       # code for the SMOTE model
|   |   +---📁tabular_processing          # tabular processing implementations
|   |   |   +---📁bgm_utils               # utils for bayesian gaussian mixture
|   |   |   +---📁ft_utils                # utils for feature tokenization
|   |   +---📁tab_ddpm                    # code for the tabular diffusion model
|   |   +---📁tuned_models                # machine learning efficacy models hyperparameters
|   |   |   +---📁catboost                # for catboost, from the "TabDDPM" repo
|   |   |   +---📁mlp                     # for mlp, from the "TabDDPM" repo
+---📁tests                               # testing code
|   +---📁data                            # data for testing
```
## Scripts
The most important scripts are located at the `src/tabsynth/scripts` folder and do the following:

- `pipeline.py`: used to train, sample and evaluate synthetic data for [TabDDPM](https://github.com/rotot0/tab-ddpm) (see [Figure](#pipelinepy)).
The Pipeline script itself calls the `train.py`, `sample.py`, `eval_[catboost|mlp].py` and `eval_similarity.py` (see [Figure](#train-sample-eval)).
- `tune_ddpm.py`: used for hyperparameter tuning for [TabDDPM](https://github.com/rotot0/tab-ddpm) (see [Figure](#tune-and-eval_seedspy)). 
- `eval_seeds.py`: samples multiple datasets and evaluates a trained model for multiple seeds (see [Figure](#tune-and-eval_seedspy)).
- `tune_evaluation_model.py`: allows to find the best hyperparameters for the ML-efficacy models (Catboost or MLP)


# How to run Experiments
><span style="font-size:1.3em;">*I want to generate synthetic data using diffusion model for a specific parameter set* </span>

1. Locate `src/tabsynth/exp/[dataset_name]/config.toml` and set your experiment parameters to your liking.
2. Run:
```
src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/[dataset_name]/config.toml --train --sample --eval
```
&nbsp;&nbsp;&nbsp;&nbsp; you can run the script with just a subset of --train --sample --eval, however,
sampling requires to load some pretrained model, which will be loaded from the `outputs/parent_dir/` (config.toml), so make sure to have a pretrained model saved at this location.
___

><span style="font-size:1.3em;">*I want to find the best hyperparameters for a diffusion model (Recommended for finding the best model)*</span>

1. Locate `src/tabsynth/exp/[dataset_name]/config.toml` and set your experiment parameters to your liking.
Note that the following parameters will be changed and explored during hyperparameter training, so changing them here has no effect:
```
['model_params']['rtdl_params']['d_layers']
['diffusion_params']['num_timesteps']
['train']['main']['steps'] 
['train']['main']['lr']
['train']['main']['weight_decay'] 
['train']['main']['batch_size'] 
['sample']['num_samples'] 
```

Hence, the most important parameters to set are:
```
['train.T']                   # set up any normalization or encoding you want to use
['tabular_processor']['type'] # set up tabular processing type ["identiy"|"bgm"|"ft"]
['eval.type']['eval_model']   # which evaluation model should be used for ML-efficacy ["catboost"|"mlp"]
['eval.type']['eval_type']    # keep "synthetic", so you compare your synthesized data with the real data
['eval.T']                    # any transformations that should be done before the evaluation (best kept as it is)
```

2. Run:
```
src/tabsynth/scripts/tune_ddpm.py [ds_name] [train_size] synthetic [catboost|mlp] [exp_name] --eval_seeds
```
&nbsp;&nbsp;&nbsp;&nbsp;**Explanation**:
```
'[ds_name]'       # needs to be located at "src/tabsynth/ds_name"
'[train_size]'    # to set the sample size for sampling (recommend to set it to the training dataset size)
'synthetic'       # makes sure that we compare the created synthetic dataset with the real test set
'[catboost|mlp]'  # which ML-efficacy model should be used for evaluation ('catboost' recommended)
'[exp_name]'      # name of the experimment (sets folder name in 'outputs'), e.g. ddpm_best
'--eval_seeds'    # runs extensive evaluation for multiple seeds of best found hyperparameter model (recommended)
```
&nbsp;&nbsp;&nbsp;&nbsp;**Example**:
```
src/tabsynth/scripts/tune_ddpm.py "adult" 26048 synthetic "catboost" "ddpm_best" --eval_seeds
```
___
><span style="font-size:1.3em;">*I want to generate synthetic data using the SMOTE/[CTABGAN](https://github.com/Team-TUD/CTAB-GAN)/[CTABGAN+](https://github.com/Team-TUD/CTAB-GAN-Plus)/TVAE model for a specific parameter set*</span>

1. Locate `src/tabsynth/exp/[dataset_name]/[model_name]/config.toml` and set your experiment parameters to your liking.
2. Run:
```
src/tabsynth/model_folder/pipeline_[model_name].py --config src/tabsynth/exp/[dataset_name]/[model_name]/config.toml --train --sample --eval
```
&nbsp;&nbsp;&nbsp;&nbsp; It basically works the same as for the [TabDDPM](https://github.com/rotot0/tab-ddpm) model, but just has a separate pipeline file
___
><span style="font-size:1.3em;">*I want to do the hyperparameter for the SMOTE/[CTABGAN](https://github.com/Team-TUD/CTAB-GAN)/[CTABGAN+](https://github.com/Team-TUD/CTAB-GAN-Plus)/TVAE model*</span>

1. Locate `src/tabsynth/exp/[dataset_name]/[model_name]/config.toml` and set your experiment parameters to your liking.
2. Run:
```
src/tabsynth/model_folder/tune_[model_name].py [data_path] [train_size]
```
&nbsp;&nbsp;&nbsp;&nbsp; It works the same as for the [TabDDPM](https://github.com/rotot0/tab-ddpm) model, but just has a separate tuning file.
___
## Apply to own dataset
><span style="font-size:1.3em;">*I want to use my own dataset and generate synthetic data from it*</span>

To generate synthetic data from your own dataset you need to follow the following steps:

1. Split your dataset into Training, Validation, and Test sets
2. Separate numerical, categorical and the target column (the column which should be predicted in a classification/regression scenario) from each other.
3. Ensure that the data has the right dimensionality:  
    3.1 Numerical and Categorical columns need to be of dimensionality (number_of_rows, number_of_columns)  
    3.2 Target column needs to be of shape (number_of_rows, ); For example: (26048, ) is correct, (26048,1) is not!
4. Convert you variables to numpy arrays, if there are not already
5. Save your array as separate numpy files (.npy) as `"X_[cat|num]_[train|val|test].npy"` and `"y_[train|val|test].npy"`    
(big "X" and small "y"!) at `src/tabsynth/data/[your_dset_name]/`
6. create and save a `info.json` in the same folder (see [Dataset info](#dataset-infojson)!) that stores information on the structure of the data.

Hint: Have a look at [this code](https://github.com/Yura52/tabular-dl-num-embeddings/blob/main/bin/datasets.py) which shows an implementation of how the above procedure can be done for multiple different datasets  
(you don't need the `"idx_[train|val|test].npy"` file for this repository)

Hint2: Copy & paste the above procedure into ChatGPT with a small description of you dataset :smile:
___


# Tabular Processor
## What is a Tabular Processor?
As part of my master thesis, I investigated how the generative capability of the diffusion model [TabDDPM](https://github.com/rotot0/tab-ddpm) changes when processing the tabular data beforehand to account for specific challenges of tabular data.

In principle, a tabular processor is just a classical preprocessing strategy.
This means, the raw data will be encoded by the tabular processor the encoded data will be used to train the diffusion model. The diffusion model will, after training, be used to sample new synthetic data.
Since the diffusion model was trained on encoded data, it will produce synthetic encoded data.
Therefore, the tabular processor needs to decode the encoded data back into its original (human readable) format.

The goal of the master thesis was to extend the already [existing implementation]((https://github.com/rotot0/tab-ddpm)).
Hence, the preprocessing from the original implementation (specified in the config.toml [train.T]) remained untouched and will be executed **AFTER** the tabular processing encoding.

To keep the tabular processing as separate and extendable as possible, the strategy design pattern was chosen:
![Strategy Patters](https://github.com/SvenGroen/Masterarbeit/blob/master/images/strategy.png?raw=true)

In this repository, it is realized inside the `src/tabsynth/tabular_processor` folder.
3 strategies are implemented:
1. "identity": does nothing, can be used to "turn off" tabular processing
2. "bgm": uses the preprocessing strategy of [CTABGAN+](https://github.com/Team-TUD/CTAB-GAN-Plus), which includes a bayesian gausian mixture model (BGM), logarithmic transformations and others.
3. "ft": uses a ["Feature Tokenizeation"](https://github.com/pfnet-research/TabCSDI/blob/d7655578d51b062fefb16656ba635478b458c92d/src/main_model_table_ft.py) approach from the [paper](https://openreview.net/forum?id=4q9kFrXC2Ae) "Diffusion models for missing value imputation in tabular data" , which is basically static embedding of categorical and numerical columns.

Hence, the current implementation looks like:

![](https://github.com/SvenGroen/Masterarbeit/blob/master/images/tabular_processor.png?raw=true)

The TabularDataController controls the context and is responsible for instantiating Tabular Processor instances.
Additionally, the TabularDataController handles loading and saving of the instances, as well as the data.
It also makes sure that no Tabular Processor is fitted on anything else except the training dataset.

## Adding new Tabular Processing mechanisms 

You can easily implement additional processing mechanism by following these steps:

1. Create a class inside `src/tabsynth/tabular_processor/my_processor.py` that inherits from the `TabularProcessor` class from `tabular_processor.py`
2. Implement the required abstract methods ("\_\_init\_\_", "fit", "transform", "inverse_transform")
3. Open `src/tabsynth/tabular_processor/tabular_data_controller`
4. Import your processor class
5. Add `"my_processor":MyProcessor` to the `SUPPORTED_PROCESSORS` dictionary so the script knows where to find the MyProcessor class.
6. Inside your experiment `config.toml`, set `[tabular_processor][type] = "my_processor"` and run your experiment

# Appendix
## Dataset info.json

This file contains the following information ([Adult income](https://archive.ics.uci.edu/ml/datasets/adult) dataset as example):

```
{
    "name": "Adult", // name of the dataset
    "id": "adult--default",
    // What kind of task is the dataset? binary classification (binclass), multiclass classification (multiclass) or regression (regression)
    "task_type": "binclass", 
    "n_num_features": 6, // number of numerical/continuous columns in the dataset
    "n_cat_features": 8, // number of categorical columns in the dataset
    "test_size": 16281, // size of the test dataset split
    "train_size": 26048, // size of the train dataset split
    "val_size": 6513, // size of the validation dataset split
    "dataset_config": { // NEW: required for tabular processing mechanism
        "cat_columns": [ // list of the names of the categorical columns
            "workclass",
            (...),
            "income"
        ],
        "non_cat_columns": [], // only for BGM Processor: categorical columns with a high dimensionality/cardinality
        "log_columns": [], // only for BGM Processor: numerical columns that require log transformation
        "general_columns": [ // only for BGM Processor: columns where "general transform" (GT) from CTABGAN+ (https://arxiv.org/abs/2204.00401) will be applied
            "age"
        ],
        "mixed_columns": { // numerical columns that contain a special categorical value that should be treated as categorical value
            "capital-loss": [ // column_name : special_categorical_value
                0.0
            ],
            "capital-gain": [
                0.0
            ]
        },
        "int_columns": [ // list of the names of the numerical/continuous columns
            "age",
            (...),
            "hours-per-week"
        ],
        "problem_type": "binclass", // equal to task_type (redundancy needs to be fixed in the future)
        "target_column": "income" // name of the target column
    }
}
```
## Visualizations
Green Boxes indicate changes compared to the original implementation of [TabDDPM](https://github.com/rotot0/tab-ddpm)

### Pipeline.py


<img src="https://github.com/SvenGroen/Masterarbeit/blob/master/images/pipeline-CHANGED.png?raw=true" alt="Pipeline script" width="500">



### Train, Sample, Eval

<img src="https://github.com/SvenGroen/Masterarbeit/blob/master/images/train-sample-eval-Changed.png?raw=true" width="500">

### Tune and Eval_seeds.py

<img src="https://github.com/SvenGroen/Masterarbeit/blob/master/images/tune_eval_seeds-CHANGED.png?raw=true">


## Changes made compared to the [TabDDPM](https://github.com/rotot0/tab-ddpm) repository
- separate outputs folder: The experiment results are stored in a separate "outputs" folder. This was required for accessing the results in Azure and makes it easier to find the results locally.
- debug option: some scripts also have a `--debug` flag than can be set, that changes hyperparameters in such a way, that one can quickly go through the whole script without waiting hours. 
- config.toml: added the `[tabular_processor][type]` option. If you don't want to use a tabular_processor, set it to "identity"
- info.json:
every dataset needs to have a `info.json`. Each `info.json` needs to have a `dataset_config` dictionary inside to store information about the dataset properties (see [dataset setup](#dataset-setup))
- tabular processing: added an additional processing mechanism that transforms the data before using it for training.
- evaluation: added [TabSynDex](https://github.com/vikram2000b/tabsyndex) and [Table-Evaluator](https://github.com/Baukebrenninkmeijer/table-evaluator) for an extensive evaluation.
- test folder: contains test code that test functionalities of the project
