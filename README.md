(WORK IN PROGRESS)

# Diffusion based Tabular data synthesis
This repository allows you to create synthetic tabular data using TabDDPM, CTABGAN, CTABGAN+, TVAE and SMOTE.

It contains the software code for my [master thesis](https://github.com/SvenGroen/Masterarbeit).
The code is based upon the implementation of [TabDDPM](https://github.com/rotot0/tab-ddpm) and expands their code.

Make sure to have a look at the paper "TabDDPM: Modelling Tabular Data with Diffusion Models" ([paper](https://arxiv.org/abs/2209.15421)).

Additionally, this code makes use of [TabSynDex](https://github.com/vikram2000b/tabsyndex) implementation of the corresponding paper "TabSynDex: A Universal Metric for Robust Evaluation of Synthetic Tabular Data" ([paper](https://arxiv.org/abs/2207.05295)).

## Table of Contents

1. [Setup](#setup)
2. [Project Structure](#project-structure)
3. [Running Experiments](#running-experiments)
   - [Datasets](#datasets)
   - [File Structure](#file-structure)
   - [Examples](#examples)
   
# Setup
## Environment Setup
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
Any changes that you make to the code (which is encouraged) will automatically discovered any used as well.
___


5. Running the code in Microsoft Azure (OPTIONAL):

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
## Dataset setup
The authors of [TabDDPM](https://github.com/rotot0/tab-ddpm) provided some dataset.
You can download them at https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0 and unpack it into `src/tabsynth/data`

Each dataset, must contain the data separated into training, validation and testing dataset, as well as a separation of categorical, numerical and target columns (`X_[cat|num]_[train|val|test].npy` and `y_[train|val|test].npy`)

Additionally, each dataset folder is required to have an `info.json`.
This file contains the following information ([Adult income](https://archive.ics.uci.edu/ml/datasets/adult) dataset as example):

```json
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

Please ensure that you have to add the `dataset_config` information yourself.

Hint: Have a look at `src/tabsynth/CTABGAN_Plus/columns.json`, for datasets other than "adult", which is a good starting point.


## Troubleshooting
If you have any struggles running some of the script make sure to check the following:

1. Run `pip list` and check if all library's are installed correctly (with correct version numbers). 
Also check if `tabsynth` is installed.

2. If you have struggles with path finding, e.g. finding the `data` folder, check `src/tabsynth/lib/variables.py` and may change the ROOT_DIR variable. When running the script locally, I used visual studio code with `path/to/the/github_repo` as my current working directory (cwd) and `ROOT_DIR` pointing to the root directory of the tabsynth library (i.e. the `path/to/the/github_repo/src` folder)


# Project Structure

## Folder Structure
The repository has the following folder structure:
```bash
+---📁outputs # will be created in the scripts to save all results 
|   +---📁src 
|   |   +---📁tabsynth
|   |   |   +---📁exp
|   |   |   |   +---📁dataset_name # All tuning expirements for the same dataset will be saved here
|   |   |   |   |   +---📁expirment_name # individual experiments 
+---📁processor_state
+---📁src
|   +---📁tabsynth
|   |   +---📁CTABGAN # code for the CTABGAN model
|   |   +---📁CTABGAN_Plus # code for the CTABGAN_Plus model
|   |   +---📁CTGAN # code for the TVAE model (belongs to the "CTGAN" code)
|   |   +---📁data # data folder
|   |   |   +---📁dataset_name # individual dataset
|   |   +---📁evaluation # contains code for evaluation
|   |   +---📁exp
|   |   |   +---📁dataset_name # contains the exp config.toml for each dataset
|   |   |   +---📁original_exp # stores the original experiment results from the "TabDDPM" repo
|   |   +---📁lib # various utility functions
|   |   +---📁processor_state # tabular processing states will be saved here (will be created in the script)
|   |   +---📁scripts # (Most important!) Contains all scripts for training, sampling, evaluation, etc.
|   |   +---📁smote # code for the SMOTE model
|   |   +---📁tabular_processing # tabular processing implementations
|   |   |   +---📁bgm_utils # utils for bayesian gaussian mixture
|   |   |   +---📁ft_utils # utils for feature tokenization
|   |   +---📁tab_ddpm # code for the tabular diffusion model
|   |   +---📁tuned_models # machine learning efficacy models hyperparameters
|   |   |   +---📁catboost # for catboost, from the "TabDDPM" repo
|   |   |   +---📁mlp # for mlp, from the "TabDDPM" repo
+---📁tests # testing code
|   +---📁data # data for testing
```
## How to run Experiments
>"I want to generate synthetic data using diffusion model for a specific parameter set"

1. Locate `src/tabsynth/exp/dataset_name/config.toml` and set your experiment parameters to your likening.
2. Run:
```
src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/dataset_name/config.toml --train --sample --eval
```
&nbsp;&nbsp;&nbsp;&nbsp; you can run the script with just a subset of --train --sample --eval, however,
sampling requires to load some pretrained model, which will be loaded &nbsp;&nbsp;&nbsp;&nbsp;from the `outputs/parent_dir/` (config.toml), so make sure to have a pretrained model saved at this location.
___
>"I want to generate synthetic data using the SMOTE/CTABGAN(+)/TVAE model for a specific parameter set"

1. Locate `src/tabsynth/exp/dataset_name/model_name/config.toml` and set your experiment parameters to your likening.
2. Run:
```
src/tabsynth/model_folder/pipeline_model_name.py --config src/tabsynth/exp/dataset_name/model_name/config.toml --train --sample --eval
```
&nbsp;&nbsp;&nbsp;&nbsp; you can run the script with just a subset of --train --sample --eval, however,
sampling requires to load some pretrained model, which will be loaded &nbsp;&nbsp;&nbsp;&nbsp;from the `outputs/parent_dir/` (config.toml), so make sure to have a pretrained model saved at this location.
___
>"I want to find the best hyperparameters for a diffusion model" (RECOMMENDED) 

1. Locate `src/tabsynth/exp/dataset_name/config.toml` and set your experiment parameters to your likening.
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
src/tabsynth/scripts/tune_ddpm.py "adult" 26048 synthetic "catboost" ddpm_best --eval_seeds
```




___

## Changes made compared to the [TabDDPM](https://github.com/vikram2000b/tabsyndex) repository
- separate outputs folder: The experiment results are stored in a separate "outputs" folder. This was required for accessing the results in Azure and makes it easier to find the results locally.
- debug option: some scripts also have a `--debug` flag than can be set, that changes hyperparameters in such a way, that one can quickly go through the whole script without waiting hours. 
- config.toml: added the `[tabular_processor][type]` option. If you don't want to use a tabular_processor, set it to "identity"
- info.json:
every dataset needs to have a `info.json`. Each `info.json` needs to have a `dataset_config` dictionary inside to store information about the dataset properties (see [dataset setup](#dataset-setup))
- tabular processing: added an additional processing mechanism that transforms the data before using it for training.
- evaluation: added [TabSynDex](https://github.com/vikram2000b/tabsyndex) and [Table-Evaluator](https://github.com/Baukebrenninkmeijer/table-evaluator) for an extensive evaluation.
- test folder: contains test code that test functionalities of the project