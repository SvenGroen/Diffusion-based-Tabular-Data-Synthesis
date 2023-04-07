''' This have not been changed despite some comments and documentation.'''
# DEBUG START
import os
import sys
import platform
import pkg_resources

def main():
    print_system_info()
    print_current_working_directory()
    print_installed_packages()

def print_system_info():
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print()

def print_current_working_directory():
    cwd = os.getcwd()
    print("Current Working Directory:")
    print(cwd)
    print()

    print("Folders and Files in Current Working Directory:")
    for entry in os.listdir(cwd):
        entry_path = os.path.join(cwd, entry)
        if os.path.isfile(entry_path):
            print(f"File: {entry}")
        elif os.path.isdir(entry_path):
            print(f"Folder: {entry}")
    print()

def print_installed_packages():
    print("Installed Pip Packages:")
    installed_packages = sorted([(d.project_name, d.version) for d in pkg_resources.working_set], key=lambda x: x[0].lower())
    for package_name, package_version in installed_packages:
        print(f"{package_name}=={package_version}")
    print()

if __name__ == "__main__":
    main()

# DEBUG END

import optuna
from tabsynth import lib
import argparse
from tabsynth.scripts.eval_catboost import train_catboost
from tabsynth.scripts.eval_mlp import train_mlp
from pathlib import Path
from tabsynth.lib.variables import ROOT_DIR, RUNS_IN_CLOUD

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('model', type=str)
parser.add_argument('tune_type', type=str)
parser.add_argument('device', type=str)

args = parser.parse_args()
data_path = ROOT_DIR / f"src/tabsynth/data/{args.ds_name}"
best_params = None 

assert args.tune_type in ("cv", "val")

def _suggest(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    return getattr(trial, f'suggest_{distribution}')(label, *args)

def _suggest_optional(trial: optuna.trial.Trial, distribution: str, label: str, *args):
    if trial.suggest_categorical(f"optional_{label}", [True, False]):
        return _suggest(trial, distribution, label, *args)
    else:
        return 0.0

def _suggest_mlp_layers(trial: optuna.trial.Trial, mlp_d_layers: list[int]):
    """
    Suggests the number of neurons in each layer of a multi-layer perceptron network.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The current trial object.
    mlp_d_layers : list[int]
        A list containing the minimum and maximum number of layers that the network can have,
        as well as the minimum and maximum number of neurons per layer that can be suggested.

    Returns
    -------
    list[int]
        A list containing the suggested number of neurons in each layer of the MLP.
    """
    min_n_layers, max_n_layers = mlp_d_layers[0], mlp_d_layers[1]
    d_min, d_max = mlp_d_layers[2], mlp_d_layers[3]

    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t


    n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last

    return d_layers

def suggest_mlp_params(trial):
    """
    Generates a dictionary of hyperparameters for a multilayer perceptron (MLP) 
    model by suggesting values for each parameter using Optuna's suggest functions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        An Optuna trial instance that provides the suggest methods for the hyperparameters.
        
    Returns
    -------
    dict
        A dictionary of hyperparameters for the MLP model, with the following keys:
        
        - 'lr' : float
            The learning rate of the optimizer.
        - 'dropout' : float
            The dropout rate applied to the input layer and each hidden layer.
        - 'weight_decay' : float
            The weight decay (L2 penalty) applied to all weights.
        - 'd_layers' : list[int]
            The dimensions of each layer in the MLP model.

    """
    params = {}
    params["lr"] = trial.suggest_loguniform("lr", 5e-5, 0.005)
    params["dropout"] = _suggest_optional(trial, "uniform", "dropout", 0.0, 0.5)
    params["weight_decay"] = _suggest_optional(trial, "loguniform", "weight_decay", 1e-6, 1e-2)
    params["d_layers"] = _suggest_mlp_layers(trial, [1, 8, 6, 10])

    return params

def suggest_catboost_params(trial):
    """
        Generates a dictionary of hyperparameters for a CatBoost model by suggesting values for each parameter using Optuna's suggest functions.

    Parameters
    ----------
    trial : optuna.trial.Trial
        An Optuna trial instance that provides the suggest methods for the hyperparameters.
        
    Returns
    -------
    dict
        A dictionary of hyperparameters for the CatBoost model, with the following keys:
        
        - 'learning_rate' : float
            The learning rate of the model.
        - 'depth' : int
            The depth of the tree.
        - 'l2_leaf_reg' : float
            The L2 regularization coefficient.
        - 'bagging_temperature' : float
            The temperature parameter for Bayesian bootstrap.
        - 'leaf_estimation_iterations' : int
            The number of gradient steps used to build a new leaf.
        - 'iterations' : int
            The maximum number of iterations to run.
        - 'early_stopping_rounds' : int
            The number of early stopping rounds to use.
        - 'od_pval' : float
            The p-value threshold for early stopping.
        - 'task_type' : str
            The task type (CPU or GPU).
        - 'thread_count' : int
            The number of threads to use.

    """
    params = {}
    params["learning_rate"] = trial.suggest_loguniform("learning_rate", 0.001, 1.0)
    params["depth"] = trial.suggest_int("depth", 3, 10)
    params["l2_leaf_reg"] = trial.suggest_uniform("l2_leaf_reg", 0.1, 10.0)
    params["bagging_temperature"] = trial.suggest_uniform("bagging_temperature", 0.0, 1.0)
    params["leaf_estimation_iterations"] = trial.suggest_int("leaf_estimation_iterations", 1, 10)

    params = params | {
        "iterations": 2000,
        "early_stopping_rounds": 50,
        "od_pval": 0.001,
        "task_type": "CPU", # "GPU", may affect performance
        "thread_count": 4,
        # "devices": "0", # for GPU
    }

    return params

def objective(trial: optuna.trial.Trial) -> float:
    """
    Train and evaluate a model with the given parameters on the specified dataset.

    Parameters
    ----------
    trial : optuna.trial.Trial
        An Optuna `Trial` object that provides the parameter values for each trial.

    Returns
    -------
    float
        The mean validation F1 score of the trained model.
    """
    if args.model == "mlp":
        params = suggest_mlp_params(trial)
        train_func = train_mlp
        T_dict = {
            "seed": 0,
            "normalization": "quantile", # maybe none later
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": "one-hot",
            "y_policy": "default"
        }
    else:
        params = suggest_catboost_params(trial)
        train_func = train_catboost
        T_dict = {
            "seed": 0,
            "normalization": None,
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default"
        }
    trial.set_user_attr("params", params)
    if args.tune_type == "cv":
        score = 0.0
        for fold in range(5):
            val_m1, val_m2, test_m1, test_m2 = train_func(
                parent_dir=None,
                real_data_path=data_path / f"kfolds/{fold}",
                eval_type="real",
                T_dict=T_dict,
                params=params,
                change_val=False,
                device=args.device
            )
            score += val_m1
        score /= 5

    elif args.tune_type == "val":
        # val_m1, val_m2, test_m1, test_m2 = train_func(
        #     parent_dir=None,
        #     real_data_path=data_path,
        #     eval_type="real",
        #     T_dict=T_dict,
        #     params=params,
        #     change_val=False,
        #     device=args.device
        # )
        # score = val_m2
        score = train_func(
            parent_dir=None,
            real_data_path=data_path,
            eval_type="real",
            T_dict=T_dict,
            params=params,
            change_val=False,
            device=args.device
        ).get_metric(split="val", metric="f1")


    return score

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=100, show_progress_bar=True)
    
bets_params = study.best_trial.user_attrs['params']

best_params_path = ROOT_DIR / f"src/tabsynth/tuned_models/{args.model}/{args.ds_name}_{args.tune_type}.json"
if RUNS_IN_CLOUD:
    output_path = Path("outputs").mkdir(parents=True, exist_ok=True)
    best_params_path = Path("outputs") / best_params_path

lib.dump_json(bets_params, best_params_path)