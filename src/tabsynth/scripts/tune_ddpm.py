import subprocess
from tabsynth import lib
# import lib
import os
import sys
import optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path
# from azureml.core import Run
import pprint
from tabsynth.lib.variables import ROOT_DIR, RUNS_IN_CLOUD

"""
Runs an optimization using Optuna to find the best hyperparameters for a deep generative model.

Arguments:
----------
ds_name (str) : The name of the dataset.
train_size (int) : The size of the training set.
eval_type (str) : The evaluation type. Can be 'merged' or 'synthetic'.
eval_model (str) : The type of the evaluation model. Can be 'mlp' or 'resnet'.
prefix (str) : The prefix to add to the best experiment directory.
--eval_seeds (bool) : Whether to evaluate the seeds of the best experiment or not (default False).
--debug (bool) : Whether to run in debug mode or not (default False).
--optimize_sim_score (bool) : Whether to optimize for similarity score instead of model score (default False).

""" 

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('eval_model', type=str)
parser.add_argument('prefix', type=str)
parser.add_argument('--eval_seeds', action='store_true',  default=False)
parser.add_argument("--debug", action='store_true', default=False)
parser.add_argument("--optimize_sim_score", action='store_true', default=False)
# run = Run.get_context()
args = parser.parse_args()
if args.debug:
    print("--->DEBUG MODE IS ON<---")
train_size = args.train_size
ds_name = args.ds_name
eval_type = args.eval_type 
assert eval_type in ('merged', 'synthetic')
prefix = str(args.prefix)
pipeline = ROOT_DIR / f'tabsynth/scripts/pipeline.py'
base_config_path = ROOT_DIR / f'tabsynth/exp/{ds_name}/config.toml'
parent_path = ROOT_DIR / f'tabsynth/exp/{ds_name}/'
exps_path = ROOT_DIR / f'tabsynth/exp/{ds_name}/many-exps/' # temporary dir. maybe will be replaced with tempdiвdr
if RUNS_IN_CLOUD and not "outputs" in str(parent_path):
    parent_path = 'outputs' / parent_path
    exps_path = 'outputs' / exps_path
eval_seeds = ROOT_DIR / f'tabsynth/scripts/eval_seeds.py'

my_env = os.environ.copy()
my_env["PYTHONPATH"] = os.getcwd() # Needed to run the subscripts

os.makedirs(exps_path, exist_ok=True)

print("parent_path: ", parent_path)
print("exps_path: ", exps_path)


def _suggest_mlp_layers(trial):
    """
    Suggests the number of layers and the dimensions of an MLP (multi-layer perceptron) for a given Optuna trial.

    The function first defines the range of the number of layers, as well as the minimum and maximum dimensions of each
    layer, as exponential base 2. Then, it uses the `trial.suggest_int()` method to sample an integer between `d_min`
    and `d_max`, inclusive, for each layer. It constructs the list of layer dimensions by concatenating the first,
    middle, and last layers' dimensions. The middle layer dimensions are repeated `n_layers - 2` times, where `n_layers`
    is the total number of layers. The function returns the list of layer dimensions.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial.

    Returns
    -------
    list of int
        The list of dimensions of each layer of an MLP, where each dimension is a power of 2.
    """
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):
    """
    Objective function for Optuna. This function defines the search space for the optimization, and computes the
    score to optimize, using the hyperparameters suggested by Optuna.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna's `Trial` object representing the current trial.

    Returns
    -------
    float
        The score to optimize, as a float value. The higher the score, the better the hyperparameters.
    """

    # Hyperparameter search space definition
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    d_layers = _suggest_mlp_layers(trial)
    weight_decay = 0.0    
    batch_size = trial.suggest_categorical('batch_size', [256, 4096])
    steps = trial.suggest_categorical('steps', [5000, 20000, 30000])
    # steps = trial.suggest_categorical('steps', [500]) # for debug
    gaussian_loss_type = 'mse'
    # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 1000])
    num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))

    # load config and overwrite the hyperparameters with the suggested values from Optuna
    base_config = lib.load_config(base_config_path)
    print("BASE CONFIG: ")
    pprint.pprint(base_config, width=-1)
    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = weight_decay
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['eval']['type']['eval_type'] = eval_type
    base_config['sample']['num_samples'] = num_samples
    base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    # base_config['diffusion_params']['scheduler'] = scheduler

    # for debug use only small numbers so that the experiment runs faster
    if args.debug:
        base_config['train']['main']['steps'] = 50
        base_config['train']['main']['batch_size'] = 256
        base_config['diffusion_params']['num_timesteps'] = 10
        num_samples = 100

    # set the experiment directory
    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    base_config['eval']['type']['eval_model'] = args.eval_model
    if args.eval_model == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"

        base_config['eval']['T']['cat_encoding'] = "one-hot"
    trial.set_user_attr("config", base_config)
    # save the config to the experiment directory
    lib.dump_config(base_config, exps_path / 'config.toml')

    # run the pipeline with the overwritten config
    try:
        # Added: sys.executable to use the same python version as the one used to run the script, 
        # otherwise it will use the default python version and might cause errors
        # Additionally, added my_env to use the same environment variables as the one used to run the script
        subprocess.run([sys.executable, f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--change_val'], check=True, env=my_env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    print("----->FINISHED to run pipeline-TRAIN from tune_ddpm<--------: ")
    
    # Start sampling
    n_datasets = 5 if not args.debug else 1
    score = 0.0
    sim_score = []
    # sample 5 datasets and compute the score for each one
    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        lib.dump_config(base_config, exps_path / 'config.toml')
        print("--------------------->SAMPLE SEED: ", sample_seed, "<---------------------")
        # sample and evaluate synthetic data with the trained model with the overwritten config
        # Changes: see above
        try:
            subprocess.run([sys.executable, f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval', '--change_val'], check=True,  env=my_env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        
        # load the evaluation report and compute the score
        report_path = str(Path(base_config['parent_dir']) / f'results_{args.eval_model}.json')
        report = lib.load_json(report_path)
        sim_path = str(Path(base_config['parent_dir']) / f'results_similarity.json')
        sim_report = lib.load_json(sim_path)
        sim_score.append(sim_report['sim_score'])
        if 'r2' in report['metrics']['val']:
            score += report['metrics']['val']['r2']
        else:
            score += report['metrics']['val']['macro avg']['f1-score']

    # remove tmp experiment directory
    shutil.rmtree(exps_path / f"{trial.number}")
    
    # calculate the average score
    print(f"Average similarity results:")
    for k, v in lib.average_per_key(sim_score).items():
        # run.log(k, v)
        print(f"{k}: {v}")

    print(f"ML - Score calculated: {score / n_datasets}")
    
    # return the average evaluation score, similarity score if args.optimize_sim_score is True, else ML-efficacy score
    if not args.optimize_sim_score:
        print(f"optimizing for {args.eval_model} score")
        return score / n_datasets
    else:
        print("optimizing for similarity score")
        return lib.average_per_key(sim_score)['score-mean']

# setup the Optuna study
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

print("---Starting optimizing Optune run---")
# 50 optuna trials
n_trials=50

if args.debug:
    n_trials=6 # for debug use only small numbers so that the experiment runs faster, 
    print(f"DEBUG MODE IS ON: Only Running {n_trials} Optuna trials")

# run the optimization
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
print("---Finished optimizing Optune run---")

# load the best config 
best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']

print("best_config_path: ", best_config_path)
print("Best config found with: ")
print(best_config)
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_config(best_config, best_config_path)
lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')

# Final evaluation with best found parameters,
# changes: see above
# First, train the model with the best config
# Second, sample synthetic dataset with the trained model
# Third, evaluate the synthetic dataset for multiple seeds in eval_seeds.py
try:
    subprocess.run([sys.executable, f'{pipeline}', '--config', f'{best_config_path}', '--train', '--sample'], check=True, env=my_env)
except subprocess.CalledProcessError as e:
    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

if args.eval_seeds:
    best_exp = str(parent_path / f'{prefix}_best/config.toml')
    print("---Starting eval_seeds.py---")
    try:
        sample_runs = 10 if not args.debug else 2
        subprocess.run([sys.executable, f'{eval_seeds}', '--config', f'{best_exp}', f'{sample_runs}', "ddpm", eval_type, args.eval_model, '5'], check=True, env=my_env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    print("---Finished eval_seeds.py---")