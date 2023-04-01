from pathlib import Path
import tomli
import shutil
import os
import argparse
from train import train
from sample import sample
from eval_similarity import calculate_similarity_score
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
from lib.util import RUNS_IN_CLOUD
import pandas as pd
import matplotlib.pyplot as plt
import zero
import lib
import torch


def load_config(path) :
    """
    Load the TOML configuration file.

    Parameters
    ----------
    path : str
        Path to the configuration file.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.
    """
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    """
    Save the configuration file in the parent directory.

    Parameters
    ----------
    parent_dir : str
        The parent directory where the configuration file will be saved.
    config_path : str
        The path of the configuration file.
    """
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main():
    """
    Main function to execute training, sampling, and evaluation.

    Parses command-line arguments, loads configuration, and calls appropriate functions based on the arguments.

    Command-line Arguments
    ----------------------
    --config FILE : str
        Path to the TOML configuration file.
    --train : bool, optional
        If specified, executes the training process.
    --sample : bool, optional
        If specified, executes the sampling process.
    --eval : bool, optional
        If specified, executes the evaluation process.
    --change_val : bool, optional
        If specified, changes the validation set used during training.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()


    raw_config = lib.load_config(args.config)

    # for debugging, change raw_config["device"] to "cuda:0" if you want to use GPU
    if not RUNS_IN_CLOUD:
        # for debugging
        raw_config["diffusion_params"]["num_timesteps"] = 10
        raw_config["train"]["main"]["steps"] = 2 
        raw_config["device"] = "cpu"
        raw_config['sample']['num_samples'] = 10000 # needs to be large enough to cover all classes
        # raw_config["sample"]["disbalance"] = "fill"

    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cuda:1')
    if RUNS_IN_CLOUD and not "outputs" in raw_config["parent_dir"]:
        raw_config["parent_dir"] = os.path.join('outputs', raw_config["parent_dir"])
    


    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    if args.train:
        train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=args.change_val,
            processor_type=raw_config["tabular_processor"]["type"],
        )
    if args.sample:
        sample(
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=args.change_val,
            processor_type=raw_config["tabular_processor"]["type"]
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        elif raw_config['eval']['type']['eval_model'] == 'mlp':
            train_mlp(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                device=device
            )
        elif raw_config['eval']['type']['eval_model'] == 'simple':
            train_simple(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val
            )
        calculate_similarity_score(
                parent_dir=raw_config['parent_dir'],
                real_data_path=raw_config['real_data_path'],
                eval_type=raw_config['eval']['type']['eval_type'],
                num_classes=raw_config['model_params']['num_classes'],
                is_y_cond=raw_config['model_params']['is_y_cond'],
                T_dict=raw_config['eval']['T'],
                seed=raw_config['seed'],
                change_val=args.change_val,
                table_evaluate=not args.change_val # do not evaluate during the tuning process
            )

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()