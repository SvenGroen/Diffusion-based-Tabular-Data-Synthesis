import argparse
import subprocess
import tempfile
from tabsynth import lib
import os
import sys
import shutil
from pathlib import Path
from copy import deepcopy
from tabsynth.scripts.eval_catboost import train_catboost
from tabsynth.scripts.eval_mlp import train_mlp
from tabsynth.scripts.eval_simple import train_simple
from tabsynth.scripts.eval_similarity import calculate_similarity_score
# from azureml.core import Run

pipeline = {
    'ddpm': 'src/tabsynth/scripts/pipeline.py',
    'smote': 'src/tabsynth/smote/pipeline_smote.py',
    'ctabgan': 'src/tabsynth/CTAB-GAN/pipeline_ctabgan.py',
    'ctabgan-plus': 'src/tabsynth/CTAB-GAN-Plus/pipeline_ctabganp.py',
    'tvae': 'src/tabsynth/CTGAN/pipeline_tvae.py'
}
# added: create a copy of the environment variables for running the subprocess
my_env = os.environ.copy()
my_env["PYTHONPATH"] = os.getcwd() # Needed to run the subscripts

def eval_seeds(
    raw_config,
    n_seeds,
    eval_type,
    sampling_method="ddpm",
    model_type="catboost",
    n_datasets=1,
    dump=True,
    change_val=False
):
    """
    Evaluate various models on real and/or synthetic data using different seeds and save the results.

    This function evaluates various models (CatBoost and MLP) on real data, synthetic data, or a combination of both. 
    It trains and evaluates the models for a given number of seeds and calculates the mean and standard deviation of the evaluation metrics. 
    The function also saves the evaluation results to a JSON file and logs the results.

    Parameters
    ----------
    raw_config : dict
        Configuration dictionary containing paths, sampling method, and other settings.
    n_seeds : int
        Number of seeds for training and evaluation.
    eval_type : str
        Evaluation type: 'real', 'synthetic', or 'merged'.
    sampling_method : str, optional, default="ddpm"
        Sampling method to generate synthetic data: "ddpm", "smote", "ctabgan", "ctabgan-plus", or "tvae".
    model_type : str, optional, default="catboost"
        Model type for evaluation: "catboost" or "mlp".
    n_datasets : int, optional, default=1
        Number of datasets to sample from.
    dump : bool, optional, default=True
        If True, saves the evaluation results to a JSON file.
    change_val : bool, optional, default=False
        If True, changes the validation dataset.

    Returns
    -------
    dict
        Dictionary containing the evaluation results for each model and metric.

    """
    # for Azure ML
    # run = Run.get_context()

    # create a directory for the final evaluation results
    metrics_seeds_report = lib.SeedsMetricsReport()
    parent_dir = Path(raw_config["parent_dir"])
    
    # added: create separate directory for final evaluation results
    if not "outputs" in str(parent_dir):
        final_eval_dir = Path("outputs") / parent_dir / "final_eval"
    else:
        final_eval_dir = parent_dir / "final_eval"
    # create directory if it doesn't exist
    final_eval_dir.mkdir(exist_ok=True, parents=True)
    print("Final Evaluation directory located at: ", str(final_eval_dir))

    # load dataset info.json
    d_set_info = lib.load_json(parent_dir/"info.json")
    if eval_type == 'real':
        n_datasets = 1

    # create a temporary directory to store files
    temp_config = deepcopy(raw_config)
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        temp_config["parent_dir"] = str(dir_)
        # copy model.pt to temp directory
        if sampling_method == "ddpm":
            shutil.copy2(parent_dir / "model.pt", temp_config["parent_dir"])
        elif sampling_method in ["ctabgan", "ctabgan-plus"]:
            shutil.copy2(parent_dir / "ctabgan.obj", temp_config["parent_dir"])
        elif sampling_method == "tvae":
            shutil.copy2(parent_dir / "tvae.obj", temp_config["parent_dir"])

        # create sample_seed's datasets
        for sample_seed in range(n_datasets):
            temp_config['sample']['seed'] = sample_seed
            lib.dump_config(temp_config, dir_ / "config.toml")
            if eval_type != 'real' and n_datasets > 1:
                # Sample data
                print(f"---STARTING SAMPLING {sample_seed}---")
                # added: sys.executable to run the subprocess with the same python version as the current script
                # added: my_env to make sure that env variables are passed to the subprocess
                # Note: would probably be better to just import the functions from the scripts instead of running them as subprocesses
                subprocess.run([sys.executable, f'{pipeline[sampling_method]}', '--config', f'{str(dir_ / "config.toml")}', '--sample'], check=True, env=my_env)
                print(f"---Finished SAMPLING {sample_seed}---")
            T_dict = deepcopy(raw_config['eval']['T'])
            sim_reports = []
            best_sim_score = 0

            # for each dataset, evaluate the datasets using different seeds
            for seed in range(n_seeds):
                print(f'\n**Eval Iter: {sample_seed*n_seeds + (seed + 1)}/{n_seeds * n_datasets}**')
                print(f"Model_type: {model_type}")

                # machine learning efficacy with catboost or mlp
                if model_type == "catboost":
                    T_dict["normalization"] = None
                    T_dict["cat_encoding"] = None
                    metric_report = train_catboost(
                        parent_dir=temp_config['parent_dir'],
                        real_data_path=temp_config['real_data_path'],
                        eval_type=eval_type,
                        T_dict=T_dict,
                        seed=seed,
                        change_val=change_val
                    )
                elif model_type == "mlp":
                    T_dict["normalization"] = "quantile"
                    T_dict["cat_encoding"] = "one-hot"
                    metric_report = train_mlp(
                        parent_dir=temp_config['parent_dir'],
                        real_data_path=temp_config['real_data_path'],
                        eval_type=eval_type,
                        T_dict=T_dict,
                        seed=seed,
                        change_val=change_val
                    )
                try:
                    num_classes = raw_config['model_params']['num_classes']
                    y_cond = raw_config['model_params']['is_y_cond']
                except KeyError:
                    num_classes = get_num_classes(d_set_info["dataset_config"]["problem_type"])
                    y_cond = True

                # Similarity score
                print("calculating similarity score")
                similarity_score = calculate_similarity_score(
                    parent_dir=temp_config['parent_dir'],
                    real_data_path=temp_config['real_data_path'],
                    eval_type=eval_type,
                    T_dict=T_dict,
                    seed=seed,
                    num_classes=num_classes,
                    is_y_cond=y_cond,
                    change_val=change_val,
                    table_evaluate=True#True 
                )
                # save the best similarity score
                if similarity_score["sim_score"]["score"] > best_sim_score:
                    best_sim_score = similarity_score["sim_score"]["score"]
                    # copy content to final_eval_dir
                    if dump:
                        # copy content with sub folders to final_eval_dir
                        for file in os.listdir(temp_config['parent_dir']):
                            origin = os.path.join(temp_config['parent_dir'], file)
                            source = os.path.join(final_eval_dir, file)
                            copy_file_or_tree(origin, source)

                sim_reports.append(similarity_score["sim_score"])              
                print("**Finished Evaluation Iteration**\n")
                metrics_seeds_report.add_report(metric_report)
    # calculate and print average similarity score
    print("Final result: ")
    sim_reports = lib.average_per_key(sim_reports)
    for k, v in sim_reports.items():
        # run.log("final_"+str(k), v)
        print(f"final_{k}: {v}")
    metrics_seeds_report.get_mean_std()
    res = metrics_seeds_report.print_result()
    if os.path.exists(final_eval_dir/ f"eval_{model_type}.json"):
        eval_dict = lib.load_json(final_eval_dir / f"eval_{model_type}.json")
        eval_dict = eval_dict | {eval_type: res}
    else:
        eval_dict = {eval_type: res}
    
    if dump:
        lib.dump_json(sim_reports, final_eval_dir / f"eval_similarity.json")
        lib.dump_json(eval_dict, final_eval_dir / f"eval_{model_type}.json")

    raw_config['sample']['seed'] = 0
    lib.dump_config(raw_config, parent_dir / 'config.toml')
    return res

def main():
    """
    The main function that parses command-line arguments, loads the configuration file, and calls the `eval_seeds`
    function to evaluate various models on real and/or synthetic data using different seeds.

    This function serves as the entry point of the script. It takes command-line arguments for the configuration file,
    the number of seeds, the sampling method, the evaluation type, the model type, the number of datasets, and a flag
    to control whether to dump the evaluation results or not. It then loads the configuration and passes these
    parameters to the `eval_seeds` function for the actual evaluation process.

    Command-line Arguments
    ----------------------
    --config : str
        Path to the configuration file.
    n_seeds : int
        Number of seeds for training and evaluation.
    sampling_method : str
        Sampling method to generate synthetic data: "ddpm", "smote", "ctabgan", "ctabgan-plus", or "tvae".
    eval_type : str
        Evaluation type: 'real', 'synthetic', or 'merged'.
    model_type : str
        Model type for evaluation: "catboost" or "mlp".
    n_datasets : int
        Number of datasets to sample from.
    --no_dump : bool, optional
        If set, the evaluation results will not be saved to a JSON file.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('n_seeds', type=int, default=10)
    parser.add_argument('sampling_method', type=str, default="ddpm")
    parser.add_argument('eval_type',  type=str, default='synthetic')
    parser.add_argument('model_type',  type=str, default='catboost')
    parser.add_argument('n_datasets', type=int, default=1)
    parser.add_argument('--no_dump', action='store_false',  default=True)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    eval_seeds(
        raw_config,
        n_seeds=args.n_seeds,
        sampling_method=args.sampling_method,
        eval_type=args.eval_type,
        model_type=args.model_type,
        n_datasets=args.n_datasets,
        dump=args.no_dump
    )

def copy_file_or_tree(origin, source):
    if not os.path.isdir(origin):
        if os.path.exists(source):
            os.remove(source)
        shutil.copy2(origin, source)
    else:
        if os.path.exists(source):
            shutil.rmtree(source)
        shutil.copytree(origin, source)

def get_num_classes(problem_type):
    if problem_type == "binclass":
        return 2
    elif problem_type == "regression":
        return 0
    else:
        return 3

if __name__ == '__main__':
    main()