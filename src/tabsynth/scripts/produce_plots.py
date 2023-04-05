''' 
This script loads multiple experiment results and generates plots for each method.
It can be used to create plots after all experiments have been executed by loading the respective synthetic data and real data from the experiment folders.
The plots are saved in a designated output directory.
'''


from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import classification_report, r2_score
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import zero
from pathlib import Path
from tabsynth import  lib
from lib import concat_features, read_pure_data, get_catboost_config, read_changed_val
from tabular_processing.tabular_processor import TabularProcessor
import json
from evaluation.tabsyndex import tabsyndex
from tabular_processing.tabular_data_controller import TabularDataController
import time
import os
from tabsynth import  lib
import numpy as np
import pandas as pd
import json
from pathlib import Path


def main():
    """
    Executes the main script to produce plots for different methods in a comparative analysis.
    
    This script loads multiple experiment results, applies transformations to the data, and generates plots for each method. 
    The plots are saved in a designated output directory.
    """
    base_path = json.load(open("secrets.json", "r"))["Experiment_Folder"]

    # Define the output directory of the experiments for the different models.
    # Todo: Rename the output directories
    method2exp = {
        "trash": "adult/21_12_2022-identity-50optuna-ts26048-catboost-tune-CatboostAndSimilarityEval-syntheticEval/outputs/exp/adult/ddpm_identity_best/final_eval/", # for some reason the plots did look different in the first run, should be fixed for continouse columns but not for categorical columns yet.
        "real": "adult/20_12_2022-REAL-BASELINE/outputs/exp/adult/ddpm_real/final_eval/",
        "tab-ddpm": "adult/21_12_2022-identity-50optuna-ts26048-catboost-tune-CatboostAndSimilarityEval-syntheticEval/outputs/exp/adult/ddpm_identity_best/final_eval/",
        "tab-ddpm-bgm": "adult/20_12_2022-bgm-50optuna-ts26048-catboost-tune-CatboostAndSimilarityEval-syntheticEval/outputs/exp/adult/ddpm_bgm_best/final_eval/",
        "tab-ddpm-simTune": "adult/12_02_2023-identity_sim_tune-50optuna-ts26048-catboost-tune-CatboostAndSimilarityEval-syntheticEval/outputs/exp/adult/ddpm_identity_sim_tune_best/final_eval/",
        "tab-ddpm-bgm-simTune" : "adult/12_02_2023-bgm_sim_tune-50optuna-ts26048-catboost-tune-CatboostAndSimilarityEval-syntheticEval/outputs/exp/adult/ddpm_bgm_sim_tune_best/final_eval/",
        "tab-ddpm-simTune-minmax": "adult/02_03_2023-identity_sim_tune_min_max/outputs/exp/adult/ddpm_identity_sim_tune_minmax_best/final_eval/",
        "tab-ddpm-bgm-simTune-minmax": "adult/02_03_2023-bgm_sim_tune_min_max/outputs/exp/adult/ddpm_bgm_sim_tune_minmax_best/final_eval/",
        "tab-ddpm-ft" : "adult/08_02_2023-ft-50optuna-ts26048-catboost-tune-CatboostAndSimilarityEval-syntheticEval/outputs/exp/adult/ddpm_ft_best/final_eval/",
        "tab-ddpm-ft-simTune": "adult/15_03_2023-ft-tabddpm-SimTune/outputs/exp/adult/ddpm_ft_sim_tune_quantile_best/final_eval/",
        "smote": "adult/02_01_2023-identity-SMOTE/outputs/exp/adult/smote/final_eval/",
        "ctabgan+": "adult/05_01_2023-identity-CTABGAN-Plus/outputs/exp/adult/ctabgan-plus/final_eval/",
        "ctabgan": "adult/02_01_2023-identity-CTABGAN/outputs/exp/adult/ctabgan/final_eval/",
        "ctabgan_simTune": "adult/08_03_2023-identity-CTABGAN-simtune/outputs/exp/adult/ctabgan/final_eval",
        "tvae": "adult/03_01_2023-identity-TVAE/outputs/exp/adult/tvae/final_eval/",
        "tvae_simTune": "adult/13_02_2023-TVAE_sim_tune-identity-50optuna-ts26048-catboost-tune-CatboostAndSimilarityEval-syntheticEval/outputs/exp/adult/tvae/final_eval",
    } 
    for k,v in method2exp.items():
        method2exp[k] = Path(os.path.join(base_path, v))

    out_dir = Path(os.path.join(base_path,"changed_plots"))
    out_dir.mkdir(exist_ok=True, parents=True)

    i=0
    # for name, path create plots for each method
    for name, path in method2exp.items():
        raw_config = lib.load_config(path / "config.toml")
        out = out_dir / name
        if name == "real":
            raw_config["tabular_processing"]= "identity"
            raw_config["eval"]["type"]['eval_type'] = "real"

        produce_plots(
                    parent_dir=path,
                    real_data_path=raw_config['real_data_path'],
                    eval_type=raw_config['eval']['type']['eval_type'],
                    num_classes=2,
                    T_dict=raw_config['eval']['T'],
                    seed=raw_config['seed'],
                    save_dir=out
                    )
        i += 1
        # if i==4:
        #     break

def produce_plots(    
    parent_dir,
    real_data_path,
    eval_type,
    T_dict,
    seed = 0,
    num_classes = 2,
    save_dir = None,
    ):  
    """
    Produces plots comparing the similarity between real and synthetic data for different methods.
    
    Parameters
    ----------
    parent_dir : str
        Path to the parent directory of the experiment.
    real_data_path : str
        Path to the real data directory.
    eval_type : str
        Type of evaluation to be performed. Choices are 'real' or 'synthetic'.
        "real" means that the real training data is used for evaluation.
        "synthetic" means that the synthetic training data is used for evaluation.
        Either "real" or "synthetic" will be compared to the "real" test data. 
    T_dict : dict
        Dictionary containing transformation configurations.
    seed : int, optional
        Random seed for reproducibility, by default 0.
    num_classes : int, optional
        Number of classes in the target variable, by default 2.
    save_dir : str, optional
        Path to the directory where plots will be saved, by default None.
    """
    print("Starting Similarity Evaluation")
    zero.improve_reproducibility(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)
        # info.json is not always copied to synthetic_data_path but is needed for tabular transformer (tvae tune for example)
        if not "info.json" in os.listdir(synthetic_data_path):
            try:
                # copy info.json from real_data_path with shutil
                import shutil
                shutil.copy(os.path.join(real_data_path, "info.json"), synthetic_data_path)
            except Exception as e:
                print("Could not copy info.json from real_data_path to synthetic_data_path, Error: ", e)


    # Todo? combine val and train data for evaluation (if eval_type == 'real')
    # X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, 'val')


    # merged is possible to set in eval_catboost.py, but not supported here. Should be removed in the future from eval_catboost.py as well
    print('-'*100)
    if eval_type == 'merged':
        print("Merged eval similarity is not supported.")
        return
    elif eval_type not in ['real', "synthetic"]:
        raise "Choose eval method"

    path = real_data_path if eval_type == 'real' else synthetic_data_path
    # train Controller and test Controller
    print(f"Loading {eval_type} Training data for comparison to the real Test data from {str(path)}")
    print(f"Test (and val) Data will be Loaded from {real_data_path}")
    train_transform = TabularDataController(
        path,
        "identity",
        num_classes=num_classes,
        splits=["train"])
    df_train = train_transform.to_pd_DataFrame(splits=["train"])

    test_transform = TabularDataController(
        real_data_path,
        "identity",
        num_classes=num_classes,

        splits=["test"])

    df_test = test_transform.to_pd_DataFrame(splits=["test"])
    
    # Create plots using TableEvaluator
    print("Starting table Evaluator")
    from evaluation.table_evaluator_fix import TableEvaluatorFix as TableEvaluator

    target_col=train_transform.config["dataset_config"]["target_column"]
    cat_col = train_transform.config["dataset_config"]["cat_columns"]
    num_col = train_transform.config["dataset_config"]["int_columns"]
    if target_col in cat_col:
        df_test[target_col] = df_test[target_col].astype(str)
        df_train[target_col] = df_train[target_col].astype(str)
    elif target_col in num_col:
        df_test[target_col] = df_test[target_col].astype(float)
        df_train[target_col] = df_train[target_col].astype(float)

    print("SEED: ", seed)
    # not necessary for now
    # if len(df_test) > len(df_train):
    #     df_test = df_test.sample(n=len(df_train), random_state=seed)
    # elif len(df_train) > len(df_test):
    #     df_train = df_train.sample(n=len(df_test), random_state=seed)

    te = TableEvaluator(df_test, df_train, cat_cols=train_transform.config["dataset_config"]["cat_columns"])
    save_dir = os.path.join(save_dir, "plots")
    print("Visual Eval")
    te.visual_evaluation(save_dir=save_dir)


if __name__ == "__main__":
    main()

# not necessary for now
def _equal_length(df1, df2):
        """
    Make two DataFrames have equal length by sampling.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame.
    df2 : pd.DataFrame
        The second DataFrame.

    Returns
    -------
    df1 : pd.DataFrame
        The first DataFrame with equal length.
    df2 : pd.DataFrame
        The second DataFrame with equal length.
    """
    len1 = len(df1)
    len2 = len(df2)
    if len1 > len2:
        df1 = df1.sample(n=len2)
    else:
        df2 = df2.sample(n=len1)
    return df1, df2
