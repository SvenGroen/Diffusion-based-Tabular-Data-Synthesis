'''
The basic structure of the code follows the eval_catboost.py script. Instead of calculating the ml efficacy, it claculates the TabSynDex similarity score.
Additionally, visualizations of synthetic data are created.
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
import lib
from lib import concat_features, read_pure_data, get_catboost_config, read_changed_val
from tabular_processing.tabular_processor import TabularProcessor
import json
from tabsyndex.tabsyndex import tabsyndex
from tabular_processing.tabular_data_controller import TabularDataController
import time

def calculate_similarity_score(    
    parent_dir,
    real_data_path,
    eval_type,
    T_dict,
    seed = 0,
    num_classes = 2,
    is_y_cond = False,
    change_val = True,
    table_evaluate = True
    ):  
    """
    Calculate the similarity score between real and synthetic datasets.
    Additionally, visualizations of synthetic data are created.

    Parameters
    ----------
    parent_dir : str
        The parent directory path.
    real_data_path : str
        The path to the real data.
    eval_type : str
        The evaluation type, either "real" or "synthetic".
        "real" means that the real training data is used for evaluation.
        "synthetic" means that the synthetic training data is used for evaluation.
        Either "real" or "synthetic" will be compared to the "real" test data. 
    T_dict : dict
        The dictionary containing transformation settings.
    seed : int, optional
        The random seed for reproducibility. Default is 0.
    num_classes : int, optional
        The number of classes. Default is 2.
    is_y_cond : bool, optional
        Whether the transformation is conditioned on y. Default is False.
    change_val : bool, optional
        Whether to change the validation dataset. Default is True.
    table_evaluate : bool, optional
        Whether to use table evaluation. Default is True.

    Returns
    -------
    report : dict
        The evaluation report containing the similarity score and other metrics.
    """
    st = time.time()
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

     # load validation data    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(real_data_path, val_size=0.2)
    else: 
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, 'val')

    # we need train, val and test data as pandas dataframes for similarity evaluation, so we create them using the TabularDataController
    # Note: there is probably a more efficient way to do this, but this is the easiest way to do it for now

    # validation Controller
    val_transform = TabularDataController(
        real_data_path, 
        "identity",
        num_classes=num_classes,
        is_y_cond=is_y_cond,
        splits=["val"])
    # add validation data to the controller (might have changed if change_val is True)
    val_transform.x_cat["val"] = X_cat_val
    val_transform.x_num["val"] = X_num_val
    val_transform.y["val"] = y_val
    df_val = val_transform.to_pd_DataFrame(splits=["val"])

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
        is_y_cond=is_y_cond,
        splits=["train"])
    df_train = train_transform.to_pd_DataFrame(splits=["train"])

    test_transform = TabularDataController(
        real_data_path,
        "identity",
        num_classes=num_classes,
        is_y_cond=is_y_cond,
        splits=["test"])
    df_test = test_transform.to_pd_DataFrame(splits=["test"])
    
    # calculate similarity score, make sure that the dataframes have the same length, otherwise the similarity score will likely throw an error
    if change_val:
        print(f"comparing {eval_type} to real validation data:")
        df_val, df_train = _equal_length(df_val, df_train)
        sim_score = tabsyndex(df_val, df_train, 
            cat_cols=train_transform.config["dataset_config"]["cat_columns"], )
    else:
        print(f"comparing {eval_type} to real test data:")
        df_test, df_train = _equal_length(df_test, df_train)
        sim_score = tabsyndex(df_test, df_train, 
            cat_cols=train_transform.config["dataset_config"]["cat_columns"],)

    print("*"*100)
    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = real_data_path
    report['sim_score'] = sim_score 

    # print similarity score results
    print("SIMILARITY SCORE")
    for key, value in report['sim_score'].items():
        print(f"{key}: {value}")
    print("*"*100)    

    
    # save dataframes and similarity score
    if not change_val:
        # save dataframes only after final evaluation
        df_train.to_csv(os.path.join(parent_dir, f"train_{eval_type}.csv"), index=False)
        df_test.to_csv(os.path.join(parent_dir, f"test_real.csv"), index=False)

    # Plotting 
    if table_evaluate:
        print("Starting table Evaluator")
        try:
            # TableEvaluatorFix is a modified version of the original TableEvaluator script for visualizations
            from tabsyndex.table_evaluator_fix import TableEvaluatorFix as TableEvaluator
            target_col=train_transform.config["dataset_config"]["target_column"]
            cat_col = train_transform.config["dataset_config"]["cat_columns"]
            num_col = train_transform.config["dataset_config"]["int_columns"]
            # assert cols have the correct type so no errors are thrown
            if target_col in cat_col:
                df_test[target_col] = df_test[target_col].astype(str)
                df_train[target_col] = df_train[target_col].astype(str)
            elif target_col in num_col:
                df_test[target_col] = df_test[target_col].astype(float)
                df_train[target_col] = df_train[target_col].astype(float)

            # make sure that the dataframes have the same length
            # Note: might not be necessary
            if len(df_test) > len(df_train):
                df_test = df_test.sample(n=len(df_train), random_state=seed)
            elif len(df_train) > len(df_test):
                df_train = df_train.sample(n=len(df_test), random_state=seed)

            te = TableEvaluator(df_test, df_train, cat_cols=train_transform.config["dataset_config"]["cat_columns"])
            save_dir = os.path.join(parent_dir, "plots")
            # create visualizations
            print("Visual Eval")
            te.visual_evaluation(save_dir=save_dir)

            # also calculate similarity score from the original author, it is saved but not used in the master thesis
            print("Multiple Eval")
            output = te.evaluate(return_outputs=True,verbose=True, target_col = train_transform.config["dataset_config"]["target_column"])
            print(output)

            report['table_evaluator'] = output
        except Exception as e:
            import traceback
            print("TableEvaluator failed, error: ")
            print(traceback.format_exc())
        print("Finished Table Evaluator")

    # save results
    if parent_dir is not None:
        lib.dump_json(report, os.path.join(parent_dir, "results_similarity.json"))
    end = time.time()
    print("Finishing Similarity Evaluation. Time Passed (in min): ", (end-st)/60)
    return report

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
