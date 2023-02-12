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
from tabular_processing.tabular_transformer import TabularTransformer
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

    # apply Transformations ? 
    T = lib.Transformations(**T_dict)
    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(real_data_path, val_size=0.2)
    else: 
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, 'val')

    # validation
    val_transform = TabularTransformer(
        real_data_path, 
        "identity",
        num_classes=num_classes,
        is_y_cond=is_y_cond,
        splits=["val"])
    val_transform.x_cat["val"] = X_cat_val
    val_transform.x_num["val"] = X_num_val
    val_transform.y["val"] = y_val
    df_val = val_transform.to_pd_DataFrame(splits=["val"])

    print('-'*100)
    if eval_type == 'merged':
        print("Merged eval similarity is not supported.")
        return
    elif eval_type not in ['real', "synthetic"]:
        raise "Choose eval method"

    path = real_data_path if eval_type == 'real' else synthetic_data_path






    print(f"Loading {eval_type} Training data for comparison to the real Test data from {str(path)}")
    print(f"Test (and val) Data will be Loaded from {real_data_path}")
    train_transform = TabularTransformer(
        path,
        "identity",
        num_classes=num_classes,
        is_y_cond=is_y_cond,
        splits=["train"])
    df_train = train_transform.to_pd_DataFrame(splits=["train"])

    test_transform = TabularTransformer(
        real_data_path,
        "identity",
        num_classes=num_classes,
        is_y_cond=is_y_cond,
        splits=["test"])

    df_test = test_transform.to_pd_DataFrame(splits=["test"])
    
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


    print("SIMILARITY SCORE")
    for key, value in report['sim_score'].items():
        print(f"{key}: {value}")
    print("*"*100)    

    

    if not change_val:
        # save dataframes only after final evaluation
        df_train.to_csv(os.path.join(parent_dir, f"train_{eval_type}.csv"), index=False)
        df_test.to_csv(os.path.join(parent_dir, f"test_real.csv"), index=False)

        # Plotting
    if table_evaluate:
        print("Starting table Evaluator")
        try:
            from tabsyndex.table_evaluator_fix import TableEvaluatorFix as TableEvaluator
            target_col=train_transform.config["dataset_config"]["target_column"]
            cat_col = train_transform.config["dataset_config"]["cat_columns"]
            num_col = train_transform.config["dataset_config"]["int_columns"]
            if target_col in cat_col:
                df_test[target_col] = df_test[target_col].astype(str)
                df_train[target_col] = df_train[target_col].astype(str)
            elif target_col in num_col:
                df_test[target_col] = df_test[target_col].astype(float)
                df_train[target_col] = df_train[target_col].astype(float)

            if len(df_test) > len(df_train):
                df_test = df_test.sample(n=len(df_train))
            elif len(df_train) > len(df_test):
                df_train = df_train.sample(n=len(df_test))

            te = TableEvaluator(df_test, df_train, cat_cols=train_transform.config["dataset_config"]["cat_columns"])
            save_dir = os.path.join(parent_dir, "plots")
            print("Visual Eval")
            te.visual_evaluation(save_dir=save_dir)
            # te.plot_mean_std(fname=save_dir/'mean_std.png')
            # te.plot_cumsums(fname=save_dir/'cumsums.png')
            # te.plot_distributions(fname=save_dir/'distributions.png')
            # te.plot_correlation_difference()
            # plot_correlation_difference(real = df_test, fake=df_train, cat_cols=train_transform.config["dataset_config"]["cat_columns"], plot_diff=True, fname=save_dir/'correlation_difference.png')
            # te.plot_pca(fname=save_dir/'pca.png') 
            print("Multiple Eval")
            output = te.evaluate(return_outputs=True,verbose=True, target_col = train_transform.config["dataset_config"]["target_column"])
            print(output)

            report['table_evaluator'] = output
        except Exception as e:
            import traceback
            print("TableEvaluator failed, error: ")
            print(traceback.format_exc())
        print("Finished Table Evaluator")

    if parent_dir is not None:
        lib.dump_json(report, os.path.join(parent_dir, "results_similarity.json"))
    
    end = time.time()
    print("Finishing Similarity Evaluation. Time Passed (in min): ", (end-st)/60)
    return report

def _equal_length(df1, df2):
    len1 = len(df1)
    len2 = len(df2)
    if len1 > len2:
        df1 = df1.sample(n=len2)
    else:
        df2 = df2.sample(n=len1)
    return df1, df2
