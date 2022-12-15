from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import classification_report, r2_score
import numpy as np
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

def calculate_similarity_score(    
    parent_dir,
    real_data_path,
    eval_type,
    T_dict,
    seed = 0,
    num_classes = 2,
    is_y_cond = False,
    change_val = True
    ):

    zero.improve_reproducibility(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)
    
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
    report['metrics'] = sim_score 

    
    print("SIMILARITY SCORE")
    for key, value in report['metrics'].items():
        print(f"{key}: {value}")
    print("*"*100)    

    if parent_dir is not None:
        lib.dump_json(report, os.path.join(parent_dir, "results_sim_score.json"))

    if not change_val:
        # save dataframes only after final evaluation
        df_train.to_csv(os.path.join(parent_dir, f"train_{eval_type}.csv"), index=False)
        df_test.to_csv(os.path.join(parent_dir, f"test_{eval_type}.csv"), index=False)

    return report

    # Plott
    # Save dataframe


def _equal_length(df1, df2):
    len1 = len(df1)
    len2 = len(df2)
    if len1 > len2:
        df1 = df1.sample(n=len2)
    else:
        df2 = df2.sample(n=len1)
    return df1, df2