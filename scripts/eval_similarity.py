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
    report['sim_score'] = sim_score 


    print("SIMILARITY SCORE")
    for key, value in report['sim_score'].items():
        print(f"{key}: {value}")
    print("*"*100)    

    

    if not change_val:
        # save dataframes only after final evaluation
        df_train.to_csv(os.path.join(parent_dir, f"train_{eval_type}.csv"), index=False)
        df_test.to_csv(os.path.join(parent_dir, f"test_{eval_type}.csv"), index=False)

        # Plotting
    if table_evaluate:
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

            te = TableEvaluator(df_test, df_train, cat_cols=train_transform.config["dataset_config"]["cat_columns"])
            save_dir = os.path.join(parent_dir, "plots")
            te.visual_evaluation(save_dir=save_dir)
            # te.plot_mean_std(fname=save_dir/'mean_std.png')
            # te.plot_cumsums(fname=save_dir/'cumsums.png')
            # te.plot_distributions(fname=save_dir/'distributions.png')
            # te.plot_correlation_difference()
            # plot_correlation_difference(real = df_test, fake=df_train, cat_cols=train_transform.config["dataset_config"]["cat_columns"], plot_diff=True, fname=save_dir/'correlation_difference.png')
            # te.plot_pca(fname=save_dir/'pca.png') 
            output = te.evaluate(return_outputs=True,verbose=True, target_col = train_transform.config["dataset_config"]["target_column"])
            print(output)

            report['table_evaluator'] = output
        except Exception as e:
            print("TableEvaluator failed")
            print(e)

    if parent_dir is not None:
        lib.dump_json(report, os.path.join(parent_dir, "results_similarity.json"))
    
    return report

def _equal_length(df1, df2):
    len1 = len(df1)
    len2 = len(df2)
    if len1 > len2:
        df1 = df1.sample(n=len2)
    else:
        df2 = df2.sample(n=len1)
    return df1, df2

def plot_correlation_difference(real: pd.DataFrame, fake: pd.DataFrame, plot_diff: bool = True, cat_cols: list = None, annot=False, fname=None):
    """

    TAKEN FROM TABLE EVALUATOR BUT REPLACING THE ASSOCIATE FUNCTION

    Plot the association matrices for the `real` dataframe, `fake` dataframe and plot the difference between them. Has support for continuous and Categorical
    (Male, Female) data types. All Object and Category dtypes are considered to be Categorical columns if `dis_cols` is not passed.

    - Continuous - Continuous: Uses Pearson's correlation coefficient
    - Continuous - Categorical: Uses so called correlation ratio (https://en.wikipedia.org/wiki/Correlation_ratio) for both continuous - categorical and categorical - continuous.
    - Categorical - Categorical: Uses Theil's U, an asymmetric correlation metric for Categorical associations

    :param real: DataFrame with real data
    :param fake: DataFrame with synthetic data
    :param plot_diff: Plot difference if True, else not
    :param cat_cols: List of Categorical columns
    :param boolean annot: Whether to annotate the plot with numbers indicating the associations.
    """


    assert isinstance(real, pd.DataFrame), f'`real` parameters must be a Pandas DataFrame'
    assert isinstance(fake, pd.DataFrame), f'`fake` parameters must be a Pandas DataFrame'
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    if cat_cols is None:
        cat_cols = real.select_dtypes(['object', 'category'])
    if plot_diff:
        fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    real_corr = calculate_correlation_map(real)
    fake_corr = calculate_correlation_map(fake)

    if plot_diff:
        diff = abs(real_corr - fake_corr)
        sns.set(style="white")
        sns.heatmap(diff, ax=ax[2], cmap=cmap, vmax=.3, square=True, annot=annot, center=0,
                    linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')

    titles = ['Real', 'Fake', 'Difference'] if plot_diff else ['Real', 'Fake']
    for i, label in enumerate(titles):
        title_font = {'size': '18'}
        ax[i].set_title(label, **title_font)
    plt.tight_layout()

    if fname is not None: 
        plt.savefig(fname)

    plt.show()

def calculate_correlation_map(df):
    # Create an empty correlation matrix
    from scipy.stats import pearsonr, spearmanr
    corr_matrix = pd.DataFrame(columns=df.columns, index=df.columns)
    
    # Iterate over the columns in the DataFrame
    for col1 in df.columns:
        for col2 in df.columns:
            # Skip self-correlations
            if col1 == col2:
                continue
                
            # Check the data types of the columns
            dtype1 = df[col1].dtype
            dtype2 = df[col2].dtype
            
            # Continuous - Continuous: Use Pearson's correlation coefficient
            if dtype1 == 'float' and dtype2 == 'float':
                corr, _ = pearsonr(df[col1], df[col2])
                corr_matrix.loc[col1, col2] = corr
                
            # Continuous - Categorical: Use correlation ratio
            elif dtype1 == 'float' and dtype2 == 'object':
                corr = df[col1].corr(df[col2], method='spearman')
                corr_matrix.loc[col1, col2] = corr
                
            # Categorical - Continuous: Use correlation ratio
            elif dtype1 == 'object' and dtype2 == 'float':
                corr = df[col2].corr(df[col1], method='spearman')
                corr_matrix.loc[col1, col2] = corr
                
            # Categorical - Categorical: Use Theil's U
            elif dtype1 == 'object' and dtype2 == 'object':
                corr = df[col1].corr(df[col2], method='theil_u')
                corr_matrix.loc[col1, col2] = corr
                
    return corr_matrix