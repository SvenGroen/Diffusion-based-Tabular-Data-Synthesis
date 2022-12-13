import argparse
import json
import pandas as pd
import numpy as np
import catboost.datasets
import matplotlib.pyplot as plt
from pathlib import Path
from tabular_processing.dataset import TaskType, _make_split, _apply_split, _save
from tabular_processing.bgm_utils.bgm_transformer import BGMTransformer
from table_evaluator import load_data, TableEvaluator

CAT_MISSING_VALUE = '__nan__'

def postprocess(data_folder, config, preprocess_method):
    
    df_real, _ = load_dataset(name=None, data_folder=data_folder,filename="synthetic/output_r.csv", download=False)
    df_syn, _ = load_dataset(name=None, data_folder=data_folder,filename="synthetic/output_syn.csv", download=False)
    
    # Loading original real_data
    df_trainval, df_test = load_dataset(name=config["name"], data_folder=data_folder, download=True)
    data=pd.concat([df_trainval, df_test], axis=0) 

    data_config = config["dataset_config"]

    if preprocess_method.lower() == "bgm":
        transformer = BGMTransformer(data.copy(), **data_config)
        transformer.fit_transform()
        transformer.data = df_syn.to_numpy()
        cat_len, num_len, y_len = transformer.x_cat.shape[-1], transformer.x_num.shape[-1], transformer.y.shape[-1]
        x_num = transformer.data[:,:num_len]
        x_cat = transformer.data[:,num_len:cat_len+num_len]
        y = transformer.data[:,cat_len+num_len:cat_len+num_len+y_len]

        inverse = transformer.inverse_transform(x_cat, x_num, y_pred=y)

        df_r = data
        df_f = inverse
        if len(df_r) > len(df_f):
            df_r = df_r.head(len(df_f))
        else:
            df_f = df_f.head(len(df_r))

        for col in df_r:
            if len(df_f[col].unique()) != len(df_r[col].unique()):
                print(f"Column {col} has {len(df_f[col].unique())} unique entries but should have {len(df_r[col].unique())}")
            else:
                print(f"Column {col} identical")

        # plot_columns(df_r, df_f, save_path=data_folder)
        table_evaluator = TableEvaluator(df_r, df_f, cat_cols=config["dataset_config"]["cat_columns"], verbose=True)
        table_evaluator.visual_evaluation(save_dir=data_folder)
        output = table_evaluator.evaluate(target_col=config["dataset_config"]["target_column"], return_outputs=True)





        # save


        # data_ = transformer.inverse_transform(x_cat, x_num, y_pred=y)
        # todo: save the transformer data + split



    pass


# Function that takes two dataframes as input and plots a barchart for each column
def plot_columns(df_real, df_syn, save_path=None):
  # Get the names of the columns in each dataframe
  col_names_real = df_real.columns
  col_names_syn = df_syn.columns

  # Iterate over the columns in the first dataframe
  for col_name in col_names_real:
    # Get the values in the current column of the first dataframe
    values_real = df_real[col_name].value_counts()

    # Check if the current column exists in the second dataframe
    if col_name in col_names_syn:
      

        # If it exists, get the values in the current column of the second dataframe
        values_syn = df_syn[col_name].value_counts()    

        # Create a figure with one subplot
        fig, ax = plt.subplots(figsize=(10, 5)) 
        
        

        if df_real[col_name].dtype.kind in 'biufc':
            # The column is numerical, so we will plot a histogram for each dataframe
            
            ax.hist(values_real.values, label="Real Data", bins=50, color="red")
            ax.hist(values_syn.values, label="Synthetic Data", bins=50, color="blue")
        else:
             # if idx is not the same at dummy values to syn
            if len(values_real) != len(values_syn):
                for idx in values_syn.index:
                    if idx not in values_real.index:
                        values_real[idx] = 0
                for idx in values_real.index:
                    if idx not in values_syn.index:
                        values_syn[idx] = 0
                values_real = values_real.sort_index()
                values_syn = values_syn.sort_index()

            idx = np.arange(len(values_real))

            # Plot a barchart for the values in the current column of the first dataframe
            ax.bar(idx, values_real.values, label="Real Data", width=0.4, color="red") 
            # Plot a barchart for the values in the current column of the second dataframe
            ax.bar(idx + 0.4, values_syn.values, label="Synthetic Data", width=0.4, color="blue")  
            # add x tick labels 
            ax.set_xticks(idx + 0.4 / 2)
            ax.set_xticklabels(values_real.index, rotation=90)


        # Add a legend
        ax.legend() 

        # Set the title and labels for the plot
        ax.set_title(col_name)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")  
        
        

        # Add a text field with the total number of unique entries for each column
        ax.text(0.5, 0.5, f"Real Data - Unique Entries: {len(values_real)}\nSynthetic Data - Unique Entries: {len(values_syn)}", transform=ax.transAxes)    
        # Show the plot
        plt.show()
        # save plot
        if save_path is not None:
            save_path = (Path(save_path) / "plots").mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path/f"{col_name}.png")

# Example usage


def load_dataset(name:str, data_folder:str, download:bool=False, filename:str=None):
    if str(name).lower() == "adult":
        if not download:
            path = data_folder / "adult.data" if filename is None else data_folder / filename
            columns = ["age","workclass","fnlwgt", "education", "education-num",
                        "marital-status", "occupation", "relationship", 
                        "race", "sex","capital-gain", "capital-loss", "hours-per-week",
                        "native-country", "income"]
            data = pd.read_csv(path, names=columns, na_values="?")
            return data, None
        else:
            df_trainval, df_test = catboost.datasets.adult()
            assert (df_trainval.dtypes == df_test.dtypes).all()
            assert (df_trainval.columns == df_test.columns).all()
            return df_trainval, df_test
    else:
        assert filename is not None, "filename must be provided if download is False"
        return pd.read_csv(data_folder / filename), None


    return df_trainval, df_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', metavar='FILE')
    parser.add_argument('preprocess_method', type=str, default="bgm")
    args = parser.parse_args()
    data_folder = Path(args.data_folder)
    raw_config = json.load(open(data_folder / "info.json"))
    preprocess_method = args.preprocess_method

    postprocess(data_folder=data_folder, config=raw_config, preprocess_method=preprocess_method)

if __name__ == '__main__':
    main()
