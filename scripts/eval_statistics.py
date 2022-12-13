from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from table_evaluator import load_data, TableEvaluator



path = Path("data/adult/synthetic")
real = "output_r0.csv"
syn = "output_syn0.csv"

cat_columns= [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
            "income"
        ]
columns = ["age","workclass","fnlwgt", "education", "education-num",
                        "marital-status", "occupation", "relationship", 
                        "race", "sex","capital-gain", "capital-loss", "hours-per-week",
                        "native-country", "income"]

df_r = pd.read_csv(path/real, names=columns, skiprows=1)
df_f = pd.read_csv(path/syn, names=columns, skiprows=1)

if len(df_r) > len(df_f):
    df_r = df_r.head(len(df_f))
else:
    df_f = df_f.head(len(df_r))
from pandas.api.types import is_numeric_dtype, is_integer_dtype

for col in cat_columns:
    # get type of column
    col_type = df_f[col].dtypes
    # set dtype to df_r
    if is_numeric_dtype(col_type):
        df_r[col] = df_r[col].astype("float64")
        # set dtype to df_f
        df_f[col] = df_f[col].astype("float64")

print(df_f.head())
print(df_r.head())

table_evaluator = TableEvaluator(df_r, df_f, cat_cols=cat_columns, verbose=True)
table_evaluator.visual_evaluation(save_dir=path/"plots")
# table_evaluator.plot_distributions(fname=path/"plots"/'distributions.png')
# table_evaluator.plot_correlation_difference(fname=path/"plots"/'correlation_difference.png')

output = table_evaluator.evaluate(target_col="income", return_outputs=True)
