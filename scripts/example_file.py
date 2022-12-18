import pandas as pd
import numpy as np
from table_evaluator import TableEvaluator

df_real = pd.read_csv("data/adult/train.csv")
print(df_real.head())

te = TableEvaluator(df_real, df_real, cat_cols=["6","7", "8", "9", "10", "11", "12", "13", "y"])

te.visual_evaluation()
output = te.evaluate(return_outputs=True,verbose=True, target_col = "y")
print(output)