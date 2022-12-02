from typing import cast
import numpy as np
import pandas as pd

from ..dataset import TaskType
from .transform.transformer import DataTransformer
from .transform.data_preparation import DataPrep

class BGMTransformer:
    def __init__(self,
                data: pd.DataFrame,
                cat_columns : list,
                log_columns : list,
                mixed_columns: dict,
                general_columns: list,
                non_cat_columns: list,
                int_columns: list,
                problem_type: TaskType, #"binclass" or "multiclass" or "regression"
                target_column: str,
    ):
        self.data = data
        self.raw_data = data.copy()
        self.cat_columns = cat_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_cat_columns = non_cat_columns
        self.int_columns = int_columns
        assert problem_type.upper() in TaskType.__members__, "problem_type must be one of 'binclass', 'multiclass', 'regression'"
        self.problem_type = TaskType[problem_type.upper()]
        self.target_column = target_column
        self.data_prep = None
        self.data_transformer = None
        self.x_cat = None
        self.x_num = None
        self.x = self.data.loc[:, data.columns != self.target_column]
        self.y = self.data[self.target_column]
        self.y_num_classes = 2 if problem_type == TaskType.BINCLASS else len(self.y.unique()) if problem_type == TaskType.MULTICLASS else 1 # 1 for regression
    
    def fit(self):
        self.data_prep = DataPrep(raw_df=self.data,
                                categorical=self.cat_columns,
                                log=self.log_columns,
                                mixed=self.mixed_columns,
                                general=self.general_columns,
                                non_categorical=self.non_cat_columns,
                                integer=self.int_columns)
        df = self.data_prep.df
        self.data_transformer = DataTransformer(train_data=df.loc[:, df.columns != self.target_column],
                                        categorical_list=self.data_prep.column_types["categorical"],
                                        mixed_dict=self.data_prep.column_types["mixed"],
                                        general_list=self.data_prep.column_types["general"],
                                        non_categorical_list=self.data_prep.column_types["non_categorical"],
                                        n_clusters=10, eps=0.005)
        self.data_transformer.fit()
        
        return self

    def transform(self):
        assert self.data_prep is not None, "You must fit the transformer first"
        self.y = self.data_prep.df.pop(self.target_column).to_numpy()
        self.data = self.data_transformer.transform(self.data_prep.df.values)
        self.split_cat_num()
        return self
    
    def get_cat_num_y(self):
        return self.x_cat, self.x_num, self.y

    def fit_transform(self):
        self.fit()
        self.transform()
        return self
    
    def inverse_transform(self,x_cat:np.array, x_num:np.array, y_pred:np.array):
        assert self.data_prep is not None, "You must fit the transformer first"
        assert isinstance(y_pred,np.ndarray), "y_pred must be a numpy array"
        assert isinstance(x_cat,np.ndarray), "x_cat must be a numpy array"
        assert isinstance(x_num,np.ndarray), "x_num must be a numpy array"
        # apply activation functions?
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1,1)
        data=self.inverse_split_cat_num(x_cat, x_num)
        self.data, inv_ids = self.data_transformer.inverse_transform(data)
        self.data_prep.df = pd.concat([self.data_prep.df, pd.DataFrame(y_pred, columns=[self.target_column])], axis=1)
        self.data = np.concatenate((self.data, y_pred), axis=1)
        self.data = self.data_prep.inverse_prep(self.data)
        return self.data

    def split_cat_num(self):
        self.x_cat, self.x_num = self.data_transformer.split_cat_num(self.data, cat_style="labels")
        return self.x_cat, self.x_num

    def inverse_split_cat_num(self, x_cat, x_num):
        self.data = self.data_transformer.inverse_split_cat_num(x_cat, x_num)
        return self.data