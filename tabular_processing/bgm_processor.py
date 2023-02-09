from typing import cast
import numpy as np
import pandas as pd

from .tabular_processor import TabularProcessor
from .dataset import TaskType
from .bgm_utils.transformer import DataTransformer
from .bgm_utils.data_preparation import DataPrep
from .util import get_column_names

class BGMProcessor(TabularProcessor):
    def __init__(self,
                x_cat:  np.ndarray,
                x_num  : np.ndarray, 
                y  : np.ndarray,
                cat_columns : list,
                log_columns : list,
                mixed_columns: dict,
                general_columns: list,
                non_cat_columns: list,
                int_columns: list,
                problem_type: TaskType, #"binclass" or "multiclass" or "regression"
                target_column: str,
    ):
        super().__init__(x_cat, x_num, y)
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
        self.cat_values = None
        self.y_num_classes = 2 if problem_type == TaskType.BINCLASS else len(self.y.unique()) if problem_type == TaskType.MULTICLASS else 1 # 1 for regression

    def splitted_to_dataframe(self, x_cat, x_num, y):
        cat_columns = [i for i in self.cat_columns if i != self.target_column]
        num_columns = [i for i in self.int_columns if i != self.target_column]
        x_cat = pd.DataFrame(x_cat, columns=cat_columns)
        x_num = pd.DataFrame(x_num, columns=num_columns)
        y = pd.DataFrame(y, columns=[self.target_column])
        data = pd.concat([x_cat, x_num, y], axis=1)
        return data

    def dataframe_to_splitted(self, data):
        cat_columns = [i for i in self.cat_columns if i != self.target_column]
        num_columns = [i for i in self.int_columns if i != self.target_column]
        x_cat = data[cat_columns].to_numpy()
        x_num = data[num_columns].to_numpy()
        y = data[self.target_column].to_numpy()
        return x_cat, x_num, y

    def fit(self, meta_data=None):
        self.cat_values = meta_data
        data = self.splitted_to_dataframe(self.x_cat, self.x_num, self.y)
        self.data_prep = DataPrep(categorical=self.cat_columns,
                                log=self.log_columns,
                                mixed=self.mixed_columns,
                                general=self.general_columns,
                                non_categorical=self.non_cat_columns,
                                integer=self.int_columns)
        df = self.data_prep.prep(raw_df=data, cat_values=meta_data)
        self.data_transformer = DataTransformer(train_data=df.loc[:, df.columns != self.target_column],
                                        categorical_list=self.data_prep.column_types["categorical"],
                                        mixed_dict=self.data_prep.column_types["mixed"],
                                        general_list=self.data_prep.column_types["general"],
                                        non_categorical_list=self.data_prep.column_types["non_categorical"],
                                        n_clusters=10, eps=0.005)
        self.data_transformer.fit()
        self._was_fit = True
        return self

    def transform(self, x_cat : np.ndarray, x_num : np.ndarray, y : np.ndarray):
        assert self.data_prep is not None, "You must fit the transformer first"
        data = self.splitted_to_dataframe(x_cat, x_num, y)
        data = self.data_prep.prep(raw_df=data, cat_values=self.cat_values)
        self.y = data.pop(self.target_column).to_numpy()
        data = self.data_transformer.transform(data.values)
        self.x_cat, self.x_num = self.split_cat_num(data=data)
        return self.x_cat, self.x_num, self.y

    def fit_transform(self):
        self.fit()
        self.transform(self.x_cat, self.x_num, self.y) # sets self.x_cat, self.x_num, self.y
        return self.x_cat, self.x_num, self.y

    def inverse_transform(self, x_cat:np.ndarray, x_num:np.ndarray, y_pred:np.ndarray):
        assert self.data_prep is not None, "You must fit the transformer first"
        assert isinstance(y_pred,np.ndarray), "y_pred must be a numpy array"
        assert isinstance(x_cat,np.ndarray), "x_cat must be a numpy array"
        assert isinstance(x_num,np.ndarray), "x_num must be a numpy array"
        # apply activation functions?
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1,1)
        data=self.inverse_split_cat_num(x_cat, x_num)
        data, inv_ids = self.data_transformer.inverse_transform(data)
        columns = self.data_prep.df.columns
        data = pd.DataFrame(data, columns=columns)
        data = pd.concat([data, pd.DataFrame(y_pred, columns=[self.target_column])], axis=1)
        self.data_prep.df = data # data prep object needs updated df
        # TODO: WHAT TO DO WITH NANS?
        # data = data.fillna(0)
        data = self.data_prep.inverse_prep(data)
        return self.dataframe_to_splitted(data)

    def split_cat_num(self, data):
        self.x_cat, self.x_num = self.data_transformer.split_cat_num(data, cat_style="labels")
        return self.x_cat, self.x_num

    def inverse_split_cat_num(self, x_cat, x_num):
        data = self.data_transformer.inverse_split_cat_num(x_cat, x_num)
        return data