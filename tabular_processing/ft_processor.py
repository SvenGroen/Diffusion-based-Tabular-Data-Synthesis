from .tabular_processor import TabularProcessor
import numpy as np
import torch
import pandas as pd
from typing import Tuple
from .ft_utils.ft_tokenizer import Tokenizer 
from .dataset import TaskType
from sklearn.preprocessing import OrdinalEncoder

class FTProcessor(TabularProcessor):
    def __init__(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray, cat_columns:list, problem_type:TaskType, target_column:str, **kwargs):
        super().__init__(x_cat, x_num, y)
        self.cat_columns = cat_columns
        self.problem_type = problem_type
        self.target_column = target_column

        self.d_numerical = x_num.shape[-1]
        self.tokenizer = Tokenizer(d_numerical=self.d_numerical,
            categories=list(pd.DataFrame(x_cat).nunique(axis=0)),
            d_token = 8, # embedding_dim
            bias=True)
        self.attr_enc = OrdinalEncoder()
        self.target_enc = OrdinalEncoder()



    def transform(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        x_cat = self.attr_enc.transform(x_cat)
        y = self.target_enc.transform(y.reshape(-1,1))

        # transform from [bs, num_features] to [bs, 1, num_features]
        x_num = x_num.reshape(-1, 1, x_num.shape[1])
        x_cat = x_cat.reshape(-1, 1, x_cat.shape[1])
        
        # transform to tensor
        x_num = torch.from_numpy(x_num).float()
        x_cat = torch.from_numpy(x_cat).int()

        out = self.tokenizer(x_num, x_cat)
        out = out.reshape(out.shape[0], -1).numpy()

        # set self.x_cat, self.x_num, self.y
        # all cat columns are transformed to numerical due to the embedding layer
        self.x_cat, self.x_num, self.y = None, out, y.squeeze(1) #np.empty_like(x_cat)


        return self.x_cat, self.x_num, self.y
        


    def inverse_transform(self, x_cat, x_num, y_pred) -> np.ndarray:
        assert self._was_fit, "You must call fit before inverse_transform"
        assert x_cat is None, "x_cat must be None"
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1,1)
        
        # add dummy dim at the end of x_num
        x_num = torch.Tensor(x_num.reshape(-1, x_num.shape[1], 1))
        x_cat, x_num = self.tokenizer.recover(x_num, d_numerical=self.d_numerical)
        x_cat = self.attr_enc.inverse_transform(x_cat - 1) # x_cat return by tokenizer is 1-indexed (--> "new_Batch_cat[j, i] = nearest + 1")
        y_pred = self.target_enc.inverse_transform(y_pred)
        # to numpy
        x_num = x_num.numpy()
        # remove dummy dim
        y_pred = y_pred.squeeze(1)

        return x_cat, x_num, y_pred


    def fit(self, meta_data: dict = None) -> None:
        # if target column is in cat_columns, concat it to x_cat
        if self.target_column in self.cat_columns:
            self.target_enc.fit(self.y.reshape(-1,1))
        self.attr_enc.fit(self.x_cat)
        self._was_fit = True
        return self

    def fit_transoform(self, meta_data: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.fit()
        self.transform(self.x_cat, self.x_num, self.y) # sets self.x_cat, self.x_num, self.y
        return self.x_cat, self.x_num, self.y