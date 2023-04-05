from .tabular_processor import TabularProcessor
import numpy as np
import torch
import pandas as pd
from typing import Tuple
from .ft_utils.ft_tokenizer import Tokenizer 
from .dataset import TaskType
from sklearn.preprocessing import OrdinalEncoder

class FTProcessor(TabularProcessor):
    """
    Processor class that transforms tabular data using a Feature Tokenization approach.
    Inherits from the TabularProcessor abstract class.

    Parameters
    ----------
    x_cat : np.ndarray
        Categorical features to be transformed
    x_num : np.ndarray
        Numerical features to be transformed
    y : np.ndarray
        Targets to be transformed
    cat_columns : list
        List of categorical column names
    problem_type : TaskType
        Type of task (TaskType.BINCLASS, TaskType.MULTICLASS, or TaskType.REGRESSION)
    target_column : str
        Name of target column
    kwargs : optional
        Additional keyword arguments

    Attributes
    ----------
    cat_columns : list
        List of categorical column names
    problem_type : TaskType
        Type of task (TaskType.BINCLASS, TaskType.MULTICLASS, or TaskType.REGRESSION)
    target_column : str
        Name of target column
    d_numerical : int
        Number of numerical features
    tokenizer : Tokenizer
        Instance of Tokenizer class
    attr_enc : OrdinalEncoder
        OrdinalEncoder for categorical features
    target_enc : OrdinalEncoder
        OrdinalEncoder for target feature

    Methods
    -------
    transform(x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Transforms the tabular data using a feature tokenization approach.
    inverse_transform(x_cat, x_num, y_pred) -> np.ndarray:
        Inverse transforms the transformed tabular data to its original state.
    fit(meta_data: dict = None) -> None:
        Fits the FTProcessor with the provided metadata.
    fit_transoform(meta_data: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Fits and transforms the FTProcessor with the provided metadata.

    """
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
        """
        Transforms the tabular data using a feature tokenization approach.

        Parameters
        ----------
        x_cat : np.ndarray
            Categorical features to be transformed
        x_num : np.ndarray
            Numerical features to be transformed
        y : np.ndarray
            Targets to be transformed

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Transformed categorical features, transformed numerical features, transformed targets
        """

        x_cat = self.attr_enc.transform(x_cat)
        x_cat = x_cat + 1 # tokenization starts from 1 (will in reverse function be subtracted by 1)
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
        """
        Transforms the transformed data back to the original format.

        Parameters
        ----------
        x_cat : np.ndarray
            The transformed categorical features
        x_num : np.ndarray
            The transformed numerical features
        y_pred : np.ndarray
            The predicted target values

        Returns
        -------
        np.ndarray
            The data in its original format, containing the categorical features, numerical features, and predicted target values.
        """
        assert self._was_fit, "You must call fit before inverse_transform"
        assert x_cat is None, "x_cat must be None"
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1,1)
        
        # add dummy dim at the end of x_num
        x_num = torch.Tensor(x_num.reshape(-1, x_num.shape[1], 1))
        x_cat, x_num = self.tokenizer.recover(x_num, d_numerical=self.d_numerical)
        x_cat = self.attr_enc.inverse_transform(x_cat -1) # x_cat return by tokenizer is 1-indexed (--> "new_Batch_cat[j, i] = nearest + 1")
        y_pred = self.target_enc.inverse_transform(y_pred)
        # to numpy
        x_num = x_num.numpy()
        # remove dummy dim
        y_pred = y_pred.squeeze(1)

        return x_cat, x_num, y_pred


    def fit(self, meta_data: dict = None) -> None:
        """
        Fits the transformer to the tabular data.

        Parameters
        ----------
        meta_data : dict, optional
            Metadata about the categorical features, by default None

        Returns
        -------
        None
        """
        # if target column is in cat_columns, concat it to x_cat
        if self.target_column in self.cat_columns:
            self.target_enc.fit(self.y.reshape(-1,1))
        self.attr_enc.fit(self.x_cat)
        print("self.attr_enc.categories_:", self.attr_enc.categories_)
        self._was_fit = True
        return self

    def fit_transoform(self, meta_data: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits the transformer to the tabular data and returns the transformed data.

        Parameters
        ----------
        meta_data : dict, optional
            Metadata about the categorical features, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the transformed categorical features, numerical features, and target values.
        """
        self.fit()
        self.transform(self.x_cat, self.x_num, self.y) # sets self.x_cat, self.x_num, self.y
        return self.x_cat, self.x_num, self.y