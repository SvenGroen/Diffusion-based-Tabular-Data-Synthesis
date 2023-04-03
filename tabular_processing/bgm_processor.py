'''
Credits: 
Processing is based upon the work of the authors of:
https://github.com/Team-TUD/CTAB-GAN-Plus
https://github.com/Team-TUD/CTAB-GAN

The code was adapted to fit the needs of the TabularProcessor class.
'''

from typing import Dict, Optional, cast
import numpy as np
import pandas as pd

from .tabular_processor import TabularProcessor
from .dataset import TaskType
from .bgm_utils.transformer import DataTransformer
from .bgm_utils.data_preparation import DataPrep
from .util import get_column_names
from typing import Tuple, Union, List


class BGMProcessor(TabularProcessor):
    """
    A class to transform tabular data using Bayesian Gaussian Mixture models.
    The BGMProcessor class is a concrete implementation of the abstract base class TabularProcessor that is used for processing tabular data using the Bayesian Gaussian Mixture model (BGM). 
    The class provides methods to transform and inverse transform tabular data, and also includes methods for fitting the transformer to data, splitting and inverse splitting categorical and numerical data, 
    and for converting data between pandas DataFrames and numpy arrays. 
    The class is used in conjunction with other classes and utilities in the package to preprocess and transform tabular data for machine learning tasks such as binary or multi-class classification, and regression.
    
    Parameters
    ----------
    x_cat : np.ndarray
        categorical features
    x_num : np.ndarray
        numerical features
    y : np.ndarray
        target variable
    cat_columns : list
        column names of categorical features
    log_columns : list
        column names of features to be log transformed
    mixed_columns : dict
        column names and number of clusters for mixed features
    general_columns : list
        column names of features to be general transformed
    non_cat_columns : list
        column names of non-categorical features
    int_columns : list
        column names of integer features
    problem_type : TaskType
        type of problem, one of 'binclass', 'multiclass', or 'regression'
    target_column : str
        name of target column


    Attributes
    ----------
    x_cat : np.ndarray
        categorical features
    x_num : np.ndarray
        numerical features
    y : np.ndarray
        target variable
    cat_columns : list
        column names of categorical features
    log_columns : list
        column names of features to be log transformed
    mixed_columns : dict
        column names and number of clusters for mixed features
    general_columns : list
        column names of features to be general transformed
    non_cat_columns : list
        column names of non-categorical features
    int_columns : list
        column names of integer features
    problem_type : TaskType
        type of problem, one of 'binclass', 'multiclass', or 'regression'
    target_column : str
        name of target column
    data_prep : DataPrep
        data preparation object 
    data_transformer : DataTransformer
        data transformer object 
    cat_values : Dict[str, np.ndarray]
        dictionary of categorical feature names and their unique values
    y_num_classes : int
        number of classes in target variable, 2 for binary classification, number of unique values for multi-class classification, 1 for regression

    Methods
    -------
    splitted_to_dataframe(x_cat, x_num, y)
        Converts split categorical, numerical, and target variables to a pandas DataFrame
    dataframe_to_splitted(data)
        Converts a pandas DataFrame to split categorical, numerical, and target variables
    fit(meta_data=None)
        Fits the transformer to the input data
    transform(x_cat, x_num, y)
        Applies the transformer to the input data
    fit_transform()
        Fits the transformer to the input data and applies it to the input data
    inverse_transform(x_cat, x_num, y_pred)
        Inverses the transformer from the predicted target variable and the split categorical and numerical variables
    split_cat_num(data)
        Splits the input data into categorical and numerical data
    inverse_split_cat_num(x_cat, x_num)
        Combines the categorical and numerical data into a single data set
    """
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
        """
        The __init__ function of the BGMProcessor class initializes the object with the categorical, numerical, and target columns, as well as additional information about the columns such as whether they are should be encoded logarithmic, mixed, or general. 
        It also takes in the problem type, which can be "binclass", "multiclass", or "regression", and the target column name. 
        It then calls the __init__ function of the parent class TabularProcessor and sets the cat_values attribute to None, the y_num_classes attribute based on the problem type, and the data_prep and data_transformer attributes to None.
        """
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

    def splitted_to_dataframe(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Convert a numpy array of categorical and numerical data, and an array of target values into a pandas dataframe.
        Parameters
        ----------
        x_cat : np.ndarray
            Numpy array of categorical data.
        x_num : np.ndarray
            Numpy array of numerical data.
        y : np.ndarray
            Numpy array of target values.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the categorical and numerical data along with the target values.

        """
        cat_columns = [i for i in self.cat_columns if i != self.target_column]
        num_columns = [i for i in self.int_columns if i != self.target_column]
        x_cat = pd.DataFrame(x_cat, columns=cat_columns)
        x_num = pd.DataFrame(x_num, columns=num_columns)
        y = pd.DataFrame(y, columns=[self.target_column])
        data = pd.concat([x_cat, x_num, y], axis=1)
        return data

    def dataframe_to_splitted(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert a pandas dataframe to numpy arrays of categorical and numerical data and an array of target values.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe containing the categorical and numerical data and the target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of numpy arrays containing the categorical and numerical data and the target values (in this order).

        """
        cat_columns = [i for i in self.cat_columns if i != self.target_column]
        num_columns = [i for i in self.int_columns if i != self.target_column]
        x_cat = data[cat_columns].to_numpy()
        x_num = data[num_columns].to_numpy()
        y = data[self.target_column].to_numpy()
        return x_cat, x_num, y

    def fit(self, meta_data: Optional[Dict[str, np.ndarray]] = None) -> 'BGMProcessor':
        """
        Fit the transformer to the data.

        Parameters
        ----------
        meta_data : dict, optional
            Dictionary containing metadata about the categorical columns, by default None.
            The dictionary should have the column names as keys and the unique values for each column as values.

        Returns
        -------
        BGMProcessor
            Returns the instance of the class with the transformer fitted to the data.

        """
        if meta_data is None:
            raise ValueError("meta_data must be provided and should contain the unique values for each categorical column")
        self.cat_values = meta_data
        data = self.splitted_to_dataframe(self.x_cat, self.x_num, self.y)

        # Preprocess data
        self.data_prep = DataPrep(categorical=self.cat_columns,
                                log=self.log_columns,
                                mixed=self.mixed_columns,
                                general=self.general_columns,
                                non_categorical=self.non_cat_columns,
                                integer=self.int_columns)
        df = self.data_prep.prep(raw_df=data, cat_values=meta_data)

        # Transform data
        self.data_transformer = DataTransformer(train_data=df.loc[:, df.columns != self.target_column],
                                        categorical_list=self.data_prep.column_types["categorical"],
                                        mixed_dict=self.data_prep.column_types["mixed"],
                                        general_list=self.data_prep.column_types["general"],
                                        non_categorical_list=self.data_prep.column_types["non_categorical"],
                                        n_clusters=10, eps=0.005)
        self.data_transformer.fit()
        self._was_fit = True
        return self

    def transform(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform the data using the fitted transformer.

        Parameters
        ----------
        x_cat : np.ndarray
            Numpy array of categorical data.
        x_num : np.ndarray
            Numpy array of numerical data.
        y : np.ndarray
            Numpy array of target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of numpy arrays containing the transformed categorical and numerical data and the target values.

        """
        assert self.data_prep is not None, "You must fit the transformer first"
        data = self.splitted_to_dataframe(x_cat, x_num, y)
        data = self.data_prep.prep(raw_df=data, cat_values=self.cat_values)
        self.y = data.pop(self.target_column).to_numpy()
        data = self.data_transformer.transform(data.values)
        self.x_cat, self.x_num = self.split_cat_num(data=data)
        return self.x_cat, self.x_num, self.y

    def fit_transform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the transformer to the data and transform the data using the fitted transformer.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of numpy arrays containing the transformed categorical and numerical data and the target values.

        """
        self.fit()
        self.transform(self.x_cat, self.x_num, self.y) # sets self.x_cat, self.x_num, self.y
        return self.x_cat, self.x_num, self.y

    def inverse_transform(self, x_cat: np.ndarray, x_num: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inverse transform the data using the fitted transformer.

        Parameters
        ----------
        x_cat : np.ndarray
            Numpy array of categorical data.
        x_num : np.ndarray
            Numpy array of numerical data.
        y_pred : np.ndarray
            Numpy array of predicted target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of numpy arrays containing the categorical and numerical data and the predicted target values.

        """

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

    def split_cat_num(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the data into categorical and numerical data.

        Parameters
        ----------
        data : np.ndarray
            Numpy array containing the combined categorical and numerical data.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of numpy arrays containing the categorical and numerical data.

        """
        self.x_cat, self.x_num = self.data_transformer.split_cat_num(data, cat_style="labels")
        return self.x_cat, self.x_num

    def inverse_split_cat_num(self, x_cat: np.ndarray, x_num: np.ndarray) -> np.ndarray:
        """
        Inverse split the data into categorical and numerical data.

        Parameters
        ----------
        x_cat : np.ndarray
            Numpy array of categorical data.
        x_num : np.ndarray
            Numpy array of numerical data.

        Returns
        -------
        np.ndarray
            Numpy array containing the combined categorical and numerical data.

        """
        data = self.data_transformer.inverse_split_cat_num(x_cat, x_num)
        return data