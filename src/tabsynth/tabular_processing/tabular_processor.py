from abc import ABC, abstractmethod
from typing import Tuple, Union, List
import numpy as np


class TabularProcessor(ABC):
    """
    Abstract base class for a tabular processor. This class is used to transform and inverse_transform
    tabular data and is used by the TabularDataController class.
    Current Implementations:
        - IdentityProcessor (tabular_processing/identity_processor.py) 
            - This processor does not transform the data in any way.
        - BGMProcessor (tabular_processing/bgm_processor.py)
            - This processor transforms the data using a Bayesian Gaussian Mixture model.
        - FTProcessor (tabular_processing/ft_processor.py)
            - This processor transforms the data using a Feature Tokenization approach.

    Attributes
    ----------
    x_cat : numpy.ndarray
        Categorical features of the data.
    x_num : numpy.ndarray
        Numerical features of the data.
    y : numpy.ndarray
        Target variable.
    seed : int
        Seed for random operations.
    _was_fit : bool
        Flag indicating if the processor was fitted.

    Methods
    -------
    __init__(x_cat, x_num, y)
        Constructs all the necessary attributes for the TabularProcessor object.
    transform(x_cat, x_num, y) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Transforms the tabular data.
    inverse_transform(x_cat, x_num, y) -> numpy.ndarray
        Inverse transforms the tabular data.
    fit(meta_data) -> None
        Fits the processor with given metadata.
    to_pd_DataFrame(x_cat, x_num, y, x_cat_cols, x_num_cols, y_cols) -> pandas.DataFrame
        Converts tabular data to a Pandas DataFrame.
    """
    
    @abstractmethod
    def __init__(self, x_cat : np.ndarray, x_num  : np.ndarray, y  : np.ndarray):
        """
        Initializes the TabularProcessor instance with the given tabular data.
        
        Parameters:
        -----------
        x_cat : numpy.ndarray
            Categorical features of the tabular data.
        x_num : numpy.ndarray
            Numerical features of the tabular data.
        y : numpy.ndarray
            Labels of the tabular data.
        """
        self.x_cat = x_cat
        self.x_num = x_num
        self.y = y
        self.seed = 0
        self._was_fit = False

    @abstractmethod
    def transform(self, x_cat : np.ndarray, x_num : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms the given tabular data using the implemented transformation strategy.
        
        Parameters:
        -----------
        x_cat : numpy.ndarray
            Categorical features of the tabular data.
        x_num : numpy.ndarray
            Numerical features of the tabular data.
        y : numpy.ndarray
            Labels of the tabular data.
        
        Returns:
        --------
        A tuple containing transformed x_cat, x_num, and y.
        """
        pass

    @abstractmethod
    def inverse_transform(self, x_cat, x_num, y) -> np.ndarray:
        """
        Reverses the transformation applied to the tabular data using the implemented strategy.
        
        Parameters:
        -----------
        x_cat : numpy.ndarray
            Transformed categorical features of the tabular data.
        x_num : numpy.ndarray
            Transformed numerical features of the tabular data.
        y : numpy.ndarray
            Transformed labels of the tabular data.
            
        Returns:
        --------
        The original tabular data before the transformation was applied.
        """
        pass

    @abstractmethod
    def fit(self, meta_data:dict) -> None:
        """
        Fits the implemented strategy to the tabular data.
        
        Parameters:
        -----------
        meta_data : dict
            Metadata used by the transformation strategy during fitting.
        """
        pass

    @staticmethod
    def to_pd_DataFrame(x_cat : np.ndarray, x_num  : np.ndarray, y  : np.ndarray, x_cat_cols :list, x_num_cols:list, y_cols: Union[str, List[str]]):
        """
        Converts the tabular data to a pandas DataFrame.
        
        Parameters:
        -----------
        x_cat : numpy.ndarray
            Categorical features of the tabular data.
        x_num : numpy.ndarray
            Numerical features of the tabular data.
        y : numpy.ndarray
            Labels of the tabular data.
        x_cat_cols : list
            List of column names for categorical features.
        x_num_cols : list
            List of column names for numerical features.
        y_cols : str or list
            Column name(s) for the label(s).
            
        Returns:
        --------
        A pandas DataFrame representation of the tabular data.
        """
        import pandas as pd
        x_cat_df = pd.DataFrame(x_cat, columns=x_cat_cols)
        x_num_df = pd.DataFrame(x_num, columns=x_num_cols)
        y_df = pd.DataFrame(y, columns=[y_cols] if not isinstance(y_cols, list) else y_cols)
        return pd.concat([x_cat_df, x_num_df, y_df], axis=1)

