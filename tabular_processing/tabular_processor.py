from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class TabularProcessor(ABC):
    """
    Abstract base class for tabular. This class is used to transform and inverse_transform
    tabular data and is used by the TabularTransformer class.
    Current Implementations:
        - IdentityProcessor (tabular_processing/identity_processor.py) 
            - This processor does not transform the data in any way.
        - BGMProcessor (tabular_processing/bgm_processor.py)
            - This processor transforms the data using a Bayesian Gaussian Mixture model.
    """
    @abstractmethod
    def __init__(self, x_cat : np.ndarray, x_num  : np.ndarray, y  : np.ndarray):
        self.x_cat = x_cat
        self.x_num = x_num
        self.y = y
        self.seed = 0
        self._was_fit = False

    @abstractmethod
    def transform(self, x_cat : np.ndarray, x_num : np.ndarray, y : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def inverse_transform(self, x_cat, x_num, y) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, meta_data:dict) -> None:
        pass

    @staticmethod
    def to_pd_DataFrame(x_cat, x_num, y, x_cat_cols, x_num_cols, y_cols):
        import pandas as pd
        x_cat_df = pd.DataFrame(x_cat, columns=x_cat_cols)
        x_num_df = pd.DataFrame(x_num, columns=x_num_cols)
        y_df = pd.DataFrame(y, columns=[y_cols] if not isinstance(y_cols, list) else y_cols)
        return pd.concat([x_cat_df, x_num_df, y_df], axis=1)

