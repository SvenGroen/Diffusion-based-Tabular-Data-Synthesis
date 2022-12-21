from typing import Tuple
import numpy as np

from .tabular_processor import TabularProcessor



class IdentityProcessor(TabularProcessor):
    def __init__(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray, **kwargs):
        super().__init__(x_cat, x_num, y)


    def transform(self, x_cat:  np.ndarray, x_num  : np.ndarray, y  : np.ndarray,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.x_cat, self.x_num, self.y = x_cat, x_num, y
        return self.x_cat, self.x_num, self.y

    def inverse_transform(self, x_cat:np.ndarray, x_num:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        # No inverse transform is needed for the identity processor, so this method
        return x_cat, x_num, y_pred

    def fit_transform(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # No fitting is needed for the identity processor, so this method
        # is a no-op.
        return self.transform(self.x_cat, self.x_num, self.y)

    def fit(self, *args, **kwargs) -> None:
        # No fitting is needed for the identity processor, so this method
        # is a no-op.
        return self