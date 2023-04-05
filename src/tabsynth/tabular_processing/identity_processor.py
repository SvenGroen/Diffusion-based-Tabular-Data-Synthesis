from typing import Tuple
import numpy as np

from .tabular_processor import TabularProcessor



class IdentityProcessor(TabularProcessor):
    """
    A class that represents a TabularProcessor that returns the same input data.

    Parameters
    ----------
    x_cat : np.ndarray
        categorical features
    x_num : np.ndarray
        numerical features
    y : np.ndarray
        targets

    Methods
    -------
    transform(x_cat, x_num, y)
        Returns the input data.
    inverse_transform(x_cat, x_num, y_pred)
        Returns the input data.
    fit_transform(*args, **kwargs)
        Calls the transform method.
    fit(*args, **kwargs)
        Does nothing.
    """
    def __init__(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray, **kwargs):
        super().__init__(x_cat, x_num, y)


    def transform(self, x_cat:  np.ndarray, x_num  : np.ndarray, y  : np.ndarray,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the input data.

        Parameters
        ----------
        x_cat : np.ndarray
            categorical features
        x_num : np.ndarray
            numerical features
        y : np.ndarray
            targets

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The input data.
        """
        self.x_cat, self.x_num, self.y = x_cat, x_num, y
        return self.x_cat, self.x_num, self.y

    def inverse_transform(self, x_cat:np.ndarray, x_num:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        """
        Returns the input data.

        Parameters
        ----------
        x_cat : np.ndarray
            categorical features
        x_num : np.ndarray
            numerical features
        y_pred : np.ndarray
            predictions

        Returns
        -------
        np.ndarray
            The input data.
        """
        # No inverse transform is needed for the identity processor, so this method
        return x_cat, x_num, y_pred

    def fit_transform(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calls the transform method.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The input data.
        """
        # No fitting is needed for the identity processor, so this method
        # is a no-op.
        return self.transform(self.x_cat, self.x_num, self.y)

    def fit(self, *args, **kwargs) -> None:
        """
        Does nothing.
        """
        # No fitting is needed for the identity processor, so this method
        # is a no-op.
        self._was_fit = True
        return self