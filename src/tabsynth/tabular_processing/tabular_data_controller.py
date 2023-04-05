import json
from tempfile import TemporaryDirectory

from .dataset import TaskType, _apply_split, _make_split, _save
from .tabular_processor import TabularProcessor
from .bgm_processor import BGMProcessor
from .identity_processor import IdentityProcessor
from .ft_processor import FTProcessor
from pathlib import Path
from typing import Tuple, Union
from enum import Enum
from tabsynth import  lib
import pickle
import numpy as np

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


SUPPORTED_PROCESSORS = {
    "identity": IdentityProcessor, 
    "bgm": BGMProcessor,
    "ft": FTProcessor
    }

class TabularDataController:
    """
    The TabularDataController class is the "context" of the "design strategy pattern" in the given code. (with the TabularProcessor class as the "strategy", see https://en.wikipedia.org/wiki/Strategy_pattern)
    It is responsible for loading, processing, transforming, and saving tabular data. 
    It takes care of loading the data from the specified directory and then concatenates it for the specified splits.
    Splits refer to the splitted dataset, that is, the training, validation, and test splits. 
    After loading the data, it applies the specified tabular processor strategy to the concatenated splits of the tabular data. 
    The tabular processor is chosen based on the processor_type parameter passed to the constructor of the class.

    The TabularDataController class also provides methods to fit and fit_transform the tabular data. 
    These methods allow the context to fit the tabular processor with the available data before transformation. 
    The fit method fits the tabular processor with the given tabular data, while the fit_transform method first fits the processor with the data and then transforms the data using the fitted processor.
    The fit method ensures, that the tabular processor does not have access to the test data split.
    
    The class follows the design strategy pattern, where the context (the TabularDataController) delegates processing responsibilities to the chosen tabular processor.

    Parameters
    ----------
    data_path : str or Path object
        path to the directory containing the data
    processor_type : str
        type of processor to be used
    num_classes : int
        number of classes (default is 2)
    splits : list of strings   
        list of splits (default is ["train", "val", "test"])
    cat_values : dict
        dictionary containing all unique categorical values for the entire dataset

    Attributes
    ----------
    data_path : str or Path object
        path to the directory containing the data
    processor_type : str
        type of processor to be used
    num_classes : int
        number of classes (default is 2)
    splits : list of strings
        list of splits (default is ["train", "val", "test"])
    cat_values : dict
        dictionary containing all unique categorical values for the entire dataset
    x_cat : dict
        dictionary containing categorical features for each split
    x_num : dict
        dictionary containing numerical features for each split
    y : dict
        dictionary containing labels for each split
    dim_info : dict
        dictionary containing information about the dimensionality of the data before and after transformation

    Methods
    -------
    fit(reload=True, save_processor=True):
        Fits the processor.
    load_processor(path="./processor_state/", filename=None):
        Loads the processor from a file.
    save_processor(path="./processor_state/"):
        Saves the processor to a file.
    fit_transform():
        Fits the processor and transforms the data.
    to_pd_DataFrame(splits=["train"]):
        Converts the data to a pandas DataFrame.
    transform():
        Transforms the data.
    inverse_transform(x_cat, x_num, y):
        Transforms the data back to its original form.
    load_data(splits=["train", "val", "test"]):
        Loads the data from a file.
    save_data():
        Saves the data to a file.
    _get_processor_instance():
        Returns an instance of the processor.
    _get_concat_splits(splits=["train", "val"]):
        Concatenates the features and labels for the given splits.
    _get_all_category_values():
        Returns a dictionary containing all unique categorical values for the entire dataset.
    """
    def __init__(self, data_path: Union[str, Path], processor_type: str, num_classes: int = 2, splits: list[str] = ["train","val", "test"], **kwargs):
        self.data_path = data_path if isinstance(data_path, Path) else Path(data_path)
        self.config = json.load(open(self.data_path / "info.json"))
        self.x_cat = {}
        self.x_num = {}
        self.y = {}
        # self.is_y_cond = is_y_cond
        self.num_classes = num_classes
        self.load_data(splits=splits)
        self.processor_type = processor_type if processor_type is not None else "identity"
        self.processor = self._get_processor_instance()
        self.cat_values = self._get_all_category_values()
        self.dim_info={}
    
    def _get_processor_instance(self):
        """
        Returns a new instance of a processor based on the processor type specified in the constructor.
        Raises a ValueError if the processor type is not supported.
        If the processor was previously fitted and saved to disk, it will be loaded instead of creating a new instance.
        If the processor was not loaded, the constructor parameters specified in the dataset configuration info.json file will be used.

        Returns
        -------
        Processor
        An instance of a supported processor based on the processor_type specified in the constructor.
        """
        if self.processor_type not in SUPPORTED_PROCESSORS:
            raise ValueError(f"Processor type {self.processor_type} is not supported.")
        print("Selected tabular processor: ", self.processor_type)
        params = self.config["dataset_config"]
        x_cat, x_num, y = self._get_concat_splits(splits=["train"]) # changed (only allow processor to see train data)
        return SUPPORTED_PROCESSORS[self.processor_type](x_cat, x_num, y, **params)

    def _get_concat_splits(self, splits:list[str] = ["train","val"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Concatenates the data of specified splits.

        Parameters
        ----------
        splits : list of str, optional
            List of splits to concatenate (default is ["train", "val"])

        Returns
        -------
        tuple of np.ndarray
            Returns a tuple of three arrays, x_cat, x_num, y. Each array contains the concatenated data from the specified splits.

        Raises
        ------
        ValueError
            If the specified split does not exist in the object.
        """
        x_cat, x_num, y =  None, None, None
        for split in splits:
            x_cat = np.concatenate([x_cat, self.x_cat[split]]) if x_cat is not None else self.x_cat[split]
            x_num = np.concatenate([x_num, self.x_num[split]]) if x_num is not None else self.x_num[split]
            y = np.concatenate([y, self.y[split]]) if y is not None else self.y[split] # maybe expand_dims?
        return x_cat, x_num, y

    def _get_all_category_values(self):
        """
        Get unique category values across all splits of the dataset.

        Returns
        -------
        all_cat_values: dict
            A dictionary where the keys are the categorical column names and the values
            are the unique category values across all splits.
        """
        all = self.to_pd_DataFrame(splits=["train","val","test"])
        # all_col = all[self.config["dataset_config"]["cat_columns"]] # subset where only categorical columns are present
        all_cat_values = {}
        for col in self.config["dataset_config"]["cat_columns"]:
            all_cat_values[col] = all[col].unique()
        return all_cat_values

    def fit(self, reload: bool = True, save_processor: bool = True, **kwargs):
        """
        Fits the data processor on the training set of the data and saves the trained processor if required.

        Parameters
        ----------
        reload : bool, optional
            If True, loads previously saved processor if available (default is True).
        save_processor : bool, optional
            If True, saves the trained processor (default is True).
        **kwargs
            Other arguments that can be passed to the `fit()` method of the processor.

        Returns
        -------
        None
        """
        was_loaded = False # to also save unnecessary saving if model was just loaded
        if reload:
            try:
                self.processor = self.load_processor()
                was_loaded = True
            except FileNotFoundError as e:
                print("Error while loading processor state, file was not found: ", e)
                
        if not was_loaded:
            print("Fitting processor")
            self.processor.fit(meta_data=self.cat_values)
        if save_processor and not was_loaded:
            self.save_processor()
        pass


    def load_processor(self, path: Union[str, Path]="./processor_state/", filename: str=None):
        """
        Loads the `TabularProcessor` object from a file at the specified path using pickle.
        
        Parameters
        ----------
        path : str or Path, optional
            Path to the directory where the processor state file is located (default is "./processor_state/").
        filename : str, optional
            The name of the processor state file (default is None).

        Returns
        -------
        processor : object
            The loaded processor state object.

        Raises
        ------
        FileNotFoundError
            If the specified file is not found.

        Example
        -------
        >>> controller = TabularDataController(...)
        >>> processor = controller.load_processor(path="./saved_processor_state/", filename="processor_ft.pkl")
        Loaded processor of type ft state from: ./saved_processor_state/processor_ft.pkl
        """
        path = path if isinstance(path, Path) else Path(path)
        if filename is None:
            filename = f"processor_{self.processor_type}.pkl"
        path = path / filename
        # load with pickle
        try:
            processor = pickle.load(open(path, "rb"))
            print(f"Loaded processor of type {self.processor_type} state from: ", path)
        except Exception as e:
            print("Error while loading processor state: ", e)
            raise e       
        return processor

    def save_processor(self, path: Union[str, Path]="./processor_state/"):
        """
        Saves the current processor instance using pickle.

        Parameters
        ----------
        path : str
            The path where the processor instance should be saved.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the processor instance has not been fit.
        """
        path = path if isinstance(path, Path) else Path(path)
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"processor_{self.processor_type}.pkl"
        # save with pickle
        try:
            with open(path, "wb") as f:
                pickle.dump(self.processor, f)
                print("Saved processor state to: ", path)
        except Exception as e:
            raise e
        return path

    def fit_transform(self,**kwargs):
        """
        Fit the preprocessor to the data and then transform it.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the fit() and transform() methods.

        """
        self.fit(**kwargs)
        self.transform(**kwargs)
        pass

    def to_pd_DataFrame(self, splits=["train"]):
        """
        Converts the internal data matrix to a pandas DataFrame object.

        Returns
        -------
        pandas.DataFrame
            A DataFrame object representing the data matrix.

        Raises
        ------
        ValueError
            If the internal data matrix is empty.
        """
        cat_cols = self.config["dataset_config"]["cat_columns"]
        num_cols = self.config["dataset_config"]["int_columns"]
        y_col = self.config["dataset_config"]["target_column"]
        # remove target column from cat_cols or num_cols
        if y_col in cat_cols:
            cat_cols = [col for col in cat_cols if col != y_col]
        else:
            num_cols = [col for col in num_cols if col != y_col]
        x_cat, x_num, y = self._get_concat_splits(splits=splits)
        df = self.processor.to_pd_DataFrame(x_cat, x_num, y, cat_cols, num_cols, y_col)
        return df

    def transform(self, **kwargs):
        """
        Transforms the data using the processor transform function.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to be passed to the fit() and transform() methods.

        Returns
        -------
        x_cat : dict
            A dictionary containing the transformed categorical feature matrix of each split.
        x_num : dict
            A dictionary containing the transformed numerical feature matrix of each split.
        y : dict
            A dictionary containing the target variable of each split.
        """

        self.dim_info["original"] = save_dimensionality(self.x_cat["train"],self.x_num["train"])
        splits = ["train","val"]
        for split in splits:
            x_cat, x_num, y = self._get_concat_splits(splits=[split]) # TODO: Check if no need to transform test and val
            x_cat, x_num, y = self.processor.transform(x_cat, x_num, y)
            self.x_cat[split] = x_cat
            self.x_num[split] = x_num
            self.y[split] = y

        self.dim_info["transformed"] = save_dimensionality(self.x_cat["train"],self.x_num["train"])
        return self.x_cat, self.x_num, self.y

    def inverse_transform(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inverse transform the processed numerical and categorical data along with target variable 'y',
        returning the original input data.

        Parameters
        ----------
        x_cat : numpy.ndarray
            Processed categorical data.
        x_num : numpy.ndarray
            Processed numerical data.
        y : numpy.ndarray
            Target variable.

        Returns
        -------
        numpy.ndarray
            Original categorical data.
        numpy.ndarray
            Original numerical data.
        numpy.ndarray
            Original target variable.
        """
        x_num=safe_convert(x_num, np.float64)
        x_cat=safe_convert(x_cat, np.int64)

        x_cat, x_num, y = self.processor.inverse_transform(x_cat, x_num, y)
        x_num = safe_convert(x_num, np.float64)
        y = safe_convert(y, np.int64)

        return x_cat, x_num, y

    def load_data(self, splits:list=["train", "val", "test"]):
        """
        Load data from the data path and store it in memory.

        If the `num_classes` attribute is greater than 0, assume classification and load the categorical and numerical 
        data for each split. Otherwise, assume regression data.

        Parameters
        ----------
        splits : list of str, optional
            A list of the names of the splits to load. Default is ["train", "val", "test"].

        Returns
        -------
        None
        """
        # load data from data_path
        # taken and extended from utils_train.py
        if self.num_classes > 0:
            for split in splits:
                X_num_t, X_cat_t, y_t = lib.read_pure_data(self.data_path, split)
                if self.x_num is not None:
                    self.x_num[split] = X_num_t
                # if not self.is_y_cond:
                #     X_cat_t = concat_y_to_X(X_cat_t, y_t)
                if self.x_cat is not None:
                    self.x_cat[split] = X_cat_t
                self.y[split] = y_t
        else:
        # regression
            for split in splits:
                x_num_t, x_cat_t, y_t = lib.read_pure_data(self.data_path, split)
                # if not self.is_y_cond:
                #     x_num_t = concat_y_to_X(x_num_t, y_t)
                if self.x_num is not None:
                    self.x_num[split] = x_num_t
                if self.x_cat is not None:
                    self.x_cat[split] = x_cat_t
                self.y[split] = y_t

        # remaining splits are empty 
        all_splits = ["train", "val", "test"]
        remain = [split for split in all_splits if split not in splits]
        for split in remain:
            self.x_cat[split] = np.empty_like(self.x_cat[splits[-1]])
            self.x_num[split] = np.empty_like(self.x_num[splits[-1]])
            self.y[split] = np.empty_like(self.y[splits[-1]])


    def save_data(self):
        """
        Save preprocessed data using the `processor_type` and `data_path`.

        Returns
        -------
        Path
            The path where the data is saved.
        """
        # create temporary directory
        out_dir = self.data_path / self.processor_type
        out_dir.mkdir(exist_ok=True, parents=True)
        x_cat_train, x_num_train, y_train = self._get_concat_splits(splits=["train"]) #changed
        x_cat_val, x_num_val, y_val = self._get_concat_splits(splits=["val"])
        x_cat_test, x_num_test, y_test = self._get_concat_splits(splits=["test"])
        test = {
            k: {"test":v} for k, v in 
            (("X_num", x_num_test), 
            ("X_cat", x_cat_test), 
            ("y", y_test))
            }
        val = {
            k: {"val":v} for k, v in 
            (("X_num", x_num_val), 
            ("X_cat", x_cat_val), 
            ("y", y_val))
            }
        train = {
            k: {"train":v} for k, v in 
            (("X_num", x_num_train), 
            ("X_cat", x_cat_train), 
            ("y", y_train))
            }
        data = {split:test[split] | val[split] | train[split] for split in ["X_num", "X_cat", "y"]}
        # data = {split: val[split] | train[split] for split in ["X_num", "X_cat", "y"]}
        
        train_len = len(x_cat_train) if x_cat_train is not None else 0
        val_len = len(x_cat_val) if x_cat_val is not None else 0
        train_val_len = train_len + val_len
        data["idx"] = {"test": np.arange(
                train_val_len, train_val_len + len(x_cat_test), dtype=np.int64
            )}

        data["idx"].update({"train": np.arange(train_len, dtype=np.int64), "val": np.arange(train_len, train_len + val_len, dtype=np.int64)})
        if data["X_cat"]["train"] is None:
            data["X_cat"] = None
        if data["X_num"]["train"] is None:
            data["X_num"] = None
        if data["y"]["train"] is None:
            data["y"] = None
        _save(out_dir, f"{self.config['name'].lower()}-{self.processor_type}", task_type=TaskType[self.config["dataset_config"]["problem_type"].upper()], **data)
        return out_dir

def save_dimensionality(x_cat, x_num):
    """
    Saves the number of dimensions of categorical and numerical features in the given data.

    Parameters
    ----------
    x_cat : numpy.ndarray
        The categorical features of the data.
    x_num : numpy.ndarray
        The numerical features of the data.

    Returns
    -------
    dict
        A dictionary containing the number of dimensions of the categorical and numerical features.
    """
    num_dim = x_num.shape[1] if x_num is not None else -1
    cat_dim = x_cat.shape[1] if x_cat is not None else -1
    return {"num_dim": num_dim, "cat_dim": cat_dim}

def safe_convert(x, dtype):
    """
    Safely converts an array to the specified data type, catching any ValueError that may occur.

    Parameters
    ----------
    x : numpy.ndarray
        The array to convert.
    dtype : type
        The data type to convert the array to.

    Returns
    -------
    numpy.ndarray or None
        The converted array if it was possible, otherwise None.
    """
    if x is not None:
        try:
            return x.astype(dtype)
        except ValueError:
            return x
    return x