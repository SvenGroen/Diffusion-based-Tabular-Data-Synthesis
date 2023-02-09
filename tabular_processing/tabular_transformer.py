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
import lib
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

class TabularTransformer:
    def __init__(self, data_path: Union[str, Path], processor_type: str, is_y_cond: bool = False, num_classes: int = 2, splits: list[str] = ["train","val", "test"]):
        self.data_path = data_path if isinstance(data_path, Path) else Path(data_path)
        self.config = json.load(open(self.data_path / "info.json"))
        self.x_cat = {}
        self.x_num = {}
        self.y = {}
        self.is_y_cond = is_y_cond
        self.num_classes = num_classes
        self.load_data(splits=splits)
        self.processor_type = processor_type if processor_type is not None else "identity"
        self.processor = self._get_processor_instance()
        self.cat_values = self._get_all_category_values()
        self.dim_info={}
    
    def _get_processor_instance(self):
        if self.processor_type not in SUPPORTED_PROCESSORS:
            raise ValueError(f"Processor type {self.processor_type} is not supported.")
        print("Selected tabular processor: ", self.processor_type)
        params = self.config["dataset_config"]
        x_cat, x_num, y = self._get_concat_splits(splits=["train"]) # changed
        return SUPPORTED_PROCESSORS[self.processor_type](x_cat, x_num, y, **params)

    def _get_concat_splits(self, splits:list[str] = ["train","val"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_cat, x_num, y =  None, None, None
        for split in splits:
            x_cat = np.concatenate([x_cat, self.x_cat[split]]) if x_cat is not None else self.x_cat[split]
            x_num = np.concatenate([x_num, self.x_num[split]]) if x_num is not None else self.x_num[split]
            y = np.concatenate([y, self.y[split]]) if y is not None else self.y[split] # maybe expand_dims?
        return x_cat, x_num, y

    def _get_all_category_values(self):
        all = self.to_pd_DataFrame(splits=["train","val","test"])
        # all_col = all[self.config["dataset_config"]["cat_columns"]] # subset where only categorical columns are present
        all_cat_values = {}
        for col in self.config["dataset_config"]["cat_columns"]:
            all_cat_values[col] = all[col].unique()
        return all_cat_values

    def fit(self, reload: bool = True, save_processor: bool = True, **kwargs):
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
            return None
        return path



    def fit_transform(self,**kwargs):
        self.fit(**kwargs)
        self.transform(**kwargs)
        pass

    def to_pd_DataFrame(self, splits=["train"]):
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

        self.dim_info["original"] = save_dimensionality(self.x_cat["train"],self.x_num["train"])
        self.processor.fit(meta_data=self.cat_values)
        splits = ["train","val"]
        for split in splits:
            x_cat, x_num, y = self._get_concat_splits(splits=[split]) # TODO: Check if no need to transform test and val
            # train_len = self.x_cat["train"].shape[0]
            # val_len = self.x_cat["val"].shape[0]
            x_cat, x_num, y = self.processor.transform(x_cat, x_num, y)
            # set transformed data
            # self.x_cat["train"], self.x_cat["val"] = x_cat[:train_len], x_cat[train_len:train_len+val_len]
            # self.x_num["train"], self.x_num["val"] = x_num[:train_len], x_num[train_len:train_len+val_len]
            # self.y["train"], self.y["val"] = y[:train_len], y[train_len:train_len+val_len]
            self.x_cat[split] = x_cat
            self.x_num[split] = x_num
            self.y[split] = y

        self.dim_info["transformed"] = save_dimensionality(self.x_cat["train"],self.x_num["train"])
        return self.x_cat, self.x_num, self.y

    def inverse_transform(self, x_cat, x_num, y):
        x_num=safe_convert(x_num, np.float64)
        x_cat=safe_convert(x_cat, np.int64)

        x_cat, x_num, y = self.processor.inverse_transform(x_cat, x_num, y)
        x_num = safe_convert(x_num, np.float64)
        y = safe_convert(y, np.int64)

        return x_cat, x_num, y

    def load_data(self, splits=["train", "val", "test"]):
        # load data from data_path
        # taken and extended from utils_train.py
        
        if self.num_classes > 0:
            for split in splits:
                X_num_t, X_cat_t, y_t = lib.read_pure_data(self.data_path, split)
                if self.x_num is not None:
                    self.x_num[split] = X_num_t
                if not self.is_y_cond:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                if self.x_cat is not None:
                    self.x_cat[split] = X_cat_t
                self.y[split] = y_t
        else:
        # regression
            for split in splits:
                x_num_t, x_cat_t, y_t = lib.read_pure_data(self.data_path, split)
                if not self.is_y_cond:
                    x_num_t = concat_y_to_X(x_num_t, y_t)
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

        # trainval_idx = _make_split(train_len, trainval["y"], 2)
        # for x in data['X_cat'].values():
        #     x[x == 'nan'] = CAT_MISSING_VALUE
        # for k, v in _apply_split(trainval, trainval_idx).items():
        #     data[k].update(v)
        if data["X_cat"]["train"] is None:
            data["X_cat"] = None
        if data["X_num"]["train"] is None:
            data["X_num"] = None
        if data["y"]["train"] is None:
            data["y"] = None
        _save(out_dir, f"{self.config['name'].lower()}-{self.processor_type}", task_type=TaskType[self.config["dataset_config"]["problem_type"].upper()], **data)
        return out_dir

def save_dimensionality(x_cat, x_num):
        num_dim = x_num.shape[1] if x_num is not None else -1
        cat_dim = x_cat.shape[1] if x_cat is not None else -1
        return {"num_dim": num_dim, "cat_dim": cat_dim}

def safe_convert(x, dtype):
    if x is not None:
        try:
            return x.astype(dtype)
        except ValueError:
            return x
    return x