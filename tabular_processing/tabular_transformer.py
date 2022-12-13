import json
from tempfile import TemporaryDirectory

from .dataset import TaskType, _apply_split, _make_split, _save
from .tabular_processor import TabularProcessor
from .bgm_processor import BGMProcessor
from .identity_processor import IdentityProcessor
from pathlib import Path
from typing import Tuple, Union
from enum import Enum
import lib
import numpy as np

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


SUPPORTED_PROCESSORS = {
    "identity": IdentityProcessor, 
    "bgm": BGMProcessor}

class TabularTransformer:
    def __init__(self, data_path: Union[str, Path], processor_type: str, is_y_cond: bool = False, num_classes: int = 2):
        self.data_path = data_path if isinstance(data_path, Path) else Path(data_path)
        self.config = json.load(open(self.data_path / "info.json"))
        self.x_cat = {}
        self.x_num = {}
        self.y = {}
        self.is_y_cond = is_y_cond
        self.num_classes = num_classes
        self.load_data()
        self.processor_type = processor_type if processor_type is not None else "identity"
        self.processor = self._get_processor_instance()
    
    def _get_processor_instance(self):
        if self.processor_type not in SUPPORTED_PROCESSORS:
            raise ValueError(f"Processor type {self.processor_type} is not supported.")
        params = self.config["dataset_config"]
        x_cat, x_num, y = self._get_concat_splits(splits=["train","val"])
        return SUPPORTED_PROCESSORS[self.processor_type](x_cat, x_num, y, **params)

    def _get_concat_splits(self, splits:list[str] = ["train","val"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_cat, x_num, y =  np.array([]), np.array([]), np.array([])
        for split in splits:
            x_cat = np.concatenate([x_cat, self.x_cat[split]]) if x_cat.size else self.x_cat[split]
            x_num = np.concatenate([x_num, self.x_num[split]]) if x_num.size else self.x_num[split]
            y = np.concatenate([y, self.y[split]]) if y.size else self.y[split] # maybe expand_dims?
        return x_cat, x_num, y

    def fit(self):
        self.processor.fit()
        pass

    def fit_transform(self):
        self.fit()
        self.transform()
        pass

    def transform(self):
        self.processor.fit()
        x_cat, x_num, y = self._get_concat_splits(splits=["train","val"]) # TODO: Check if no need to transform test
        train_len = self.x_cat["train"].shape[0]
        val_len = self.x_cat["val"].shape[0]
        x_cat, x_num, y = self.processor.transform(x_cat, x_num, y)
        # set transformed data
        self.x_cat["train"], self.x_cat["val"] = x_cat[:train_len], x_cat[train_len:train_len+val_len]
        self.x_num["train"], self.x_num["val"] = x_num[:train_len], x_num[train_len:train_len+val_len]
        self.y["train"], self.y["val"] = y[:train_len], y[train_len:train_len+val_len]
        pass  

    def inverse_transform(self, x_cat, x_num, y):
        try:
            x_cat = x_cat.astype(np.int64) # inverse preprocess from before might transform labels into strings
        except ValueError:
            pass
        x_cat, x_num, y = self.processor.inverse_transform(x_cat, x_num, y)
        try:
            y = y.astype(np.int64) # inverse preprocess from before might transform labels into strings
        except ValueError:
            pass
        return x_cat, x_num, y

    def load_data(self):
        # load data from data_path
        # taken from utils_train.py
        if self.num_classes > 0:
            for split in ['train', 'val', 'test']:
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
            for split in ['train', 'val', 'test']:
                x_num_t, x_cat_t, y_t = lib.read_pure_data(self.data_path, split)
                if not self.is_y_cond:
                    x_num_t = concat_y_to_X(x_num_t, y_t)
                if self.x_num is not None:
                    self.x_num[split] = x_num_t
                if self.x_cat is not None:
                    self.x_cat[split] = x_cat_t
                self.y[split] = y_t


    def save_data(self):
        # create temporary directory
        out_dir = self.data_path / self.processor_type
        out_dir.mkdir(exist_ok=True, parents=True)
        x_cat_train, x_num_train, y_train = self._get_concat_splits(splits=["train","val"])
        x_cat_test, x_num_test, y_test = self._get_concat_splits(splits=["test"])
        data = {
            k: {"test":v} for k, v in 
            (("X_num", x_num_test), 
            ("X_cat", x_cat_test), 
            ("y", y_test))
            }
        train_len = len(x_cat_train)
        data["idx"] = {"test": np.arange(
                train_len, train_len + len(x_cat_test), dtype=np.int64
            )}
        
        trainval = {"X_num": x_num_train, "X_cat": x_cat_train, "y": y_train}
        trainval_idx = _make_split(train_len, trainval["y"], 2)
        # for x in data['X_cat'].values():
        #     x[x == 'nan'] = CAT_MISSING_VALUE
        for k, v in _apply_split(trainval, trainval_idx).items():
            data[k].update(v)
        _save(out_dir, f"adult-{self.processor_type}", task_type=TaskType[self.config["dataset_config"]["problem_type"].upper()], **data)
        return out_dir
    