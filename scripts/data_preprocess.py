import argparse
import json
import pandas as pd
import numpy as np
import catboost.datasets
from pathlib import Path
from tabular_processing.dataset import TaskType, _make_split, _apply_split, _save
from tabular_processing.bgm_processing.bgm_transformer import BGMTransformer
CAT_MISSING_VALUE = '__nan__'

def preprocess(data_folder, config, preprocess_method):
    
    df_trainval, df_test = load_dataset(name=config["name"], data_folder=data_folder, download=True)
    data=pd.concat([df_trainval, df_test], axis=0) # maybe use only train_data?
    data_config = config["dataset_config"]

    if preprocess_method.lower() == "bgm":
        transformer = BGMTransformer(data.copy(), **data_config)
        transformer.fit_transform()
        x_cat, x_num, y = transformer.get_cat_num_y()
        x_cat_train = x_cat[:len(df_trainval)]
        x_cat_test = x_cat[len(df_trainval):]
        train_len = len(df_trainval)
        data = {
            k: {"test":v} for k, v in 
            (("X_num", x_num[train_len:]), 
            ("X_cat", x_cat[train_len:]), 
            ("y", y[train_len:]))
            }
        data["idx"] = {"test": np.arange(
                train_len, train_len + len(df_test), dtype=np.int64
            ) }
        train = {"X_num": x_num[:train_len], "X_cat": x_cat[:train_len], "y": y[:train_len]}
        train_idx = _make_split(train_len, train["y"], 2)
        # for x in data['X_cat'].values():
        #     x[x == 'nan'] = CAT_MISSING_VALUE
        for k, v in _apply_split(train, train_idx).items():
            data[k].update(v)

        _save(data_folder, "adult", task_type=TaskType[data_config["problem_type"].upper()], **data)

        # save


        data_ = transformer.inverse_transform(x_cat, x_num, y_pred=y)
        # todo: save the transformer data + split



    pass

def load_dataset(name, data_folder, download=False, filename=None):
    if name.lower() == "adult":
        if not download:
            path = data_folder / "adult.data" if filename is None else filename
            columns = ["age","workclass","fnlwgt", "education", "education-num",
                        "marital-status", "occupation", "relationship", 
                        "race", "sex","capital-gain", "capital-loss", "hours-per-week",
                        "native-country", "income"]
            data = pd.read_csv(path, names=columns, na_values="?")
            return data, None
        else:
            df_trainval, df_test = catboost.datasets.adult()
            assert (df_trainval.dtypes == df_test.dtypes).all()
            assert (df_trainval.columns == df_test.columns).all()
            return df_trainval, df_test

    return df_trainval, df_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', metavar='FILE')
    parser.add_argument('preprocess_method', type=str, default="bgm")
    args = parser.parse_args()
    data_folder = Path(args.data_folder)
    raw_config = json.load(open(data_folder / "info.json"))
    preprocess_method = args.preprocess_method

    preprocess(data_folder=data_folder, config=raw_config, preprocess_method=preprocess_method)

if __name__ == '__main__':
    main()
