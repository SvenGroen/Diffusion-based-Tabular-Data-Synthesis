''' This have not been changed despite some comments and documentation.'''
import numpy as np
import os
import lib
from tab_ddpm.modules import MLPDiffusion, ResNetDiffusion

def get_model(
    model_name,
    model_params,
    n_num_features,
    category_sizes
): 
    """
    Creates and returns an instance of the specified diffusion model.

    Parameters
    ----------
    model_name : str
        Name of the model to be created. Either "mlp" or "resnet".
    model_params : dict
        A dictionary of parameters to be passed to the constructor of the specified model.
    n_num_features : int
        The number of numerical features in the input data.
    category_sizes : list of int
        The size of each categorical feature in the input data.

    Returns
    -------
    model : object
        An instance of the specified diffusion model.

    """
    print(model_name)
    if model_name == 'mlp':
        model = MLPDiffusion(**model_params)
    elif model_name == 'resnet':
        model = ResNetDiffusion(**model_params)
    else:
        raise "Unknown model!"
    return model

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using an exponential moving average.

    Parameters
    ----------
    target_params : iterator of Tensors
        The target parameter sequence.
    source_params : iterator of Tensors
        The source parameter sequence.
    rate : float, optional
        The EMA rate (closer to 1 means slower). Default is 0.999.

    Returns
    -------
    None
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def make_dataset(
    data_path: str,
    T: lib.Transformations,
    num_classes: int,
    is_y_cond: bool,
    change_val: bool,
    skip_splits: list = []
) -> lib.Dataset:
    """
    Reads and transforms the dataset from the given path, applies the provided
    transformations, and returns a `lib.Dataset` object.

    Parameters
    ----------
    data_path : str
        The path to the dataset.
    T : lib.Transformations
        The transformation to apply to the dataset.
    num_classes : int
        The number of classes for classification datasets. Set to 0 for regression datasets.
    is_y_cond : bool
        Whether the y label is concatenated with X. Only applicable for classification datasets.
    change_val : bool
        Whether to change the validation set to the test set.
    skip_splits : list, optional
        A list of splits to skip, by default []

    Returns
    -------
    lib.Dataset
        The transformed dataset.
    """

    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} 

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = lib.read_pure_data(data_path, split)
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    info = lib.load_json(os.path.join(data_path, 'info.json'))

    D = lib.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=lib.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = lib.change_val(D)
    
    return lib.transform_dataset(D, T, None, skip_splits=skip_splits)