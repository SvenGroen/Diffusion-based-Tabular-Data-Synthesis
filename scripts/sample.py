import torch
import numpy as np
import zero
import os
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tab_ddpm.utils import FoundNANsError
from tabular_processing.tabular_data_controller import TabularDataController
from utils_train import get_model, make_dataset
from lib import round_columns
import lib

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False,
    processor_type = None
):
    """
    Generate synthetic tabular data from a pre-trained model using Gaussian-Multinomial Diffusion.

    The function generates synthetic samples from a pre-trained model and saves the generated data to
    an output directory. It uses Gaussian-Multinomial Diffusion to sample the data and handles
    numerical and categorical features separately, performing necessary preprocessing and inverse
    transformations. The function also supports generating data with class disbalance.

    Parameters
    ----------
    parent_dir : str
        The directory where generated data will be saved.
    real_data_path : str, optional
        The path to the real dataset, defaults to 'data/higgs-small'.
    batch_size : int, optional
        The batch size for data generation, defaults to 2000.
    num_samples : int, optional
        The number of samples to generate, defaults to 0.
    model_type : str, optional
        The type of the model, defaults to 'mlp'. Options: ["resnet", "mlp"].
    model_params : dict, optional
        The parameters of the model, defaults to None.
    model_path : str, optional
        The path to the pre-trained model, defaults to None.
    num_timesteps : int, optional
        The number of timesteps for the diffusion process, defaults to 1000.
    gaussian_loss_type : str, optional
        The type of Gaussian loss, defaults to 'mse'. Options: ["mse", "kl"].
    scheduler : str, optional
        The scheduler for the diffusion process, defaults to 'cosine'. Options: ["linear", "cosine"].
    T_dict : dict, optional
        The transformations dictionary, defaults to None.
    num_numerical_features : int, optional
        The number of numerical features in the dataset, defaults to 0.
    disbalance : str, optional
        The disbalance mode, can be 'fix' or 'fill', defaults to None.
    device : torch.device, optional
        The device to run the function on, defaults to 'cuda:1'.
    seed : int, optional
        The random seed, defaults to 0.
    change_val : bool, optional
        Whether to change the validation set or not, defaults to False.
    processor_type : str, optional
        The type of the data processor, defaults to None.

    Returns
    -------
    None
        This function does not return any value. It saves the generated data to the output directory.
    """
    zero.improve_reproducibility(seed)

    # add TabularDataController to revert transform synthetic sampled data at the end
    tabular_controller = TabularDataController(
        real_data_path, 
        processor_type,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'])
    
    # also transform because information from transformation process might be needed
    # will be reloaded if TabularProcessor is already saved
    tabular_controller.fit_transform(reload = True, save_processor = True) 
    real_data_path = os.path.join(real_data_path, processor_type)

    # create dataset and load transformations
    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val,
        skip_splits=["val","test"]
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    # get model
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    # load model state
    try:
        model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        print("Model Loaded Successfully!")
    except FileNotFoundError:
        print('-------->Model not found<--------')
        

    # load diffusion process
    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()
    
    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)


    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
    X_cat, X_num = None, None
    
    # Reverse the "standard" preprocessing transformations
    num_numerical_features = tabular_controller.dim_info["transformed"]["num_dim"] + int(D.is_regression and not model_params["is_y_cond"])
     
    X_num_ = X_gen
    if tabular_controller.dim_info["transformed"]["cat_dim"] > 0: # Changed: if transformed data has categorical features, transform them back
    # Old: if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0: # Changed
        # Old: _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]
        X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    # Inverse TabularProcessing transformations
    X_cat, X_num, y_gen = tabular_controller.inverse_transform(X_cat, X_num, y_gen) 

    # Save for Evaluation
    print("Saving Synthetic Data at: ", str(parent_dir))
    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
    if X_cat is not None and X_cat.shape[1] > 0:
        np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    np.save(os.path.join(parent_dir, 'y_train'), y_gen)