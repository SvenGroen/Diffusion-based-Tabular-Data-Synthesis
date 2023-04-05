from copy import deepcopy
import torch
import os
import numpy as np
import zero
from tabsynth.tab_ddpm import GaussianMultinomialDiffusion
from tabsynth.scripts.utils_train import get_model, make_dataset, update_ema
from tabsynth import lib
import pandas as pd
from azureml.core import Run
from tabsynth.tabular_processing.tabular_data_controller import TabularDataController

class Trainer:
    """
    Trainer class for training a GaussianMultinomialDiffusion model.

    Parameters
    ----------
    diffusion : GaussianMultinomialDiffusion object
        The diffusion object containing the denoising function and other attributes.
    train_iter : DataLoader object
        The training data iterator.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    steps : int
        Number of steps to train the model.
    device : torch.device
        Device to use for training (default is 'cuda:1').

    Attributes
    ----------
    ema_model : deepcopy of the denoising function of diffusion object
    optimizer : torch.optim.AdamW object
        Optimizer for training the model.
    loss_history : pandas DataFrame object
        A DataFrame object to store the history of the training losses.
    log_every : int
        The interval of steps to log the training loss.
    print_every : int
        The interval of steps to print the training loss.
    ema_every : int
        The interval of steps to update the exponential moving average (EMA) model.

    Methods
    -------
    _anneal_lr(step)
        Anneals the learning rate based on the current step.
    _run_step(x, out_dict)
        Runs a single training step and updates the model parameters.
    run_loop()
        Runs the training loop.

    """
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        # run = Run.get_context()
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            # run.log("mloss", np.around(curr_loss_multi / curr_count, 4))
            # run.log("gloss", np.around(curr_loss_gauss / curr_count, 4))
            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1

def train(
    parent_dir,
    real_data_path = 'data/higgs-small',
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False,
    processor_type = None
):
    """
    Train a GaussianMultinomialDiffusion model on tabular data.

    Parameters
    ----------
    parent_dir : str
        The directory where the model and loss history will be saved.
    real_data_path : str, optional
        The path to the tabular data, defaults to 'data/higgs-small'.
    steps : int, optional
        The number of steps to train the model, defaults to 1000.
    lr : float, optional
        Learning rate for the optimizer, defaults to 0.002.
    weight_decay : float, optional
        Weight decay for the optimizer, defaults to 1e-4.
    batch_size : int, optional
        Batch size for training, defaults to 1024.
    model_type : str, optional
        Type of the model architecture to use, defaults to 'mlp'. Options: ["resnet", "mlp"].
    model_params : dict, optional
        Dictionary containing the parameters for the model architecture, defaults to None.
    num_timesteps : int, optional
        Number of diffusion steps, defaults to 1000.
    gaussian_loss_type : str, optional
        Type of the Gaussian loss to use, defaults to 'mse'. Options: ["mse", "kl"].
    scheduler : str, optional
        Type of scheduler to use, defaults to 'cosine'. Options: ["linear", "cosine"].
    T_dict : dict, optional
        Dictionary containing the transformation parameters, defaults to None.
    num_numerical_features : int, optional
        Number of numerical features in the tabular data, defaults to 0.
    device : torch.device, optional
        Device to use for training, defaults to 'cuda:1'.
    seed : int, optional
        Seed for reproducibility, defaults to 0.
    change_val : bool, optional
        Whether to change the validation data, defaults to False.
    processor_type : str, optional
        Type of processor to use, defaults to None.

    Returns
    -------
    None
        This function does not return any value. It saves the model and loss history to the parent directory.
    """

    # normalize paths
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    # improve reproducibility
    zero.improve_reproducibility(seed)

    # add data processing
    tabular_controller = TabularDataController(
        real_data_path, 
        processor_type,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'])
    tabular_controller.fit_transform(reload = True, save_processor = True)
    real_data_path = tabular_controller.save_data()


    # Set up dataset and model
    T = lib.Transformations(**T_dict)
    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val,
        skip_splits=["val", "test"]
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in

    # create model
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    ) # maybe later add U-Net 1D
    model.to(device)

    # Prepare data loader
    # train_loader = lib.prepare_beton_loader(dataset, split='train', batch_size=batch_size)
    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    # Set up diffusion model
    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()
    
    # Set up trainer class and run training loop
    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()
    
    # Save model and loss history
    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    try:
        print("Found files in " + str(parent_dir) + ": ")
        print(os.listdir(str(parent_dir)))
        import shutil
        shutil.copyfile(str(parent_dir + "/model.pt"), "outputs/model.pt")
        shutil.copyfile(str(parent_dir + "/model_ema.pt"), "outputs/model_ema.pt")
        print("Saved model to outputs folder")
    except Exception as e:
        print(e)

