import pickle
import socket
from functools import partial

import optuna
import torch
import torch.nn.functional as F
from optuna import Study
from optuna._callbacks import MaxTrialsCallback
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from models import get_mlp, get_resnet, get_ft_transformer
from torch_utils import setup_training_env, get_data
from utils import time_series_split, print_trial, EarlyStopper, setup_logging

logger = setup_logging(__name__)
logger.info(f'Running on {socket.gethostname()}.')
CHECKPOINT_DIR = 'checkpoints'
DEVICE = setup_training_env()
groups, dataset = get_data(DEVICE)


def loss_fn(y_hat, y, sw):
    """Return the sample-weighted binary cross-entropy loss."""
    unweighted_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='none')
    averaged_loss = unweighted_loss.mean(1)
    return (averaged_loss * sw).mean()


@torch.no_grad()
def test_loop(model, loader):
    """Return validation loss and other evaluation metrics."""
    model.eval()
    loss = 0
    auc = 0
    n_batches = len(loader)
    for X, y, sw in loader:
        y_hat = model(X)
        loss += loss_fn(y_hat, y, sw)
        auc += roc_auc_score(y.cpu(), expit(y_hat.cpu()))
    return loss / n_batches, auc / n_batches


def train_loop(model, optimizer, loader):
    for X, y, sw in loader:
        model.train()
        optimizer.zero_grad()
        # It is numerically more stable to combine sigmoid with BCE.
        loss = loss_fn(model(X), y, sw)
        loss.backward()
        optimizer.step()


def objective(trial, get_model, disable_progress_bar=False):
    # Training parameters:
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    train_batch_size = 2048  # 1024 for ft_transformer. (4096/8192 otherwise)
    val_batch_size = 2048

    # Create model:
    model = get_model(trial, dataset, DEVICE)

    # Create optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop:
    n_epochs = int(1e3)  # We set this reasonably high and either prune or stop early.
    early_stopper = EarlyStopper()
    tr, val = time_series_split(len(dataset), groups)
    for epoch in tqdm(range(n_epochs), disable=disable_progress_bar, position=0):
        train_loader = DataLoader(Subset(dataset, tr), batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val), batch_size=val_batch_size)

        train_loop(model, optimizer, train_loader)
        val_loss, val_auc = test_loop(model, val_loader)

        # The sampler may compare to other trials, not this trial, so we need additional early stopping criteria.
        if early_stopper.should_stop(val_loss):
            logger.info('Stopping early.')
            break

        trial.report(val_loss, epoch)
        trial.set_user_attr('val_auc', val_auc)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


def save_checkpoint(study: Study, _: FrozenTrial, filename):
    with open(filename, 'wb') as f:
        pickle.dump(study.sampler, f)


if __name__ == '__main__':
    model_dict = {'mlp': get_mlp, 'resnet': get_resnet, 'ft_transformer': get_ft_transformer}
    switch = 'ft_transformer'  # {mlp, resnet, ft_transformer}
    obj = partial(objective, get_model=model_dict[switch])
    logger.info(f'Beginning {switch} study...')

    checkpoint_filename = f'{CHECKPOINT_DIR}/{switch}_sampler.pkl'
    save_checkpoint = partial(save_checkpoint, filename=checkpoint_filename)
    try:  # Restore the sampler if it exists.
        with open(checkpoint_filename, 'rb') as f:
            logger.info('Restoring sampler...')
            sampler = pickle.load(f)
    except FileNotFoundError:
        sampler = optuna.samplers.TPESampler(seed=0)

    # noinspection PyArgumentList
    study = optuna.create_study(
        study_name=f'{switch}_study',
        storage=f'sqlite:///checkpoints/{switch}_study.db',
        direction=StudyDirection.MINIMIZE,
        sampler=sampler,
        load_if_exists=True)

    n_trials = 60
    n_trials_left = n_trials - len(study.trials)  # Set n_trials_left for the progress bar.
    study.optimize(obj, show_progress_bar=True, n_trials=n_trials_left,
                   callbacks=[MaxTrialsCallback(n_trials, states=None), save_checkpoint])

    # Use `optuna-dashboard sqlite:///checkpoints/{switch}_study.db` to visualize the results.
    print_trial('Best Trial', study.best_trial)
