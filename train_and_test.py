import socket
import types
from functools import partial

import torch
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from torch_utils import setup_training_env, get_data
from tune import train_loop, test_loop
from utils import time_series_split, EarlyStopper, setup_logging

logger = setup_logging(__name__)
logger.info(f'Running on {socket.gethostname()}.')
CHECKPOINT_DIR = 'checkpoints'
DEVICE = setup_training_env()
groups, dataset = get_data(DEVICE)


#%% MLP setup
n_epochs = 60
lr = 2e-5
weight_decay = 1.2e-6
train_batch_size = 4096
val_batch_size = 8192

model = MLP(
    d_in=len(dataset[0][0]),  # Number of features
    d_out=len(dataset[0][1]),  # Number of labels per sample
    n_blocks=7,
    d_block=400,
    dropout=0.435).to(DEVICE)

#%% ResNet setup
n_epochs = 30
lr = 1.6e-5
weight_decay = 2.24e-6
train_batch_size = 4096
val_batch_size = 8192

model = ResNet(
    d_in=len(dataset[0][0]),
    d_out=len(dataset[0][1]),
    n_blocks=1,
    d_block=437,
    d_hidden=None,  # We set d_hidden as a factor of d_block (the dim of the residual block) below.
    d_hidden_multiplier=3.33,
    dropout1=0.435,
    dropout2=0.489).to(DEVICE)

#%% FTTransformer setup
n_epochs = 20
lr = 0.00057
weight_decay = 7.23e-6
train_batch_size = 2048
val_batch_size = 2048

params = FTTransformer.get_default_kwargs()
overrides = {
    'n_blocks': 3,
    'd_block': 440,
    'attention_dropout': 0.123,
    'ffn_d_hidden_multiplier': 2,
    'ffn_dropout': 0.302,
    'residual_dropout': 0.0009
}
params.update(overrides)
model = FTTransformer(
    n_cont_features=len(dataset[0][0]),
    cat_cardinalities=[],  # No categorical features
    d_out=len(dataset[0][1]),
    **params).to(DEVICE)
model.forward = types.MethodType(partial(FTTransformer.forward, x_cat=None), model)

#%% Training and test
model_type = type(model).__name__
logger.info(f'Beginning {model_type} training...')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
early_stopper = EarlyStopper()
tr, te = time_series_split(len(dataset), groups)
train_loader = DataLoader(Subset(dataset, tr), batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(Subset(dataset, te), batch_size=val_batch_size)
for epoch in tqdm(range(n_epochs), position=0):
    train_loop(model, optimizer, train_loader)
    loss, auc = test_loop(model, test_loader)
    print(f"Test metrics for {model_type}:")
    print(f"  Loss: {loss}")
    print(f"  AUC: {auc}")
    torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/{model_type}.ckpt')
