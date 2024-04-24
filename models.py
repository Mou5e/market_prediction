import types
from functools import partial

import keras
import torch.nn as nn
from keras import layers, Model
from optuna import Trial
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from torch.types import Device
from torch.utils.data import Dataset


def get_mlp(trial: Trial, dataset: Dataset, device: Device) -> nn.Module:
    n_block = trial.suggest_int('n_block', 1, 8)
    d_block = trial.suggest_int('d_block', 1, 512)
    dropout = trial.suggest_float('dropout', 0, 0.5)

    return MLP(
        d_in=len(dataset[0][0]),  # Number of features
        d_out=len(dataset[0][1]),  # Number of labels per sample
        n_blocks=n_block,
        d_block=d_block,
        dropout=dropout).to(device)


def get_resnet(trial: Trial, dataset: Dataset, device: Device) -> nn.Module:
    n_blocks = trial.suggest_int('n_blocks', 1, 8)
    d_block = trial.suggest_int('d_block', 64, 512)
    d_hidden_multiplier = trial.suggest_float('d_hidden_multiplier', 1, 4)
    hidden_dropout = trial.suggest_float('hidden_dropout', 0, 0.5)
    residual_dropout = trial.suggest_float('residual_dropout', 0, 0.5)

    return ResNet(
        d_in=len(dataset[0][0]),
        d_out=len(dataset[0][1]),
        n_blocks=n_blocks,
        d_block=d_block,
        d_hidden=None,  # We set d_hidden as a factor of d_block (the dim of the residual block) below.
        d_hidden_multiplier=d_hidden_multiplier,
        dropout1=hidden_dropout,
        dropout2=residual_dropout).to(device)


def get_ft_transformer(trial: Trial, dataset: Dataset, device: Device):
    # We use the default number of attention heads (8), but tune everything else.
    params = FTTransformer.get_default_kwargs()
    overrides = {
        'n_blocks': trial.suggest_int('n_blocks', 1, 4),  # Number of transformer blocks
        'd_block': trial.suggest_int('d_block', 64, 512, step=8),  # Dim of the transformer block (feature dim)
        'attention_dropout': trial.suggest_float('attention_dropout', 0, 0.5),

        # The FFN dim is a multiple of the transformer block dim.
        'ffn_d_hidden_multiplier': trial.suggest_float('ffn_d_hidden_multiplier', 1, 4),
        'ffn_dropout': trial.suggest_float('ffn_dropout', 0, 0.5),

        # Dropout for all residual branches (FFN and attention):
        'residual_dropout': trial.suggest_float('residual_dropout', 0, 0.2)
    }
    params.update(overrides)

    model = FTTransformer(
        n_cont_features=len(dataset[0][0]),
        cat_cardinalities=[],  # No categorical features
        d_out=len(dataset[0][1]),
        **params).to(device)

    # Patch forward pass to accept no categoricals:
    model.forward = types.MethodType(partial(FTTransformer.forward, x_cat=None), model)

    return model


def get_ae(num_columns: int, num_labels: int, hidden_units: list[int], dropout_rates: list[float],
           lr: float = 1e-3) -> Model:
    """Denoising autoencoder.

    Based on the following notebooks:
    https://www.kaggle.com/code/aimind/bottleneck-encoder-mlp-keras-tuner-8601c5/notebook
    https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp/notebook
    """

    inp = layers.Input(shape=(num_columns,))
    x0 = layers.BatchNormalization()(inp)  # Input normalization.

    # Noising encoder
    encoder = layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = layers.Dense(hidden_units[0])(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('swish')(encoder)

    # Decoder
    decoder = layers.Dropout(dropout_rates[0])(encoder)
    decoder = layers.Dense(num_columns, name='decoder')(decoder)

    # Additional classification layer after the autoencoder
    x_ae = layers.Dense(hidden_units[1])(decoder)
    x_ae = layers.BatchNormalization()(x_ae)
    x_ae = layers.Activation('swish')(x_ae)
    x_ae = layers.Dropout(dropout_rates[2])(x_ae)
    out_ae = layers.Dense(num_labels, activation='sigmoid', name='ae_action')(x_ae)

    # We feed the encoder's latent representation and the original input into an MLP that predicts the action
    # for each resp. At test time, we'll use the mean of all predicted actions as the final probability.
    x = layers.Concatenate()([x0, encoder])
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rates[3])(x)
    for i in range(2, len(hidden_units)):
        x = layers.Dense(hidden_units[i])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        x = layers.Dropout(dropout_rates[i + 2])(x)
    out = layers.Dense(num_labels, activation='sigmoid', name='action')(x)

    model = keras.models.Model(inputs=inp, outputs=[decoder, out_ae, out])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss={'decoder': keras.losses.MeanSquaredError(),
                        'ae_action': keras.losses.BinaryCrossentropy(),
                        'action': keras.losses.BinaryCrossentropy()},
                  metrics={'decoder': keras.metrics.MeanAbsoluteError(name='MAE'),
                           'ae_action': keras.metrics.AUC(name='AUC'),
                           'action': keras.metrics.AUC(name='AUC')})

    return model
