import math
import pandas as pd
import numpy as np
import logging
from numpy.typing import ArrayLike
from optuna import Trial

from time_series_split import PurgedGroupTimeSeriesSplit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging(name: str) -> logging.Logger:
    """Set up default logging configuration.

    Usage: setup_logging(__name__)."""
    logging.basicConfig(
        format='[%(levelname)1.1s %(asctime)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%d/%b/%Y %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def load_parquet(path: str, n_lines: int = math.inf):
    """Loads the dataset from the Jane Street market prediction Kaggle competition.

    https://www.kaggle.com/competitions/jane-street-market-prediction
    """
    if n_lines != math.inf:
        logger.info(f'Loading first {n_lines} lines...')
        import pyarrow as pa
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        first_n_rows = next(pf.iter_batches(batch_size=n_lines))
        return pa.Table.from_batches([first_n_rows]).to_pandas()
    else:
        logger.info(f'Loading data...')
        return pd.read_parquet(path)


def prepare_data(n_lines: int = math.inf) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = load_parquet('data/market_prediction.parquet', n_lines)
    features = [c for c in train.columns if 'feature' in c]

    logger.info(f'Preprocessing data...')

    if n_lines > 6e5:  # Only filter by date if we pulled enough data.
        # Remove first 85 days, which show likely regime change.
        train = train.query('date > 85').reset_index(drop=True)

    # Weight 0 doesn't contribute to scoring.
    train = train.query('weight > 0').reset_index(drop=True)

    # Fill in NA values with previous value (for time series continuity).
    train[features] = train[features].ffill().fillna(0)

    # Only resp is used for scoring, but resp_1...4 (the responses at further time horizons) appear useful for
    # out-of-sample prediction.
    resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']

    # We'll perform multi-label classification.
    y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T

    # Futhermore, we'll use the average resp as sample weights.
    sw = np.mean(np.abs(train[resp_cols].to_numpy()), axis=1)

    X = train[features].to_numpy()
    date = train['date'].to_numpy()  # We use the date to produce our cross val folds.

    logger.info(f'Loaded X {X.shape} and y {y.shape}.')

    return features, date, X, y, sw


def time_series_folds(n: int, groups: ArrayLike, n_splits: int = 5, group_gap: int = 31):
    """Set up our cross validation splits s.t. train always precedes test in time.

    Furthermore, add a gap between train and test to avoid information leakage from lagging features.
    """
    gkf = PurgedGroupTimeSeriesSplit(n_splits=n_splits, group_gap=group_gap)

    return enumerate(gkf.split(np.zeros(n), groups=groups))


def time_series_split(n: int, groups: ArrayLike, group_gap: int = 31, test_frac: float = 0.2):
    """A one-fold version of time_series_folds."""
    unique_groups, ind = np.unique(groups, return_index=True)
    n_groups = len(unique_groups)
    assert group_gap < n_groups

    group_test_size = (n_groups - group_gap) * test_frac
    group_test_start = int(n_groups - group_test_size)
    group_train_end = group_test_start - group_gap
    return np.arange(ind[0], ind[group_train_end]), np.arange(ind[group_test_start], n)


def weighted_average(scores: list[float]) -> float:
    """Return a weighted average of the cross validation scores.

    Earlier folds are further in the past and use less data, so we give them less weight.
    """
    weights = []
    n = len(scores)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        weights.append(1 / (2**(n + 1 - j)))
    return np.average(scores, weights=weights)


def print_trial(name: str, trial: Trial):
    print(f"{name}:")
    print(f"  Value: {trial.value}")
    print("  User attributes:")
    [print(f"    {key}: {value}") for key, value in trial.user_attrs.items()]
    print("  Params: ")
    [print(f"    {key}: {value}") for key, value in trial.params.items()]


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def should_stop(self, validation_loss: float):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
