import time

import keras
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score

from models import get_ae
from utils import prepare_data, time_series_folds, weighted_average, time_series_split

CHECKPOINT_DIR = 'checkpoints'
features, date, X, y, sw = prepare_data()

n_epochs = 100
batch_size = 4096
num_columns = 130
num_labels = 5
hidden_units = [96, 96, 896, 448, 448, 256]
dropout_rates = [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882,
                 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448]
lr = 1e-3

model = get_ae(num_columns, num_labels, hidden_units, dropout_rates, lr)

#%% Cross validation loop (used for tuning)
scores = []
for fold, (tr, te) in time_series_folds(len(y), date):
    start_time = time.time()
    ckp_path = f'{CHECKPOINT_DIR}/ae_{fold}.weights.h5'
    ckp = ModelCheckpoint(ckp_path, monitor='val_action_AUC', verbose=0,
                          save_best_only=True, save_weights_only=True, mode='max')
    es = EarlyStopping(monitor='val_action_AUC', min_delta= 1e-4, patience=10, mode='max',
                       baseline=None, restore_best_weights=True, verbose=0)
    history = model.fit(X[tr], [X[tr], y[tr], y[tr]], validation_data=(X[te], [X[te], y[te], y[te]]),
                        sample_weight=sw[tr],
                        epochs=100, batch_size=batch_size, callbacks=[ckp, es], verbose=0)
    hist = pd.DataFrame(history.history)
    score = hist['val_action_AUC'].max()
    print(f' Fold {fold} ROC AUC:\t{score}\t[{time.time() - start_time:.0f}s elapsed]')
    scores.append(score)

    keras.utils.clear_session()

print('Weighted average cross val score:', weighted_average(scores))

#%% Train and test
tr, te = time_series_split(len(y), date)
model.fit(X[tr], [X[tr], y[tr], y[tr]], sample_weight=sw[tr], epochs=n_epochs, batch_size=batch_size, verbose=0)
model.save_weights(f'{CHECKPOINT_DIR}/AE.weights.h5')
y_hat = model.predict(X[te], batch_size=batch_size, verbose=0)[2]

auc = roc_auc_score(y[te], y_hat)
print(f"Test metrics for AE:")
print(f"  AUC: {auc}")