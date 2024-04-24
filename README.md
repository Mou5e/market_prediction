# Deep learning on the Jane Street market prediction dataset
This project explores several deep learning architectures with the [Jane Street market prediction dataset](https://www.kaggle.com/competitions/jane-street-market-prediction). Specifically, we compare an autoencoder (based on [these](https://www.kaggle.com/code/aimind/bottleneck-encoder-mlp-keras-tuner-8601c5/notebook) [two](https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp/notebook) successful submissions from the competition by Yirun Zhang and aimind et al. respectively) with an MLP, ResNet, and FT-Transformer (all three based on [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) from Gorishniy et al.).

We performed hyperparameter tuning with optuna and trained on an H100.

## Entry points
* `autoencoder.py`: Train/val/test for the autoencoder.
* `tune.py`: Hyperparameter tuning for the MLP, ResNet, and FT-Transformer.
* `train_and_test.py`: Train and test for the MLP, ResNet, and FT-Transformer.

## Results
The competition uses a specific scoring function. For simplicity, we simply show the ROC AUC instead.

We suspect we need more time in hyperparameter tuning for the FT-Transformer. We have a large number of features, and the quadratic complexity of attention makes the FT-Transformer the slowest model to train by far. We can use [Linformer attention](https://arxiv.org/abs/2006.04768) to accelerate training.

| Model          | ROC AUC    |
|----------------|------------|
| Autoencoder    | 0.54038884 |
| MLP            | 0.54575911 |
| ResNet         | 0.54547325 |
| FT-Transformer | 0.54057036 |

