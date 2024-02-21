import mlflow
import optuna
import pandas as pd
import torch
from deepchem.feat import CircularFingerprint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adam, AdamW, SGD, RMSprop

from Model import FeedForward
from utils import ConcatFeaturizer, load_data

# MLFLOW_TRACKING_URI = "http://127.0.0.1:8891"
experiment_name = "optimize_hparams_lightning_2"

max_epochs = 200
es_patience = 20
reduce_lr_patience = 10
reduce_lr_factor = 0.2
reduce_lr_cooldown = 2

seed = 27
n_trials = None
timeout = 3600 * 24 * 5
device = torch.device("cuda:0")
max_samples = None

filename = "data/log_data/log_LD50_rat_intravenous_30.csv"

featurizer_variants = {
    "ConcatFeaturizer_small": ConcatFeaturizer(featurizers=[
        CircularFingerprint(radius=2, size=512),
        CircularFingerprint(radius=3, size=512),
    ]),
    "ConcatFeaturizer_large": ConcatFeaturizer(featurizers=[
        CircularFingerprint(radius=2, size=2048),
        CircularFingerprint(radius=3, size=2048),
    ]),
    "CircularFingerprint_2_2048": CircularFingerprint(radius=2, size=2048),
    "CircularFingerprint_3_2048": CircularFingerprint(radius=3, size=2048),
}
act_func_variants = {
    "nn.LeakyReLU()": nn.LeakyReLU(),
    "nn.PReLU()": nn.PReLU(),
    "nn.Tanhshrink()": nn.Tanhshrink(),
    "nn.ReLU()": nn.ReLU(),
    "nn.ELU()": nn.ELU(),
    "nn.Softplus()": nn.Softplus(),
    "nn.Tanh()": nn.Tanh(),
}
optimizer_variants = {
    'Adam': Adam,
    'AdamW': AdamW,
    'SGD': SGD,
    'RMSprop': RMSprop,
}


def estimate_params(trial):
    n_layers = trial.suggest_int("n_layers", 2, 7)
    act_func_name = trial.suggest_categorical("activation", act_func_variants.keys())
    optimizer_name = trial.suggest_categorical("optimizer", optimizer_variants.keys())
    return {
        "dims": tuple(
            trial.suggest_int(f"dim_{i}", 32, 2048, log=True)
            for i in range(n_layers)
        ),
        "act_func": act_func_variants[act_func_name],
        "scheduler_kwargs": {
            "factor": reduce_lr_factor,
            "patience": reduce_lr_patience,
            "cooldown": reduce_lr_cooldown,
        },
        "optimizer": optimizer_variants[optimizer_name],
        "optimizer_kwargs": {"lr": 1e-2}
    }


def objective(trial):
    featurizer_name = trial.suggest_categorical("featurizer", featurizer_variants.keys())
    train_dataloader, val_dataloader, test_dataloader = load_data(
        filename,
        featurizer=featurizer_variants[featurizer_name],
        batch_size=trial.suggest_int("batch_size", 8, 128, log=True),
        test_size=0.1,
        val_size=0.1,
        seed=seed,
        max_samples=max_samples,
    )

    model = FeedForward(
        **estimate_params(trial),
        num_in_features=next(iter(test_dataloader))[0].shape[-1],
        num_out_features=1,
        loss_function=nn.MSELoss(),
    )

    es_callback = EarlyStopping(patience=es_patience, monitor="val_loss")
    trainer = Trainer(accelerator="auto", callbacks=[es_callback], max_epochs=max_epochs)
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params(trial.params)
        mlflow.log_input(mlflow.data.from_pandas(pd.read_csv(filename), source=filename), context="total_data")
        trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics["val_loss"].item()


study = optuna.create_study(
    study_name=experiment_name,
    storage=f"sqlite:///output/{experiment_name}.db",
    load_if_exists=True,
    direction="minimize",
    pruner=optuna.pruners.ThresholdPruner(n_warmup_steps=100, upper=1e3)
)

mlflow.set_experiment(experiment_name=experiment_name)
study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(ValueError,))
