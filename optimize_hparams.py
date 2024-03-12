import mlflow
import optuna
import pandas as pd
import torch
from deepchem.feat import CircularFingerprint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adam, AdamW, SGD, RMSprop

from model import FeedForward
from utils import ConcatFeaturizer, create_dataloaders, EstateFeaturizer, TableFeaturizer

experiment_name = "optimize_hparams_mouse"

max_epochs = 300
es_patience = 80
reduce_lr_patience = 30
reduce_lr_factor = 0.2
reduce_lr_cooldown = 2

seed = 27
n_trials = 300
timeout = None
device = torch.device("cuda:0")
max_samples = None
batch_size = 16

filename = "data/LD50_train_multi.csv"
targets = [
    'LD50_mouse_oral_30',
    'LD50_mouse_intravenous_30',
    'LD50_mouse_subcutaneous_30',
    'LD50_mouse_skin_30',
]

featurizer_variants = {
    "Circular_2_3_2048": ConcatFeaturizer(featurizers=[
        CircularFingerprint(radius=2, size=2048),
        CircularFingerprint(radius=3, size=2048),
    ]),
    "Estate_logP_logS": ConcatFeaturizer(featurizers=[
        TableFeaturizer(df=pd.read_csv("data/features.csv")),
        EstateFeaturizer(), ])
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
    n_layers = trial.suggest_int("n_layers", 4, 7)
    act_func_name = trial.suggest_categorical("activation", act_func_variants.keys())
    optimizer_name = trial.suggest_categorical("optimizer", optimizer_variants.keys())
    return {
        "dims": tuple(
            trial.suggest_int(f"dim_{i}", 64, 2048, log=True)
            for i in range(n_layers)
        ),
        "dropouts": tuple(
            trial.suggest_float(f"dropout_{i}", 0.0, 0.5) for i in range(n_layers)
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
    train_dataloader, val_dataloader = create_dataloaders(
        filename,
        featurizer=featurizer_variants[featurizer_name],
        batch_size=batch_size,
        val_size=0.1,
        targets=targets,
        max_samples=max_samples,
    )
    # test_datasets = {
    #     target: load_data(
    #         f"data/test_data/log_{target}.csv",
    #         targets=[target],
    #         featurizer=featurizer_variants[featurizer_name],
    #     ) for target in targets
    # }

    model = FeedForward(
        **estimate_params(trial),
        num_in_features=next(iter(train_dataloader))[0].shape[-1],
        num_out_features=len(targets),
        loss_function=nn.MSELoss(),
    )

    es_callback = EarlyStopping(patience=es_patience, monitor="val_loss")
    trainer = Trainer(accelerator="auto", callbacks=[es_callback], max_epochs=max_epochs)
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params(trial.params)
        # mlflow.log_input(mlflow.data.from_pandas(pd.read_csv(filename), source=filename), context="total_data")
        trainer.fit(model, train_dataloader, val_dataloader)
        # mlflow.log_metric("val_loss", trainer.callback_metrics["val_loss"].item())

        # for target in targets:
        #     x_test, y_test = test_datasets[target]
        #     model.eva()
        #     with model.no_grad():
        #         predictions = model(x_test)
        #     mlflow.log_metric(f"r2_test_{target}", r2_score(y_test, predictions))

    return trainer.callback_metrics["val_loss"].item()


study = optuna.create_study(
    study_name=experiment_name,
    storage=f"sqlite:///output/{experiment_name}.db",
    load_if_exists=True,
    direction="minimize",
)

mlflow.set_experiment(experiment_name=experiment_name)
study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(ValueError,))
