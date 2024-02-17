import mlflow
import optuna
import pandas as pd
import random
import torch
from deepchem.feat import RDKitDescriptors, CircularFingerprint
from rdkit import Chem
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from Model import FeedForward
from utils import ConcatFeaturizer, EarlyStopping

MLFLOW_TRACKING_URI = "http://127.0.0.1:8891"
max_epoch = 100
reduce_lr_patianse = 8
reduce_lr_factor = 0.2
reduce_lr_cooldown = 2
es_patience = 10
n_trials = None
timeout = 3600 * 24
device = torch.device("cuda:0")
seed = 27

filenames = [
    "data/LD50_mouse_intravenous_30.csv",
    # "data/LD50_rat_intraperitoneal_30.csv",
    # "data/LD50_rat_intravenous_30.csv",
    # "data/LD50_rabbit_oral_30.csv",
    # "data/LD50_rabbit_skin_30.csv",
    # "data/LD50_rat_subcutaneous_30.csv",
]
featurizer_variants = {
    "ConcatFeaturizer_small": ConcatFeaturizer(featurizers=[
        RDKitDescriptors(),
        CircularFingerprint(radius=2, size=512),
        CircularFingerprint(radius=3, size=512),
    ]),
    "ConcatFeaturizer_large": ConcatFeaturizer(featurizers=[
        RDKitDescriptors(),
        CircularFingerprint(radius=2, size=2048),
        CircularFingerprint(radius=3, size=2048),
    ]),
    "CircularFingerprint_2_2048": CircularFingerprint(radius=2, size=2048),
    "CircularFingerprint_3_2048": CircularFingerprint(radius=3, size=2048),
    "RDKitDescriptors": RDKitDescriptors(),
}
act_func_variants = {
    "nn.ReLU()": nn.ReLU(),
    "nn.LeakyReLU()": nn.LeakyReLU(),
}
optimizer_variants = {
    'Adam': Adam,
    'AdamW': AdamW,
    'SGD': SGD,
    'RMSprop': RMSprop,
}
CASHED_DATA = {
    name: {
        featurizer: None
        for featurizer in featurizer_variants.keys()
    } for name in filenames
}


def estimate_params(trial):
    n_layers = trial.suggest_int("n_layers", 2, 5)
    act_func_name = trial.suggest_categorical("activation", act_func_variants.keys())
    return {
        "dims": [
            trial.suggest_int(f"dim_{i}", 32, 2048, log=True)
            for i in range(n_layers)
        ],
        "act_func": act_func_variants[act_func_name],
    }


def load_data(filename, featurizer, batch_size=16, test_size=0.1, val_size=0.1):
    try:
        X_full, targets = CASHED_DATA[filename][featurizer]
    except KeyError:
        df = pd.read_csv(filename)
        molecules = [Chem.MolFromSmiles(s) for s in df["smiles"]]
        targets = torch.tensor(df["value"].tolist(), dtype=torch.float32)

        X_full = featurizer.featurize(molecules)
        X_full = torch.from_numpy(X_full).to(torch.float32)

        CASHED_DATA[filename][featurizer] = (X_full, targets)

    x_train_val, x_test, y_train_val, y_test = train_test_split(X_full, targets, test_size=test_size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_size, random_state=seed)
    mean, std = y_train.mean(), y_train.std()

    y_train = (y_train - mean) / std
    y_val = (y_val - mean) / std
    y_test = (y_test - mean) / std

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def train_model(trial, model, optimizer, loss_function, train_dataloader, val_dataloader, test_dataloader):
    def get_loss(model, dataloader):
        with torch.no_grad():
            predictions = torch.cat([model(x.to(device)).cpu() for x, y in dataloader], dim=0)
        true = torch.cat([y for x, y in dataloader], dim=0)
        loss = loss_function(predictions, true)
        return loss

    def get_r2(model, dataloader):
        with torch.no_grad():
            predictions = torch.cat([model(x.to(device)).cpu() for x, y in dataloader], dim=0)
        true = torch.cat([y for x, y in dataloader], dim=0)
        r2 = r2_score(true.numpy(), predictions.numpy())
        return r2

    def train_one_batch(model, batch, step=0):
        x, y_true = batch
        optimizer.zero_grad()
        y_pred = model(x.to(device))
        loss = loss_function(y_pred.view(*y_true.shape), y_true.to(device))
        loss.backward()
        optimizer.step()

        val_loss = get_loss(model, val_dataloader)

        mlflow.log_metrics({"train_loss": loss.item(), "val_loss": val_loss.item()}, step=step)
        mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=step)

        trial.report(val_loss.item(), step=step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return model, loss, val_loss

    model.train()
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, patience=reduce_lr_patianse, factor=reduce_lr_factor, cooldown=reduce_lr_cooldown)
    es_scheduler = EarlyStopping(patience=es_patience, path="output/checkpoint.pt")
    for epoch in range(max_epoch):
        for i, batch in enumerate(train_dataloader):
            model, train_loss, val_loss = train_one_batch(model, batch, step=epoch * len(train_dataloader) + i)
            print(f"\rEpoch {epoch + 1}/{max_epoch}, batch {i + 1}/{len(train_dataloader)}: "
                  f"train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}", end="")

        val_loss = get_loss(model, val_dataloader)
        scheduler.step(val_loss)
        es_scheduler(val_loss, model)
        if es_scheduler.early_stop:
            break

    mlflow.log_metrics({
        "train_r2": get_r2(model, train_dataloader),
        "val_r2": get_r2(model, val_dataloader),
        "test_r2": get_r2(model, test_dataloader)
    })

    val_loss = get_loss(model, val_dataloader)
    return val_loss.item()


def objective(trial):
    featurizer_name = trial.suggest_categorical("featurizer", featurizer_variants.keys())
    filename = random.choice(filenames)
    train_dataloader, val_dataloader, test_dataloader = load_data(
        filename,
        featurizer=featurizer_variants[featurizer_name],
        batch_size=trial.suggest_int("batch_size", 16, 256, log=True),
        test_size=0.1)
    params = estimate_params(trial)
    model = FeedForward(**params, num_in_features=next(iter(test_dataloader))[0].shape[-1], num_out_features=1)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", optimizer_variants.keys())
    optimizer = optimizer_variants[optimizer_name](model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params(trial.params)
        mlflow.set_tags({"trial": trial.number, "datetime_start": trial.datetime_start})
        mlflow.log_input(mlflow.data.from_pandas(pd.read_csv(filename), source=filename), context="total_data")

        final_test_loss = train_model(trial, model, optimizer, loss_function, train_dataloader, val_dataloader,
                                      test_dataloader)

    return final_test_loss


study = optuna.create_study(
    study_name="optimize_hparams",
    storage="sqlite:///output/optimize_hparams.db",
    load_if_exists=True,
    direction="minimize",
    pruner=optuna.pruners.ThresholdPruner(n_warmup_steps=100, upper=1e7)
)

mlflow.set_experiment(experiment_name="optimize_hparams")
study.optimize(objective, n_trials=n_trials, timeout=timeout)
