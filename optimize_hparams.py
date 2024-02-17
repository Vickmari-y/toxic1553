import numpy as np
import optuna
import pandas as pd
import torch
from deepchem.feat import RDKitDescriptors, CircularFingerprint, MolecularFeaturizer
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model import FeedForward
from utils import ConcatFeaturizer

filename = "data/LC50_Bluegill_unknown_4.csv"
max_epoch = 5
n_trials = 3
timeout = None

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


def estimate_params(trial):
    n_layers = trial.suggest_int("n_layers", 1, 8)
    act_func_name = trial.suggest_categorical("activation", act_func_variants.keys())
    return {
        "dims": [
            trial.suggest_int(f"dim_{i}", 32, 2048, log=True)
            for i in range(n_layers)
        ],
        "act_func": act_func_variants[act_func_name],
    }


def load_data(filename, featurizer, batch_size=16, test_size=0.1):
    df = pd.read_csv(filename)
    molecules = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    targets = torch.tensor(df["value"].tolist(), dtype=torch.float32)

    X_full = featurizer.featurize(molecules)
    X_full = torch.from_numpy(X_full).to(torch.float32)
    x_train, x_test, y_train, y_test = train_test_split(X_full, targets, test_size=test_size, random_state=42)

    train_data = list(zip(x_train, y_train))
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_data = list(zip(x_test, y_test))

    return train_dataloader, test_data


def train_model(model, optimizer, loss_function, train_dataloader, test_data):
    for epoch in tqdm(range(max_epoch), desc="Training progress", total=max_epoch):
        for i, batch in enumerate(train_dataloader):
            x, y_true = batch
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred.view(*y_true.shape), y_true)
            loss.backward()
            optimizer.step()

    test_predictions = torch.tensor([model(x).item() for x, y in test_data])
    test_loss = loss_function(test_predictions, torch.tensor([y for x, y in test_data]))

    return test_loss.item()


def objective(trial):
    featurizer_name = trial.suggest_categorical("featurizer", featurizer_variants.keys())
    train_dataloader, test_data = load_data(
        filename,
        featurizer=featurizer_variants[featurizer_name],
        batch_size=trial.suggest_int("batch_size", 16, 256, log=True),
        test_size=0.1)
    params = estimate_params(trial)
    model = FeedForward(**params, num_in_features=test_data[0][0].shape[-1], num_out_features=1)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", optimizer_variants.keys())
    optimizer = optimizer_variants[optimizer_name](model.parameters(), lr=learning_rate)

    loss_function = nn.MSELoss()
    final_test_loss = train_model(model, optimizer, loss_function, train_dataloader, test_data)

    return final_test_loss


study = optuna.create_study(
    study_name="optimize_hparams",
    storage="sqlite:///output/optimize_hparams.db",
    load_if_exists=True,
    direction="minimize"
)

study.optimize(objective, n_trials=n_trials, timeout=timeout)
