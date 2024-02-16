import optuna
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model import FeedForward

filename = "data/LC50_Bluegill_unknown_4.csv"
max_epoch = 30


def estimate_params(trial):
    return {
        "dims": [trial.suggest_int("dim_1", 32, 1024, log=True),
                 trial.suggest_int("dim_2", 32, 1024, log=True),
                 trial.suggest_int("dim_3", 32, 1024, log=True)],

        "act_func": eval(trial.suggest_categorical("activation", ["nn.ReLU()", "nn.LeakyReLU()"]))
    }


def train_model(model, optimizer, loss_function, train_dataloader, test_data):
    for epoch in tqdm(range(max_epoch), desc="Training progress", total=max_epoch):

        for batch in train_dataloader:
            x, y_true = batch

            optimizer.zero_grad()

            y_pred = model(x)

            loss = loss_function(y_pred.view(*y_true.shape), y_true)

            loss.backward()
            optimizer.step()

    test_predictions = torch.tensor([model(x).item() for x, y in test_data])
    test_loss = loss_function(test_predictions, torch.tensor([y for x, y in test_data]))

    return test_loss


def objective(trial):
    df = pd.read_csv(filename)
    molecules = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    targets = torch.tensor(df["value"].tolist(), dtype=torch.float32)

    X_full = [list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2)) for molecule in
              tqdm(molecules, desc="Calculating descriptors")]

    X_full = torch.tensor(X_full, dtype=torch.float32)

    x_train, x_test, y_train, y_test = train_test_split(X_full, targets, test_size=0.1, random_state=42)

    train_data = list(zip(x_train, y_train))
    train_dataloader = DataLoader(train_data, batch_size=16)
    test_data = list(zip(x_test, y_test))

    params = estimate_params(trial)
    model = FeedForward(**params, num_in_features=2048, num_out_features=1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    model_loss = train_model(model, optimizer, loss_function, train_dataloader, test_data)
    return model_loss


study = optuna.create_study(
    study_name="optimize_hparams",
    # storage="sqlite:///output/optimize_hparams.db",
    direction="minimize"
)

study.optimize(objective, n_trials=10)