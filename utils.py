import os

import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset


class ConcatFeaturizer:
    def __init__(self, featurizers):
        super().__init__()
        self.featurizers = featurizers

    def featurize(self, mol, **kwargs):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        return np.concatenate([f.featurize(mol, **kwargs) for f in self.featurizers], axis=-1).reshape(1, -1)


class TableFeaturizer:
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.features = dict(zip(
            df["smiles"].tolist(),
            [row.tolist() for i, row in df[[col for col in df.columns if col != "smiles"]].iterrows()],
        ))

    def featurize(self, mol, **kwargs):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        return np.array(self.features[Chem.MolToSmiles(mol)]).reshape(1, -1)


class EstateFeaturizer:

    def featurize(self, mol, **kwargs):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        return FingerprintMol(mol)[1].reshape(1, -1)


def load_data(filename, targets, featurizer, max_samples=None):
    df = pd.read_csv(filename, nrows=max_samples)
    df = df.dropna(subset=targets, how="all")
    molecules = [Chem.MolFromSmiles(s) for s in df["smiles"]]

    X_full = []
    all_targets = []
    for mol, target in zip(molecules, df[targets].to_numpy().tolist()):
        try:
            X_full += [torch.from_numpy(featurizer.featurize(mol))]
            all_targets += [target]
        except (ValueError, KeyError) as e:
            print(f"Error featurizing molecule {Chem.MolToSmiles(mol)}: {e}")
    X_full = torch.cat(X_full, dim=0).to(torch.float32)
    all_targets = torch.tensor(all_targets, dtype=torch.float32)

    return X_full, all_targets


def create_dataloaders(filename, featurizer, batch_size=16, test_size=None, val_size=0.1, max_samples=None, seed=27,
                       targets=("values",)):
    X_full, y_full = load_data(filename, targets, featurizer, max_samples=max_samples)

    x_train, x_val, y_train, y_val = train_test_split(X_full, y_full, test_size=val_size, random_state=seed)
    if test_size is not None:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=seed)

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, drop_last=True)
    if test_size is not None:
        test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, drop_last=True)
        return train_dataloader, val_dataloader, test_dataloader
    else:
        return train_dataloader, val_dataloader


def get_preds(model, dataloader, device="cpu"):
    with torch.no_grad():
        pred = torch.cat([
            model(x.to(device)).cpu() for x, y in dataloader
        ], dim=0)
    target = torch.cat([y.cpu() for x, y in dataloader], dim=0)
    return target, pred


def report_metrics(model, loaders: dict, targets: list, device="cpu"):
    model.eval()
    for name, loader in loaders.items():
        true, pred = get_preds(model.to(device), loader, device=device)
        for i, target in enumerate(targets):
            current_true = true[:, i]
            current_pred = pred[:, i]
            mask = ~current_true.isnan()
            mlflow.log_metric(f"RMSE_{name}_{target}",
                              np.sqrt(MSELoss()(current_pred[mask], current_true[mask]).item()))
            mlflow.log_metric(f"MAE_{name}_{target}", mean_absolute_error(current_true[mask], current_pred[mask]))
            mlflow.log_metric(f"R2_{name}_{target}", r2_score(current_true[mask], current_pred[mask]))

            fig, ax = plt.subplots()
            ax.plot(current_true[mask], current_pred[mask], ".")
            ax.set_xlabel(f"True {target}, -log10(mol/kg)")
            ax.set_ylabel(f"Predicted value")
            fig.savefig(f"output/{name}.png", format='png')
            mlflow.log_artifact(f"output/{name}.png", artifact_path=f"predicted_vs_true/{target}")
            os.remove(f"output/{name}.png")
    model.train()
