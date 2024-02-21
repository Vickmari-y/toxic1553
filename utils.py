import numpy as np
import pandas as pd
import torch
from deepchem.feat import MolecularFeaturizer
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class ConcatFeaturizer(MolecularFeaturizer):
    def __init__(self, featurizers):
        super().__init__()
        self.featurizers = featurizers

    def _featurize(self, mol, **kwargs):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        return np.concatenate([f._featurize(mol, **kwargs) for f in self.featurizers], axis=-1)


def load_data(filename, featurizer, batch_size=16, test_size=0.1, val_size=0.1, max_samples=None, seed=27, normalize=False):
    df = pd.read_csv(filename, nrows=max_samples)
    molecules = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    targets = torch.tensor(df["value"].tolist(), dtype=torch.float32).view(-1, 1)

    X_full = featurizer.featurize(molecules)
    X_full = torch.from_numpy(X_full).to(torch.float32)

    x_train_val, x_test, y_train_val, y_test = train_test_split(X_full, targets, test_size=test_size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_size, random_state=seed)

    if normalize:
        mean, std = y_train.mean(), y_train.std()

        y_train = (y_train - mean) / std
        y_val = (y_val - mean) / std
        y_test = (y_test - mean) / std

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, drop_last=True)
    val_dataloader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader
