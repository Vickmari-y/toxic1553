from deepchem.feat import MolecularFeaturizer
import numpy as np
import torch
from deepchem.feat import MolecularFeaturizer
from rdkit import Chem


class ConcatFeaturizer(MolecularFeaturizer):
    def __init__(self, featurizers):
        super().__init__()
        self.featurizers = featurizers

    def _featurize(self, mol, **kwargs):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        return np.concatenate([f._featurize(mol, **kwargs) for f in self.featurizers], axis=-1)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            self.trace_func(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

