import numpy as np
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
