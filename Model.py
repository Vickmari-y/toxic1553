import rdkit
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem



class FeedForward(nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        self.linear_layer_1 = nn.Linear(num_in_features, 256)
        self.activation_function_1 = nn.ReLU()
        self.linear_layer_2 = nn.Linear(256, 64)
        self.activation_function_2 = nn.ReLU()
        self.linear_layer_3 = nn.Linear(64, num_out_features)
        self.sigma = nn.Sigmoid()


    def forward(self, x):
        x = self.linear_layer_1(x)
        x = self.activation_function_1(x)
        x = self.linear_layer_2(x)
        x = self.activation_function_2(x)
        x = self.linear_layer_3(x)
        x = self.sigma(x)
        return x
