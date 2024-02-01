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

import Model

molecules = [m for m in rdkit.Chem.SDMolSupplier("nr-ar.sdf") if m is not None]
molecules = molecules[:1000]
# listActive
# X_full

listActive = []
for i in range(len(molecules)):
    listActive += [int(molecules[i].GetProp("Active"))]

print(sum(listActive))

X_full = [list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2)) for molecule in tqdm(molecules, desc="Calculating descriptors")]

X_full = torch.tensor(X_full, dtype=torch.float32)
listActive = torch.tensor(listActive, dtype=torch.float32)

x_train, x_test, y_train, y_test = train_test_split(X_full, listActive, test_size=0.1, stratify=listActive, random_state=42)

train_data = list(zip(x_train, y_train))
train_dataloader = DataLoader(train_data, batch_size = 16)
test_data = list(zip(x_test, y_test))

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


num_in_features = 2048
num_out_features = 1

model = FeedForward(num_in_features, num_out_features)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_function = nn.BCELoss()

max_epoch = 30

loss_list = []

for epoch in tqdm(range(max_epoch), desc="Training progress", total=max_epoch):
    for batch in train_dataloader:
        (x, y_true) = batch

        optimizer.zero_grad()

        y_pred = model(x)

        loss = loss_function(y_pred.view(*y_true.shape), y_true)
        loss_list.append(loss.item())

        loss.backward()

        optimizer.step()

plt.plot(loss_list)
plt.show()

test_pred = []

for batch in test_data:
    (x, y_true) = batch

    # считаем предсказание модели на очередном батче
    y_pred = model(x)
    test_pred.append(y_pred.item())

print(r2_score(y_test, test_pred))