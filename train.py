import numpy as np
import pandas as pd
import torch
import json
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Model import FeedForward

num_in_features = 2048
num_out_features = 1
learning_rate = 1e-4
max_epoch = 30
filename = "data/LC50_Bluegill_unknown_4.csv"


def train_model(filename: str) -> nn.Module:
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

    model = FeedForward(num_in_features, num_out_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(max_epoch), desc="Training progress", total=max_epoch):
        test_predictions = torch.tensor([model(x).item() for x, y in test_data])
        test_loss = loss_function(test_predictions, torch.tensor(y_test))
        test_losses.append(test_loss)

        epoch_train_losses = []
        for batch in train_dataloader:
            x, y_true = batch

            optimizer.zero_grad()

            y_pred = model(x)

            loss = loss_function(y_pred.view(*y_true.shape), y_true)
            epoch_train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
        train_losses.append(np.mean(epoch_train_losses))

    test_pred = []
    for batch in test_data:
        x, y_true = batch
        y_pred = model(x)
        test_pred.append(y_pred.item())
    r2 = r2_score(y_test, test_pred)

    return model, r2, train_losses, test_losses


model, r2, train_losses, test_losses = train_model(filename)

# TODO: save R2 to file (txt, json, ...)
# with open("r2_score.json", "r") as file:
#     r2s = json.load(file)
# r2s[filename] = r2
# with open("r2_score.json", "w") as file:
#     json.dump(r2s, file)



# TODO: save plot as image
plt.plot(train_losses, label="train loss")
plt.savefig('train_loss.png')
plt.plot(test_losses, label="test loss")
plt.savefig('test_loss.png')

# TODO: save model to .pth
torch.save(model.state_dict(), )
