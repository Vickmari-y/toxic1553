import mlflow
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

with mlflow.start_run():
    filename = "data/LC50_Bluegill_unknown_4.csv"
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

    model = nn.Sequential(
        nn.Linear(2048, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.MSELoss()
    max_epochs = 1000

    mlflow.log_params({
        "learning_rate": 1e-4,
        "max_epochs": max_epochs,
        "num_in_features": 2048,
    })

    for epoch in range(max_epochs):
        for batch in train_dataloader:
            x, y_true = batch

            optimizer.zero_grad()

            y_pred = model(x)

            loss = loss_function(y_pred.view(*y_true.shape), y_true)

            loss.backward()
            optimizer.step()
        train_loss = loss_function(model(torch.tensor(x_train)).clone().detach(), y_train)
        test_loss = loss_function(model(torch.tensor(x_test)).clone().detach(), y_test)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        print("\rEpoch {}: train loss = {:.3f}, test loss = {:.3f}".format(epoch, train_loss, test_loss), end="")

    test_r2 = r2_score(y_test, model(torch.tensor(x_test)).clone().detach())
    print("r2 =", test_r2)
