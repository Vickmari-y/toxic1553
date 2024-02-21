import mlflow
import pandas as pd
import torch
from deepchem.feat import RDKitDescriptors, CircularFingerprint
from rdkit import Chem
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import PReLU
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from Model import FeedForward
from utils import ConcatFeaturizer, EarlyStopping

max_epoch = 100
reduce_lr_patianse = 8
reduce_lr_factor = 0.2
reduce_lr_cooldown = 2
es_patience = 10
device = torch.device("cuda:0")
seed = 27


def load_data(filename, featurizer, batch_size=16, test_size=0.1, val_size=0.1):
    df = pd.read_csv(filename)
    molecules = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    targets = torch.tensor(df["value"].tolist(), dtype=torch.float32)

    X_full = featurizer.featurize(molecules)
    X_full = torch.from_numpy(X_full).to(torch.float32)

    x_train_val, x_test, y_train_val, y_test = train_test_split(X_full, targets, test_size=test_size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_size, random_state=seed)
    mean, std = y_train.mean(), y_train.std()

    y_train = (y_train - mean) / std
    y_val = (y_val - mean) / std
    y_test = (y_test - mean) / std

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def train_model(model, optimizer, loss_function, train_dataloader, val_dataloader, test_dataloader):
    def get_loss(model, dataloader):
        with torch.no_grad():
            predictions = torch.cat([model(x.to(device)).cpu() for x, y in dataloader], dim=0)
        true = torch.cat([y for x, y in dataloader], dim=0)
        loss = loss_function(predictions, true)
        return loss

    def get_r2(model, dataloader):
        with torch.no_grad():
            predictions = torch.cat([model(x.to(device)).cpu() for x, y in dataloader], dim=0)
        true = torch.cat([y for x, y in dataloader], dim=0)
        r2 = r2_score(true.numpy(), predictions.numpy())
        return r2

    def train_one_batch(model, batch, step=0):
        x, y_true = batch
        optimizer.zero_grad()
        y_pred = model(x.to(device))
        loss = loss_function(y_pred.view(*y_true.shape), y_true.to(device))
        loss.backward()
        optimizer.step()

        val_loss = get_loss(model, val_dataloader)

        mlflow.log_metrics({"train_loss": loss.item(), "val_loss": val_loss.item()}, step=step)
        mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=step)

        return model, loss, val_loss

    model.train()
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, patience=reduce_lr_patianse, factor=reduce_lr_factor,
                                  cooldown=reduce_lr_cooldown)
    es_scheduler = EarlyStopping(patience=es_patience)
    for epoch in range(max_epoch):
        for i, batch in enumerate(train_dataloader):
            model, train_loss, val_loss = train_one_batch(model, batch, step=epoch * len(train_dataloader) + i)
            print(f"\rEpoch {epoch + 1}/{max_epoch}, batch {i + 1}/{len(train_dataloader)}: "
                  f"train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}", end="")

        val_loss = get_loss(model, val_dataloader)
        scheduler.step(val_loss)
        es_scheduler(val_loss)
        if es_scheduler.early_stop:
            break

    mlflow.log_metrics({
        "train_r2": get_r2(model, train_dataloader),
        "val_r2": get_r2(model, val_dataloader),
        "test_r2": get_r2(model, test_dataloader)
    })

    val_loss = get_loss(model, val_dataloader)
    return val_loss.item()


train_dataloader, val_dataloader, test_dataloader = load_data(
    "data/LD50_mouse_intravenous_30.csv",
    featurizer=ConcatFeaturizer(featurizers=[
        RDKitDescriptors(),
        CircularFingerprint(radius=2, size=512),
        CircularFingerprint(radius=3, size=512),
    ]),
    batch_size=32,
    test_size=0.1
)
model = FeedForward(
    dims=(96, 96, 288, 416, 480, 224, 480),
    act_func=PReLU(num_parameters=1),
    num_in_features=next(iter(test_dataloader))[0].shape[-1],
    num_out_features=1
)

optimizer = Adam(model.parameters(), lr=1e-2)
loss_function = nn.MSELoss()

with mlflow.start_run(run_name=f"digital-pharma-params"):
    train_model(model, optimizer, loss_function, train_dataloader, val_dataloader,
                test_dataloader)
