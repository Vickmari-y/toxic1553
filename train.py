import os

import mlflow
import numpy as np
import pandas as pd
import torch
from deepchem.feat import CircularFingerprint
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam

from model import FeedForward
from utils import ConcatFeaturizer, create_dataloaders, TableFeaturizer

experiment_name = "mouse_multitarget"
max_epochs = 1000
reduce_lr_patience = 30
reduce_lr_factor = 0.2
reduce_lr_cooldown = 2
es_patience = 100
device = torch.device("cuda:0")
seed = 27
max_samples = None

learning_rate = 1e-2
batch_size = 16


def get_preds(model, dataloader):
    with torch.no_grad():
        pred = torch.cat([
            model(x.to(device)).cpu() for x, y in dataloader
        ], dim=0)
    target = torch.cat([y.cpu() for x, y in dataloader], dim=0)
    return target, pred


def report_metrics(model, loaders: dict):
    model.eval()
    for name, loader in loaders.items():
        true, pred = get_preds(model.to(device), loader)
        for i, target in enumerate(targets):
            current_true = true[:, i]
            current_pred = pred[:, i]
            mask = ~current_true.isnan()
            mlflow.log_metric(f"RMSE_{name}_{target}", np.sqrt(MSELoss()(current_pred[mask], current_true[mask]).item()))
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


filename = "data/LD50_train_multi.csv"
targets = [
    # 'LD50_guinea pig_oral_30',
    # 'LD50_guinea pig_intravenous_30',
    # 'LD50_guinea pig_subcutaneous_30',
    # 'LD50_guinea pig_skin_30',
    # 'LD50_rabbit_oral_30',
    # 'LD50_rabbit_intravenous_30',
    # 'LD50_rabbit_subcutaneous_30',
    # 'LD50_rabbit_skin_30',
    'LD50_mouse_oral_30',
    'LD50_mouse_intravenous_30',
    'LD50_mouse_subcutaneous_30',
    'LD50_mouse_skin_30',
    # 'LD50_rat_oral_30',
    # 'LD50_rat_intravenous_30',
    # 'LD50_rat_subcutaneous_30',
    # 'LD50_rat_skin_30',
]

activation = nn.ReLU()
optimizer = Adam
dims = (2048, 1024, 512, 256, 128, 64)
dropouts = (0.5, 0.5, 0.5, 0.5, 0.25, 0.1)
featurizer = ConcatFeaturizer([
    CircularFingerprint(radius=2, size=1024),
    CircularFingerprint(radius=3, size=1024),
    # TableFeaturizer(df=pd.read_csv("data/features.csv")),
    # EstateFeaturizer(),
])

train_dataloader, val_dataloader = create_dataloaders(
    filename,
    featurizer=featurizer,
    batch_size=batch_size,
    val_size=0.1,
    targets=targets,
    max_samples=max_samples,
)

model = FeedForward(
    dims=dims,
    act_func=activation,
    dropouts=dropouts,
    num_in_features=next(iter(val_dataloader))[0].shape[-1],
    num_out_features=len(targets),
    loss_function=nn.MSELoss(),
    optimizer=optimizer,
    scheduler_kwargs={
        "mode": "min",
        "patience": reduce_lr_patience,
        "factor": reduce_lr_factor,
        "cooldown": reduce_lr_cooldown,
    },
    optimizer_kwargs={
        "lr": learning_rate,
    }
)

loss_function = nn.MSELoss()

mlflow.set_experiment(experiment_name=experiment_name)
with mlflow.start_run():
    # report_metrics(model, {"train-before-train": train_dataloader, "val-before-train": val_dataloader})
    es_callback = EarlyStopping(patience=es_patience, monitor="val_loss")
    trainer = Trainer(accelerator="auto", callbacks=[es_callback], max_epochs=max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)
    mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="model_state_dict")
    report_metrics(model, {"train": train_dataloader, "val": val_dataloader})
