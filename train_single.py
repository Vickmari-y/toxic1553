import mlflow
import torch
from deepchem.feat import CircularFingerprint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adam

from model import FeedForward
from utils import ConcatFeaturizer, create_dataloaders, report_metrics

experiment_name = "mouse-single-target"
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

filename = "data/LD50_train_multi.csv"

# 'LD50_mouse_intravenous_30',
# 'LD50_mouse_subcutaneous_30',
# 'LD50_mouse_skin_30',
# 'LD50_mouse_oral_30'
target = 'LD50_mouse_intravenous_30'
targets = [target]

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
with mlflow.start_run(run_name=target):
    es_callback = EarlyStopping(patience=es_patience, monitor="val_loss")
    trainer = Trainer(accelerator="auto", callbacks=[es_callback], max_epochs=max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)
    mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path="model_state_dict")
    report_metrics(model, {"train": train_dataloader, "val": val_dataloader}, targets=targets, device=device)
