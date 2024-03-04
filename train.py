import mlflow
import torch
from deepchem.feat import CircularFingerprint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adam

from model import FeedForward
from utils import ConcatFeaturizer, create_dataloaders

experiment_name = "test"
max_epochs = 1000
reduce_lr_patience = 30
reduce_lr_factor = 0.2
reduce_lr_cooldown = 2
es_patience = 100
device = torch.device("cuda:0")
seed = 27
max_samples = None

learning_rate = 1e-3
batch_size = 8

filename = "data/LD50_train_multi.csv"
targets = [
    'LD50_guinea pig_oral_30',
    'LD50_guinea pig_intravenous_30',
    'LD50_guinea pig_subcutaneous_30',
    'LD50_guinea pig_skin_30',
    'LD50_rabbit_oral_30',
    'LD50_rabbit_intravenous_30',
    'LD50_rabbit_subcutaneous_30',
    'LD50_rabbit_skin_30',
    'LD50_mouse_oral_30',
    'LD50_mouse_intravenous_30',
    'LD50_mouse_subcutaneous_30',
    'LD50_mouse_skin_30',
    'LD50_rat_oral_30',
    'LD50_rat_intravenous_30',
    'LD50_rat_subcutaneous_30',
    'LD50_rat_skin_30',
]
featurizer = ConcatFeaturizer(featurizers=[
    CircularFingerprint(radius=2, size=1024),
    CircularFingerprint(radius=3, size=1024),
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
    dims=(1024, 512, 128, 64),
    act_func=nn.ReLU(),
    num_in_features=next(iter(val_dataloader))[0].shape[-1],
    num_out_features=len(targets),
    loss_function=nn.MSELoss(),
    optimizer=Adam,
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

optimizer = Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

mlflow.set_experiment(experiment_name=experiment_name)
with mlflow.start_run(run_name=f"test"):
    es_callback = EarlyStopping(patience=es_patience, monitor="val_loss")
    trainer = Trainer(accelerator="auto", callbacks=[es_callback], max_epochs=max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)

