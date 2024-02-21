import mlflow
import torch
from deepchem.feat import CircularFingerprint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adam

from model import FeedForward
from utils import ConcatFeaturizer, load_data

MLFLOW_TRACKING_URI = "http://127.0.0.1:8891"
experiment_name = "torch_lightning"
max_epochs = 100
reduce_lr_patience = 8
reduce_lr_factor = 0.2
reduce_lr_cooldown = 2
es_patience = 20
device = torch.device("cuda:0")
seed = 27
max_samples = None

learning_rate = 1e-3
batch_size = 8

filename = "data/log_data/log_LD50_rat_intravenous_30.csv"
featurizer = ConcatFeaturizer(featurizers=[
    CircularFingerprint(radius=2, size=1024),
    CircularFingerprint(radius=3, size=1024),
])

train_dataloader, val_dataloader, test_dataloader = load_data(
    filename,
    featurizer=featurizer,
    batch_size=batch_size,
    test_size=0.1)

model = FeedForward(
    dims=(1024, 512, 128, 64),
    act_func=nn.ReLU(),
    num_in_features=next(iter(test_dataloader))[0].shape[-1],
    num_out_features=1
)

optimizer = Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

mlflow.set_experiment(experiment_name=experiment_name)
with mlflow.start_run(run_name=f"test"):
    es_callback = EarlyStopping(patience=es_patience, monitor="val_loss")
    trainer = Trainer(accelerator="auto", callbacks=[es_callback], max_epochs=max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)
