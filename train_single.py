from deepchem.feat import CircularFingerprint
import mlflow
import torch
from deepchem.feat import CircularFingerprint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from model import FeedForward
from utils import ConcatFeaturizer, create_dataloaders, report_metrics, load_data

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

featurizer = ConcatFeaturizer(featurizers=[
    CircularFingerprint(radius=2, size=2048),
    CircularFingerprint(radius=3, size=2048),
])
activation = nn.PReLU()
optimizer = AdamW
dims = (455, 700, 359, 253, 338)
dropouts = (0.3705148571109272, 0.3866904393383811, 0.09555730388487488, 0.23477937500282223, 0.2321243257281017)

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

    X, y = load_data(filename=f"data/test_data/log_{target}.csv", targets=[target], featurizer=featurizer)
    test_dataloader = DataLoader(TensorDataset(X, y), batch_size=1)

    report_metrics(model, loaders={
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }, targets=targets, device=device)
