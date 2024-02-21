import mlflow
import pandas as pd
import torch
from deepchem.feat import CircularFingerprint
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from rdkit import Chem
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from utils import ConcatFeaturizer

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


class FeedForward(LightningModule):
    def __init__(self, num_in_features, num_out_features, act_func, dims):
        super().__init__()
        self.act_func = act_func
        self.hidden_dims = (num_in_features,) + dims

        self.seq = self.make_fc_blocks()
        self.out_seq = nn.Linear(self.hidden_dims[-1], num_out_features)

        self.train_step_outputs = []
        self.val_step_outputs = []
        self.train_step_true = []
        self.val_step_true = []

    def make_fc_blocks(self):
        def fc_layer(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                self.act_func
            )

        lin_layers = [
            fc_layer(self.hidden_dims[i], self.hidden_dims[i + 1])
            for i in range(len(self.hidden_dims) - 1)
        ]
        return nn.Sequential(*lin_layers)

    def forward(self, x):
        hidden = self.seq(x)
        output = self.out_seq(hidden)
        return output

    def training_step(self, batch):
        inputs, target = batch
        output = self.forward(inputs)
        loss = loss_function(output, target.view(*output.shape))
        self.log('train_loss', loss, batch_size=target.shape[0], prog_bar=True)
        mlflow.log_metric("train_loss", loss, step=self.global_step)
        self.train_step_outputs += [output]
        self.train_step_true += [target.view(*output.shape)]
        return loss

    def validation_step(self, batch):
        inputs, target = batch
        output = self.forward(inputs)
        loss = loss_function(output, target.view(*output.shape))
        self.log('val_loss', loss, batch_size=target.shape[0])
        # mlflow.log_metric("val_loss", loss, step=self.global_step)
        self.val_step_outputs += [output]
        self.val_step_true += [target.view(*output.shape)]
        return loss

    def on_validation_epoch_end(self):
        predictions = torch.cat(self.val_step_outputs, dim=0)
        true = torch.cat(self.val_step_true, dim=0)

        r2 = r2_score(true.detach().cpu().numpy(), predictions.detach().cpu().numpy())
        loss = loss_function(predictions, true)
        mlflow.log_metrics({"val_loss": loss.item(), "val_r2": r2}, step=self.current_epoch)
        self.val_step_outputs = []
        self.val_step_true = []

    def on_train_epoch_end(self):
        predictions = torch.cat(self.train_step_outputs, dim=0)
        true = torch.cat(self.train_step_true, dim=0)

        r2 = r2_score(true.detach().cpu().numpy(), predictions.detach().cpu().numpy())
        loss = loss_function(predictions, true)
        mlflow.log_metrics({
            "train_loss": loss.item(),
            "train_r2": r2,
            "lr": self.optimizers().param_groups[0]["lr"]
        }, step=self.current_epoch)
        self.train_step_outputs = []
        self.train_step_true = []

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=reduce_lr_factor, patience=reduce_lr_patience,
                                               cooldown=reduce_lr_cooldown, verbose=True),
                "monitor": "val_loss",
                "frequency": 1  # should be set to "trainer.check_val_every_n_epoch"
            },
        }


def load_data(filename, featurizer, batch_size=16, test_size=0.1, val_size=0.1):
    df = pd.read_csv(filename, nrows=max_samples)
    molecules = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    targets = torch.tensor(df["value"].tolist(), dtype=torch.float32).view(-1, 1)

    X_full = featurizer.featurize(molecules)
    X_full = torch.from_numpy(X_full).to(torch.float32)

    x_train_val, x_test, y_train_val, y_test = train_test_split(X_full, targets, test_size=test_size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_size, random_state=seed)
    # mean, std = y_train.mean(), y_train.std()
    #
    # y_train = (y_train - mean) / std
    # y_val = (y_val - mean) / std
    # y_test = (y_test - mean) / std

    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


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
