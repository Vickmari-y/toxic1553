import mlflow
import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import r2_score
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FeedForward(LightningModule):
    def __init__(self, num_in_features, num_out_features, act_func, dims, dropouts,
                 loss_function, optimizer, scheduler_kwargs, optimizer_kwargs):
        super().__init__()
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler_kwargs = scheduler_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.act_func = act_func
        self.hidden_dims = (num_in_features,) + dims
        self.dropouts = dropouts

        self.seq = self.make_fc_blocks()
        self.out_seq = nn.Linear(self.hidden_dims[-1], num_out_features)

        self.train_step_outputs = []
        self.val_step_outputs = []
        self.train_step_true = []
        self.val_step_true = []

    def make_fc_blocks(self):
        def fc_layer(in_f, out_f, dropout):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.Dropout(dropout),
                self.act_func
            )

        lin_layers = [
            fc_layer(self.hidden_dims[i], self.hidden_dims[i + 1], self.dropouts[i])
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
        mask = ~target.isnan()
        loss = self.loss_function(output[mask], target[mask])
        self.log('train_loss', loss, batch_size=target.shape[0], prog_bar=True)
        self.train_step_outputs += [output]
        self.train_step_true += [target.view(*output.shape)]
        return loss

    def validation_step(self, batch):
        inputs, target = batch
        output = self.forward(inputs)
        mask = ~target.isnan()
        loss = self.loss_function(output[mask], target[mask])
        self.log('val_loss', loss, batch_size=target.shape[0])
        self.val_step_outputs += [output]
        self.val_step_true += [target.view(*output.shape)]
        return loss

    def on_validation_epoch_end(self):
        predictions = torch.cat(self.val_step_outputs, dim=0)
        true = torch.cat(self.val_step_true, dim=0)
        mask = ~true.isnan()

        r2 = r2_score(true[mask].detach().cpu().numpy(), predictions[mask].detach().cpu().numpy())
        loss = self.loss_function(predictions[mask], true[mask])
        mlflow.log_metrics({"loss_val": loss.item(), "r2_val": r2}, step=self.current_epoch)
        self.val_step_outputs = []
        self.val_step_true = []

    def on_train_epoch_end(self):
        predictions = torch.cat(self.train_step_outputs, dim=0)
        true = torch.cat(self.train_step_true, dim=0)
        mask = ~true.isnan()

        r2 = r2_score(true[mask].detach().cpu().numpy(), predictions[mask].detach().cpu().numpy())
        loss = self.loss_function(predictions[mask], true[mask])
        mlflow.log_metrics({
            "loss_train": loss.item(),
            "r2_train": r2,
            "lr": self.optimizers().param_groups[0]["lr"]
        }, step=self.current_epoch)
        self.train_step_outputs = []
        self.train_step_true = []

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, **self.scheduler_kwargs),
                "monitor": "val_loss",
                "frequency": 1  # should be set to "trainer.check_val_every_n_epoch"
            },
        }
