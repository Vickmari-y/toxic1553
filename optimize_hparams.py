import optuna
import torch
from torch import nn

from model import FeedForward

train_filename = "data/LC50_Bluegill_unknown_4.csv"

def estimate_params(trial):
    return {
        "model_dim_1": trial.suggest_int("model_dim_1", 32, 1024),
        "model_dim_2": trial.suggest_int("model_dim_2", 32, 1024),
        "model_dim_3": trial.suggest_int("model_dim_3", 32, 1024),
        "n_layers": trial.suggest_int("n_layers", 1, 5),
    }


def train_model(model):
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


def objective(trial):
    params = estimate_params(trial)
    model = FeedForward(**params)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    model_loss = train_model(model, optimizer, loss_function)
    return model_loss


study = optuna.create_study(
    study_name="optimize_hparams",
    storage="sqlite:///output/optimize_hparams.db",
    direction="minimize"
)

study.optimize(objective, n_trials=10)
