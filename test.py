import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn
from torch.optim import SGD
from tqdm import tqdm

from model import FeedForward
from utils import ConcatFeaturizer, load_data, TableFeaturizer, EstateFeaturizer

targets = [
    # 'LD50_guinea pig_oral_30',
    # 'LD50_guinea pig_intravenous_30',
    # 'LD50_guinea pig_subcutaneous_30',
    # 'LD50_guinea pig_skin_30',
    # 'LD50_rabbit_oral_30',
    # 'LD50_rabbit_intravenous_30',
    # 'LD50_rabbit_subcutaneous_30',
    'LD50_rabbit_skin_30',
    # 'LD50_mouse_oral_30',
    # 'LD50_mouse_intravenous_30',
    # 'LD50_mouse_subcutaneous_30',
    # 'LD50_mouse_skin_30',
    # 'LD50_rat_oral_30',
    # 'LD50_rat_intravenous_30',
    # 'LD50_rat_subcutaneous_30',
    # 'LD50_rat_skin_30',
]

activation = nn.Tanh()
optimizer = SGD
dims = (512, 256, 128, 64, 32, 32)
dropouts = (0.5, 0.5, 0.5, 0.5, 0.25, 0.1)
featurizer = ConcatFeaturizer([
    TableFeaturizer(df=pd.read_csv("data/features.csv")),
    EstateFeaturizer(),
])

metrics = {}
for i, target in enumerate(tqdm(targets)):
    filename = f"data/test_data/log_{target}.csv"
    x, y_true = load_data(filename, targets=[target], featurizer=featurizer)

    model = FeedForward(
        dims=dims,
        act_func=activation,
        dropouts=dropouts,
        num_in_features=x.shape[-1],
        num_out_features=len(targets),
        loss_function=nn.MSELoss(),
        optimizer=optimizer,
        scheduler_kwargs={},
        optimizer_kwargs={}
    )
    model.load_state_dict(torch.load("output/checkpoints/LD50_rabbit_skin_30.pt"))
    model.eval()
    with torch.no_grad():
        y_pred = model(x).cpu().numpy()
    metrics[target] = r2_score(y_true, y_pred[:, i])

print(metrics)
