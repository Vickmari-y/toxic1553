from torch import nn


class FeedForward(nn.Module):
    def __init__(self, num_in_features, num_out_features, act_func, dims):
        super().__init__()
        layers = []
        dims = [num_in_features] + dims
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            layers += [act_func]
        layers += [nn.Linear(dims[-1], num_out_features)]
        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        x = self.sequential(x)
        return x
