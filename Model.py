from torch import nn


class FeedForward(nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        self.linear_layer_1 = nn.Linear(num_in_features, 256)
        self.activation_function_1 = nn.ReLU()
        self.linear_layer_2 = nn.Linear(256, 64)
        self.activation_function_2 = nn.ReLU()
        self.linear_layer_3 = nn.Linear(64, num_out_features)
        self.sigma = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_layer_1(x)
        x = self.activation_function_1(x)
        x = self.linear_layer_2(x)
        x = self.activation_function_2(x)
        x = self.linear_layer_3(x)
        x = self.sigma(x)
        return x
