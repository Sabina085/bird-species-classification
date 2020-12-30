import torch
import torch.nn as nn
import torch.nn.functional as F


# Trained for 100 epochs, lr = 0.0001
class MLP1(torch.nn.Module): # Used for the first final submission on Kaggle
    def __init__(self, input_dim, output_dim):
        super(MLP1, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 128)
        self.linear2 = torch.nn.Linear(128, output_dim)


    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        outputs = self.linear2(x)
        return outputs


# Trained for 100 epochs, lr = 0.0001
class MLP2(torch.nn.Module): # Used for the second final submission on Kaggle
    def __init__(self, input_dim, output_dim):
        super(MLP2, self).__init__()
        self.bn1 = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.linear1 = torch.nn.Linear(input_dim, 128)
        self.bn2 = nn.LayerNorm(128)
        self.linear2 = torch.nn.Linear(128, output_dim)


    def forward(self, x):
        x = self.linear1(torch.tanh(self.bn1(x)))
        x = torch.tanh(self.bn2(x))
        outputs = self.linear2(x)
        return outputs

