import torch.nn as nn

class DiscreteNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DiscreteNet, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


