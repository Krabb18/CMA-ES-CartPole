import torch
from torch import nn

class Controller(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.ll1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.ll1(x)