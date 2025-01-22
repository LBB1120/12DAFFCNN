import torch
import torch.nn as nn

class Conv1DBranch(nn.Module):
    def __init__(self):
        super(Conv1DBranch, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, kernel_size=5)  # Input shape: (batch_size, 1, length)
        self.pool1 = nn.MaxPool1d(2, 1)
        self.conv2 = nn.Conv1d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2, 1)
        self.flatten_size = 144  # Flattened output size, adjusted based on data

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)  # Flatten the output for further processing
        return x