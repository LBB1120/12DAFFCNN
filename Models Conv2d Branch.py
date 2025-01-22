import torch
import torch.nn as nn

class Conv2DBranch(nn.Module):
    def __init__(self):
        super(Conv2DBranch, self).__init__()
        self.conv1 = nn.Conv2d(19, 6, kernel_size=5)  # Input shape: (batch_size, 19, height, width)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten_size = 16 * 5 * 5  # Flattened output size, needs adjustment based on data

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)  # Flatten the output for further processing
        return x