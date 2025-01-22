import torch
import torch.nn as nn
from models.conv1d_branch import Conv1DBranch
from models.conv2d_branch import Conv2DBranch

class DualBranchModelAFE(nn.Module):
    def __init__(self):
        super(DualBranchModelAFE, self).__init__()
        self.conv1d_branch = Conv1DBranch()
        self.conv2d_branch = Conv2DBranch()

        # Calculate the total flattened output size of both branches
        total_flatten_size = self.conv1d_branch.flatten_size + self.conv2d_branch.flatten_size

        # Define linear transformation layers in the AFE module
        self.spatial_fc = nn.Linear(total_flatten_size, total_flatten_size)  # For spatial features
        self.spectral_fc = nn.Linear(total_flatten_size, total_flatten_size)  # For spectral features

        # LayerNorm layers
        self.spatial_norm = nn.LayerNorm(total_flatten_size)
        self.spectral_norm = nn.LayerNorm(total_flatten_size)

        # GELU activation function
        self.gelu = nn.GELU()

        # Attention/Softmax for balancing feature importance
        self.attention_fc = nn.Linear(total_flatten_size, 1)  # Learned weighting mechanism
        self.softmax = nn.Softmax(dim=1)

        # Final fusion linear layers
        self.fc_fusion1 = nn.Linear(total_flatten_size, 256)
        self.fc_fusion2 = nn.Linear(256, 128)
        self.fc_fusion3 = nn.Linear(128, 64)
        self.fc_output = nn.Linear(64, 1)  # Output for binary classification

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x1, x2):
        x1 = self.conv1d_branch(x1)  # 1D branch features
        x2 = self.conv2d_branch(x2)  # 2D branch features

        # Concatenate features from both branches
        combined_features = torch.cat((x1, x2), dim=1)

        # AFE module: Process spatial and spectral features
        spatial_features = self.spatial_fc(combined_features)
        spatial_features = self.spatial_norm(spatial_features)
        spatial_features = self.gelu(spatial_features)

        spectral_features = self.spectral_fc(combined_features)
        spectral_features = self.spectral_norm(spectral_features)
        spectral_features = self.gelu(spectral_features)

        # Attention mechanism: Weight spatial and spectral features
        spatial_attention = self.attention_fc(spatial_features)
        spectral_attention = self.attention_fc(spectral_features)
        attention_weights = self.softmax(torch.cat([spatial_attention, spectral_attention], dim=1))

        # Re-weight features based on attention weights
        aggregated_features = attention_weights[:, 0].unsqueeze(1) * spatial_features + \
                              attention_weights[:, 1].unsqueeze(1) * spectral_features

        # Layer-wise feature fusion
        x = torch.relu(self.fc_fusion1(aggregated_features))
        x = self.dropout1(x)  # Add dropout to prevent overfitting
        x = torch.relu(self.fc_fusion2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc_fusion3(x))

        # Output layer
        x = torch.sigmoid(self.fc_output(x))
        return x