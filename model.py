import torch
import torch.nn as nn

class TemporalLocalizationModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_classes):
        """
        Args:
            feature_dim (int): Dimensionality of the input feature vectors (e.g., 768).
            hidden_dim (int): Number of channels for the hidden convolution layer.
            num_classes (int): Number of output classes (including background).
        """
        super(TemporalLocalizationModel, self).__init__()
        # 1D Convolution: processes the temporal dimension.
        # Input shape: [batch_size, window_size, feature_dim]
        # We need to transpose to [batch_size, feature_dim, window_size] for Conv1d.
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Adaptive pooling reduces the temporal dimension to 1.
        self.pool = nn.AdaptiveMaxPool1d(1)
        # Fully connected layer to produce class scores.
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, window_size, feature_dim]
        Returns:
            out: Tensor of shape [batch_size, num_classes]
        """
        # Transpose to shape [batch_size, feature_dim, window_size]
        x = x.transpose(1, 2)
        # Apply 1D convolution followed by ReLU.
        x = self.conv1(x)
        x = self.relu(x)
        # Apply adaptive pooling along the temporal dimension, reducing it to 1.
        x = self.pool(x)  # shape: [batch_size, hidden_dim, 1]
        # Remove the last dimension.
        x = x.squeeze(-1)  # shape: [batch_size, hidden_dim]
        # Fully connected layer to produce final scores.
        out = self.fc(x)   # shape: [batch_size, num_classes]
        return out

