import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        RevIN layer implementation.
        Args:
            num_features: the number of features (channels) in the input time series.
            eps: small value to avoid division by zero.
            affine: whether to use learnable affine parameters.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        # If affine transformation is used, initialize learnable weights and biases
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))  # learnable scaling parameter
            self.beta = nn.Parameter(torch.zeros(num_features))  # learnable shifting parameter

    def forward(self, x, mode: str = 'norm'):
        """
        Forward pass.
        Args:
            x: input tensor with shape (batch_size, time_step, num_features)
            mode: 'norm' for normalization, 'denorm' for denormalization.
        Returns:
            processed tensor.
        """
        if mode == 'norm':
            # Normalization process
            self._get_statistics(x)  # Compute mean and std
            x = self._normalize(x)   # Normalize input
        elif mode == 'denorm':
            # Denormalization process
            x = self._denormalize(x)  # Restore output
        else:
            raise ValueError("mode must be 'norm' or 'denorm'")
        return x

    def _get_statistics(self, x):
        """
        Compute the mean and standard deviation for each feature.
        x: input tensor, shape (batch_size, time_step, num_features)
        """
        # Calculate mean and std along the time_step axis (dimension 1)
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()  # (batch_size, 1, num_features)
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()  # (batch_size, 1, num_features)

    def _normalize(self, x):
        """
        Normalize the input.
        """
        x = x - self.mean  # Subtract mean
        x = x / self.stdev  # Divide by standard deviation
        if self.affine:
            # Apply learnable affine transformation
            x = x * self.gamma.view(1, 1, self.num_features) + self.beta.view(1, 1, self.num_features)
        return x

    def _denormalize(self, x):
        """
        Denormalize the output.
        """
        if self.affine:
            # Reverse the affine transformation
            x = x - self.beta.view(1, 1, self.num_features)
            x = x / (self.gamma.view(1, 1, self.num_features) + self.eps)
        x = x * self.stdev  # Multiply by standard deviation
        x = x + self.mean   # Add mean
        return x