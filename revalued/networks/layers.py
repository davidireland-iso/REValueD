"""Custom neural network layers."""
import math
import torch
import torch.nn as nn


class MLPResidualLayer(nn.Module):
    """Residual block with two linear layers and ReLU activations."""

    def __init__(self, in_dim: int, out_dim: int | None = None):
        """Initialise residual layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension. Defaults to in_dim (square case).
        """
        super().__init__()
        out_dim = out_dim or in_dim

        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.projection = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection
        """
        residual = x if self.projection is None else self.projection(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return residual + x


class VectorisedLinear(nn.Module):
    """Vectorised linear layer for ensemble networks.

    Applies different linear transformations for each ensemble member
    in a single forward pass.
    """

    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        """Initialise vectorised linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            ensemble_size: Number of ensemble members
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise parameters using uniform distribution."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vectorised linear transformation.

        Args:
            x: Input tensor of shape (ensemble_size, batch_size, in_features)

        Returns:
            Output tensor of shape (ensemble_size, batch_size, out_features)
        """
        return x @ self.weight + self.bias


class VectorisedLinearHead(nn.Module):
    """Vectorised linear layer for multi-headed ensemble networks.

    Applies different linear transformations for each ensemble member
    and each head in a single forward pass.
    """

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, num_heads: int):
        """Initialise vectorised linear head.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            ensemble_size: Number of ensemble members
            num_heads: Number of output heads
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.num_heads = num_heads

        self.weight = nn.Parameter(torch.empty(ensemble_size, num_heads, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, num_heads, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise parameters using uniform distribution."""
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vectorised linear transformation.

        Args:
            x: Input tensor of shape (ensemble_size, num_heads, batch_size, in_features)

        Returns:
            Output tensor of shape (ensemble_size, num_heads, batch_size, out_features)
        """
        return x @ self.weight + self.bias


class VectorisedMLPResidualLayer(nn.Module):
    """Vectorised residual block for ensemble networks."""

    def __init__(self, dim: int, ensemble_size: int):
        """Initialise vectorised residual layer.

        Args:
            dim: Input/output dimension
            ensemble_size: Number of ensemble members
        """
        super().__init__()
        self.fc1 = VectorisedLinear(dim, dim, ensemble_size)
        self.fc2 = VectorisedLinear(dim, dim, ensemble_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection
        """
        residual = x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return residual + x


class ActionEmbedding(nn.Module):
    """Action embedding layer with LayerNorm and MLP residual block.

    Converts integer action indices into dense embeddings, processed
    with pre-norm style LayerNorm and an MLP residual block.
    """

    def __init__(self, num_actions: int, num_heads: int, embedding_dim: int):
        """Initialise action embedding layer.

        Args:
            num_actions: Number of discrete actions (vocabulary size)
            num_heads: Number of heads
            embedding_dim: Dimension of action embeddings
        """
        super().__init__()
        self.embedding = nn.Embedding(num_actions * num_heads, embedding_dim)
        self.pre_norm = nn.LayerNorm(embedding_dim)
        self.residual = MLPResidualLayer(embedding_dim)
        self.post_norm = nn.LayerNorm(embedding_dim)
        offset = torch.arange(num_heads) * num_actions
        self.register_buffer("offset", offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through action embedding layer.

        Args:
            x: Integer action tensor of arbitrary shape (...,)

        Returns:
            Action embeddings of shape (..., embedding_dim)
        """
        x = self.embedding(x + self.offset)
        x = self.pre_norm(x)
        x = self.residual(x)
        x = self.post_norm(x)
        return x
