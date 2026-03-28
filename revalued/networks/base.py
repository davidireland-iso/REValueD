"""Base network classes and utilities."""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseQNetwork(nn.Module, ABC):
    """Abstract base class for Q-networks."""

    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int):
        """Initialise base Q-network.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_actions: Number of actions per head
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: Input tensor of shape (batch_size, state_dim)

        Returns:
            Q-values tensor
        """
        pass
