from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import torch

from revalued.networks.base import BaseQNetwork


class BaseAlgorithm(ABC):
    """Abstract base class for reinforcement learning algorithms.

    This class provides common functionality for all RL algorithms including:
    - Action selection (with/without exploration)
    - Network updates
    - Model saving/loading
    - Target network updates
    """

    def __init__(
            self,
            state_dim: int,
            action_space: Any,
            hidden_size: int = 256,
            batch_size: int = 256,
            gamma: float = 0.99,
            tau: Union[float, int] = 0.005,
            learning_rate: float = 1e-3,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            **kwargs
    ):
        """Initialise base algorithm.

        Args:
            state_dim: Dimension of state space
            action_space: Gym action space
            hidden_size: Hidden layer size for networks
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Target network update rate (soft if 0<tau<1, hard if int)
            learning_rate: Learning rate for optimiser
            device: Device to run on ('cuda' or 'cpu')
            **kwargs: Additional arguments for specific algorithms
        """
        self.state_dim = state_dim
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.device = device

        # Networks (initialised by child classes)
        self.critic: Optional[BaseQNetwork] = None
        self.critic_target: Optional[BaseQNetwork] = None
        self.optimiser: Optional[torch.optim.Optimizer] = None

        # Tracking
        self.total_steps = 0
        self.grad_steps = 0

    @abstractmethod
    def act(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Select action given state (with exploration)."""
        pass

    @abstractmethod
    def greedy_act(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Select action greedily (without exploration)."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """Perform one gradient update step."""
        pass

    @abstractmethod
    def build_networks(self) -> None:
        """Initialise neural networks."""
        pass

    def update_target_networks(self) -> None:
        """Update target networks using Polyak averaging or hard update."""
        if not self.critic or not self.critic_target:
            raise ValueError("Networks not initialised. Call build_networks() first.")

        if 0 < self.tau < 1:
            # Soft update (Polyak averaging)
            for param, target_param in zip(
                    self.critic.parameters(),
                    self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        else:
            # Hard update
            if self.grad_steps % int(self.tau) == 0:
                self.critic_target.load_state_dict(self.critic.state_dict())

    def save(self, path: Union[str, Path]) -> None:
        """Save model parameters.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'total_steps': self.total_steps,
            'grad_steps': self.grad_steps,
        }
        torch.save(checkpoint, path)

    def load(self, path: Union[str, Path]) -> None:
        """Load model parameters.

        Args:
            path: Path to load model from
        """
        if not self.critic:
            raise ValueError("Networks not initialised. Call build_networks() first.")

        checkpoint = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.optimiser.load_state_dict(checkpoint['optimiser'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.grad_steps = checkpoint.get('grad_steps', 0)
