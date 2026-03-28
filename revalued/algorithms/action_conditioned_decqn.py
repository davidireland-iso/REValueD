"""Action conditioned DecQN implementation."""
import copy
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from torch.nn import HuberLoss
from torch.nn.utils import clip_grad_norm_

from .base import BaseAlgorithm


class ActionConditionedDecQN(BaseAlgorithm):
    """Action conditioned DecQN algorithm."""

    def __init__(
        self,
        state_dim: int,
        action_space,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        n_steps: int = 1,
        grad_clip: float = 40.0,
        actor_lr: float = 1e-3,
        action_dim: int = 32,
        **kwargs
    ):
        """Initialise algorithm.

        Args:
            state_dim: Dimension of state space
            action_space: MultiDiscrete action space
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            n_steps: Number of steps for n-step returns
            grad_clip: Gradient clipping threshold
            actor_lr: Learning rate for actor
            action_dim: Dimension of action space
            **kwargs: Additional arguments passed to BaseAlgorithm
        """
        super().__init__(state_dim=state_dim, action_space=action_space, **kwargs)
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Algorithm parameters
        self.n_steps = n_steps
        self.grad_clip = grad_clip

        # Action space info
        self.num_heads = len(action_space)
        self.max_action_dim = max(space.n for space in action_space)

        # Loss function
        self.loss_fn = HuberLoss()

        # Build networks
        self.actor_lr = actor_lr
        self.action_dim = action_dim
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.build_networks()

    def build_networks(self) -> None:
        """Initialise Q-networks, Policy network and optimiser."""
        # Import here to avoid circular imports
        from ..networks import ActionConditionedDecoupledQNetwork, Actor

        # Initialise networks
        self.critic = ActionConditionedDecoupledQNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_size,
            num_actions=self.max_action_dim,
            num_heads=self.num_heads,
            action_embedding_dim=self.action_dim,
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.optimiser = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate
        )

        self.actor = Actor(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_size,
            num_actions=self.max_action_dim,
            num_heads=self.num_heads,
        )
        self.actor_optimiser = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr
        )

    def act(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Select action using epsilon-greedy exploration.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore
            self.epsilon = max(
                self.epsilon * self.epsilon_decay,
                self.epsilon_min
            )
            return self.action_space.sample()
        else:
            # Exploit
            return self.greedy_act(state, **kwargs)

    def greedy_act(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Select action greedily based on Q-values.

        Args:
            state: Current state

        Returns:
            Greedy action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions, _, _ = self.actor.sample(state_tensor)
            # q_values = self.critic.forward(state_tensor, actions=action_estimates)
            # actions = q_values.argmax(dim=-1).cpu().numpy().flatten()

        return actions.cpu().numpy().flatten()

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one gradient update on Q-network and actor.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags

        Returns:
            Dictionary of metrics
        """
        # Get Q-values for selected actions
        q_values = self.critic.forward(states, actions=actions)
        selected_q_values = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_value = selected_q_values.mean(dim=-1, keepdim=True)

        # Compute targets using double Q-learning
        with torch.no_grad():
            # Query the current policy
            next_estimated_actions, _, _ = self.actor.sample(next_states)

            # Get best actions from online network
            next_q_values = self.critic.forward(next_states, actions=next_estimated_actions)
            next_actions = next_q_values.argmax(dim=-1)

            # Evaluate actions using target network
            next_q_values_target = self.critic_target.forward(next_states, actions=next_estimated_actions)
            next_q_value = next_q_values_target.gather(
                -1, next_actions.unsqueeze(-1)
            ).squeeze(-1).mean(dim=-1, keepdim=True)

            # Compute n-step targets
            targets = rewards + (self.gamma ** self.n_steps) * (1 - dones) * next_q_value

        # Compute loss
        loss = self.loss_fn(q_value, targets)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimiser.step()

        # Update the actor
        with torch.no_grad():
            estimated_actions, _, _ = self.actor.sample(states)
            curr_q_values = self.critic.forward(states, actions=estimated_actions)
            curr_actions = curr_q_values.argmax(dim=-1)

        _, _, logits = self.actor.sample(state=states, actions=curr_actions)
        # actor_loss = -log_probs.mean()
        actor_loss = self.cross_entropy_loss(
            logits.reshape(-1, logits.size(-1)),
            curr_actions.reshape(-1)
        )
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimiser.step()

        # Update counters and target network
        self.grad_steps += 1
        self.update_target_networks()

        return {
            'critic_loss': loss.item(),
            'q_value': q_value.mean().item(),
            'epsilon': self.epsilon,
            'actor_loss': actor_loss.item(),
        }

    def load(self, path: Union[str, Path], infer_architecture: bool = True) -> None:
        """Load model parameters.

        Args:
            path: Path to load model from
            infer_architecture: If True, infer network architecture from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        if infer_architecture:
            # Infer architecture from the critic's state dict
            critic_state = checkpoint['critic']

            # Extract hidden_dim from first layer weight shape
            first_layer_key = 'input_layer.weight'
            self.hidden_size = critic_state[first_layer_key].shape[0]

        # Build networks if they don't exist
        self.build_networks()
        checkpoint = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.optimiser.load_state_dict(checkpoint['optimiser'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_optimiser.load_state_dict(checkpoint['actor_optimiser'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.grad_steps = checkpoint.get('grad_steps', 0)

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
            'actor': self.actor.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'actor_optimiser': self.actor_optimiser.state_dict(),
            'total_steps': self.total_steps,
            'grad_steps': self.grad_steps,
        }
        torch.save(checkpoint, path)