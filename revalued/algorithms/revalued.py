"""REValueD: Regularised Ensemble Value Decomposition."""
import copy
from typing import Dict, Literal

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .decqn import DecQN


class REValueD(DecQN):
    """REValueD algorithm for improved exploration and stability.

    This algorithm extends DecQN with an ensemble of Q-networks to:
    - Reduce overestimation bias through ensemble averaging
    - Support multiple update strategies (Mean, REDQ, DecQN)
    """

    def __init__(
            self,
            state_dim: int,
            action_space,
            ensemble_size: int = 10,
            update_type: Literal['Mean', 'REDQ', 'DecQN'] = 'Mean',
            **kwargs
    ):
        """Initialise REValueD.

        Args:
            state_dim: Dimension of state space
            action_space: MultiDiscrete action space
            ensemble_size: Number of Q-networks in ensemble
            update_type: Strategy for computing targets
                - 'Mean': Use mean of ensemble
                - 'REDQ': Use minimum of random subset
                - 'DecQN': Train networks independently
            **kwargs: Additional arguments passed to DecQN
        """
        self.ensemble_size = ensemble_size
        self.update_type = update_type

        super().__init__(state_dim, action_space, **kwargs)

    def build_networks(self) -> None:
        """Initialise ensemble Q-networks and optimiser."""
        # Import here to avoid circular imports
        from ..networks import EnsembleDecoupledQNetwork

        # Create action mask
        self.action_mask = self._create_action_mask()

        # Initialise ensemble networks
        self.critic = EnsembleDecoupledQNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_size,
            num_actions=self.max_action_dim,
            num_heads=self.num_heads,
            ensemble_size=self.ensemble_size
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.optimiser = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate
        )

    def act(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Select action using randomised ensemble member.

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
            # Exploit using random ensemble member
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.critic(state_tensor) + self.action_mask
                # Select random ensemble member for action selection
                ensemble_idx = np.random.randint(self.ensemble_size)
                actions = q_values[0, ensemble_idx].argmax(dim=-1).cpu().numpy().flatten()

            return actions

    def greedy_act(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Select action greedily using ensemble mean.

        Args:
            state: Current state

        Returns:
            Greedy action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.critic(state_tensor) + self.action_mask
            # Use mean of ensemble for greedy action
            mean_q_values = q_values.mean(dim=1)
            actions = mean_q_values.argmax(dim=-1).cpu().numpy().flatten()

        return actions

    def update(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one gradient update on ensemble Q-networks.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards (expanded for ensemble)
            next_states: Batch of next states
            dones: Batch of done flags (expanded for ensemble)

        Returns:
            Dictionary of metrics
        """
        # Reshape for ensemble
        batch_size = states.shape[0]
        actions = actions.view(batch_size, 1, -1).repeat(1, self.ensemble_size, 1)
        rewards = rewards.view(batch_size, 1).repeat(1, self.ensemble_size)
        dones = dones.view(batch_size, 1).repeat(1, self.ensemble_size)

        # Get Q-values for selected actions
        q_values = self.critic(states) + self.action_mask
        selected_q_values = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_value = selected_q_values.mean(dim=-1)

        # Compute targets based on update type
        with torch.no_grad():
            targets = self._compute_targets(next_states)
            targets = rewards + (self.gamma ** self.n_steps) * (1 - dones) * targets

        # Compute loss
        loss = self.loss_fn(q_value, targets).mean(dim=0).sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimiser.step()

        # Update counters and target network
        self.grad_steps += 1
        self.update_target_networks()

        return {
            'critic_loss': loss.item(),
            'q_value': q_value.mean().item(),
            'epsilon': self.epsilon
        }

    def _compute_targets(self, next_states: torch.Tensor) -> torch.Tensor:
        """Compute targets based on selected update strategy.

        Args:
            next_states: Batch of next states

        Returns:
            Target Q-values
        """
        if self.update_type == 'DecQN':
            # Independent training for each ensemble member
            next_actions = (self.critic(next_states) + self.action_mask).argmax(dim=-1)
            next_q_values = (self.critic_target(next_states) + self.action_mask)
            targets = next_q_values.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            targets = targets.mean(dim=-1)

        elif self.update_type == 'REDQ':
            # Random ensemble distillation Q-learning
            next_q_values = (self.critic_target(next_states) + self.action_mask)
            max_q_values = next_q_values.max(dim=-1)[0].mean(dim=-1)

            # Select random subset of ensemble
            idx = np.random.choice(self.ensemble_size, size=2, replace=False)
            targets = max_q_values[:, idx].min(dim=-1, keepdim=True)[0]
            targets = targets.repeat(1, self.ensemble_size)

        elif self.update_type == 'Mean':
            # Mean of ensemble
            next_q_values = (self.critic_target(next_states) + self.action_mask)
            mean_q_values = next_q_values.mean(dim=1)
            targets = mean_q_values.max(dim=-1)[0].mean(dim=-1, keepdim=True)
            targets = targets.repeat(1, self.ensemble_size)

        else:
            raise ValueError(f"Unknown update type: {self.update_type}")

        return targets
