"""Q-network architectures for DecQN and REValueD algorithms."""
import torch
import torch.nn as nn

from .base import BaseQNetwork
from .layers import (MLPResidualLayer, VectorisedLinear, VectorisedLinearHead,
                     VectorisedMLPResidualLayer, ActionEmbedding)


class DecoupledQNetwork(BaseQNetwork):
    """Decoupled Q-Network for factorised action spaces.

    This network outputs separate Q-values for each action dimension,
    allowing efficient learning in factorised action spaces.
    """

    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int, num_heads: int):
        """Initialise DecoupledQNetwork.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_actions: Maximum number of actions per head
            num_heads: Number of action dimensions (heads)
        """
        super().__init__(state_dim, hidden_dim, num_actions)
        self.num_heads = num_heads

        # Network layers
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.resnet = MLPResidualLayer(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_heads = VectorisedLinear(hidden_dim, num_actions, num_heads)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: State tensor of shape (batch_size, state_dim)

        Returns:
            Q-values of shape (batch_size, num_heads, num_actions)
        """
        # Shared layers
        x = self.input_layer(x)
        x = self.layer_norm(self.resnet(x))

        # Expand for vectorised computation
        x = x.unsqueeze(dim=0).repeat(self.num_heads, 1, 1)

        # Apply separate linear transformation for each head
        vals = self.output_heads(x).transpose(0, 1)

        return vals


class EnsembleDecoupledQNetwork(BaseQNetwork):
    """Ensemble of Decoupled Q-Networks for REValueD algorithm.

    This network maintains multiple Q-networks in an ensemble,
    each with separate parameters but computed in parallel for efficiency.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_actions: int,
        num_heads: int,
        ensemble_size: int
    ):
        """Initialise EnsembleDecoupledQNetwork.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            num_actions: Maximum number of actions per head
            num_heads: Number of action dimensions (heads)
            ensemble_size: Number of networks in ensemble
        """
        super().__init__(state_dim, hidden_dim, num_actions)
        self.num_heads = num_heads
        self.ensemble_size = ensemble_size

        # Ensemble network layers
        self.input_layer = VectorisedLinear(state_dim, hidden_dim, ensemble_size)
        self.resnet_layer = VectorisedMLPResidualLayer(hidden_dim, ensemble_size=ensemble_size)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_heads = VectorisedLinearHead(hidden_dim, num_actions, ensemble_size, num_heads)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through ensemble network.

        Args:
            x: State tensor of shape (batch_size, state_dim)

        Returns:
            Q-values of shape (batch_size, ensemble_size, num_heads, num_actions)
        """
        # Expand input for ensemble
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1).repeat(1, self.ensemble_size, 1)

        # Reshape for vectorised computation
        x = x.transpose(0, 1)

        # Shared layers (computed in parallel for all ensemble members)
        x = self.input_layer(x)
        x = self.layer_norm(self.resnet_layer(x))

        # Expand for heads
        x = x.unsqueeze(dim=1).repeat(1, self.num_heads, 1, 1)

        # Apply output heads
        q_values = self.output_heads(x)

        # Reshape to (batch_size, ensemble_size, num_heads, num_actions)
        q_values = q_values.transpose(1, 2).transpose(0, 1)

        return q_values


class ActionConditionedDecoupledQNetwork(DecoupledQNetwork):
    """Decoupled Q-Network conditioned on other heads' action embeddings."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_actions: int,
        num_heads: int,
        action_embedding_dim: int = 16,
        conditioning: str = "concat",  # "concat" or "add"
    ) -> None:
        super().__init__(state_dim, hidden_dim, num_actions, num_heads)
        assert conditioning == "concat" or conditioning == "add", f"{conditioning} not supported"
        self.action_embedding_dim = action_embedding_dim
        self.conditioning = conditioning

        self.action_embedding = ActionEmbedding(num_actions, num_heads, action_embedding_dim)

        # Projects the (num_heads - 1) * d other-action embeddings down to hidden_dim
        self.action_projection = (
            nn.Linear(
            (num_heads - 1) * action_embedding_dim, hidden_dim
        ))

        head_input_dim = hidden_dim * 2 if conditioning == "concat" else hidden_dim
        self.output_heads = VectorisedLinear(head_input_dim, num_actions, num_heads)

        # Precompute leave-one-out indices: (num_heads, num_heads - 1)
        idx = torch.arange(num_heads)
        loo_idx = torch.stack([
            torch.cat([idx[:i], idx[i+1:]])
            for i in range(num_heads)
        ])  # (num_heads, num_heads - 1)
        self.register_buffer("loo_idx", loo_idx)

    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State tensor of shape (batch_size, state_dim)
            actions: Action tensor of shape (batch_size, num_heads), integer-valued

        Returns:
            Q-values of shape (batch_size, num_heads, num_actions)
        """
        batch_size = x.shape[0]

        x = self.input_layer(x)
        x = self.layer_norm(self.resnet(x))
        # x: (batch_size, hidden_dim)

        # Action embeddings
        emb = self.action_embedding(actions)
        # emb: (batch_size, num_heads, action_embedding_dim)

        # Leave-one-out gather
        # loo_idx: (num_heads, num_heads-1) -> expand to (batch_size, num_heads, num_heads-1)
        loo = self.loo_idx.unsqueeze(0).expand(batch_size, -1, -1)
        # Gather along head dim: for each head i, pick num_heads-1 embeddings
        # emb expanded: (batch_size, num_heads, num_heads-1, d)
        other_embs = emb.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        other_embs = other_embs.gather(
            dim=1,
            index=loo.unsqueeze(-1).expand(-1, -1, -1, self.action_embedding_dim)
        )
        # other_embs: (batch_size, num_heads, num_heads-1, d)

        # Flatten and project
        other_embs = other_embs.flatten(start_dim=2)
        # (batch_size, num_heads, (num_heads-1) * d)
        action_ctx = torch.relu(self.action_projection(other_embs))
        # action_ctx: (batch_size, num_heads, hidden_dim)

        # Expand state and combine with action context
        x = x.unsqueeze(1).expand(-1, self.num_heads, -1)
        # x: (batch_size, num_heads, hidden_dim)

        if self.conditioning == "concat":
            x = torch.cat([x, action_ctx], dim=-1)
        else:
            x = x + action_ctx
        # x: (batch_size, num_heads, hidden_dim * 2) or (batch_size, num_heads, hidden_dim)

        # VectorisedLinear expects (num_heads, batch_size, features)
        x = x.transpose(0, 1)
        vals = self.output_heads(x).transpose(0, 1)
        # vals: (batch_size, num_heads, num_actions)

        return vals
