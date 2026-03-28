"""Policy network."""
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import Optional, Tuple

from .layers import MLPResidualLayer, VectorisedLinear


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int, num_heads: int) -> None:
        """Initialise Actor network."""
        super(Actor, self).__init__()
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.resnet = MLPResidualLayer(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_heads = VectorisedLinear(hidden_dim, num_actions, num_heads)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass of the actor network. Returns logits."""
        # Shared layers
        x = self.input_layer(state)
        x = self.layer_norm(self.resnet(x))

        # Expand for vectorised computation
        x = x.unsqueeze(dim=0).repeat(self.num_heads, 1, 1)

        # Apply separate linear transformation for each head
        logits = self.output_heads(x).transpose(0, 1)

        return logits

    def sample(
            self,
            state: Optional[torch.Tensor] = None,
            logits: Optional[torch.Tensor] = None,
            actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from the actor network. Returns sampled actions and log probs."""
        if (state is None) == (logits is None):
            raise RuntimeError('Either state or logits must be provided.')

        if logits is None:
            logits = self.forward(state)

        distn = Categorical(logits=logits)
        if actions is None:
            actions = distn.sample()

        return actions, distn.log_prob(actions), logits
