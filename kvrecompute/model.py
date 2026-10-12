# SPDX-License-Identifier: Apache-2.0

"""Small regressors for token-level recompute score prediction."""

# Third Party
from torch import nn
import torch


class TokenMLPRegressor(nn.Module):
    """A compact MLP that predicts one scalar recompute score per token.

    Parameters
    ----------
    input_dim:
        Number of token-level input features.
    hidden_dim:
        Hidden dimension of the MLP.
    dropout:
        Dropout probability used between hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict scalar recompute scores for a batch of tokens."""

        return self.network(features).squeeze(-1)

