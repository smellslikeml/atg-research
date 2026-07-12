from __future__ import annotations

from itertools import combinations
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import nn

"""
Finite-grid pairwise interaction scoring for the pre-ranking cross layer.

Adapted from "Complexity-Budgeted, Interaction-Aware Interpretable Model for
Tabular Data" (IAIML, arXiv:2607.07060). IAIML observes that an additive /
marginal combination of features is blind to predictive signal that only
emerges through the *joint* configuration of two features. It recovers that
signal with three coordinated mechanisms:

  1. adaptive per-feature discretization,
  2. finite-grid pairwise interaction scoring, and
  3. a partitioned (bounded) explanation budget.

InteractRank's ``CrossLayer`` combines the two-tower dot product with a handful
of query-item cross features through a single ``nn.Linear`` -- a purely
additive scorer with exactly the blind spot IAIML describes. This module ports
IAIML's core mechanism at full fidelity so those cross features can contribute
*explicit pair terms* on a finite grid, while staying within the cross layer's
tight latency budget (a few learnable B x B grids, evaluated with one einsum).

Target-native adaptations (see PR notes):
  * The per-feature discretization is a differentiable soft-binning over
    learnable bin centers, so it trains in InteractRank's existing SGD path
    rather than IAIML's offline discretization estimator.
  * The paper's statistical interaction *screening* (MI-style detection under
    nested cross-validation) is replaced by learning the grid weights jointly;
    the complexity budget below still bounds how many pair terms exist.
  * IAIML's second routing strategy (relaxing a pattern-search filter for a
    RuleFit-style miner) and its 40-dataset benchmark harness are out of
    scope -- only the "explicit pair terms for a sparse downstream scorer"
    route is implemented, which is the one that fits a differentiable head.
"""


def _default_pairs(num_features: int, max_pairs: Optional[int]) -> List[Tuple[int, int]]:
    """Enumerate candidate feature-index pairs, capped by the complexity budget.

    :param num_features: number of scalar features feeding the scorer
    :param max_pairs: maximum number of pair terms to admit (the partitioned
        explanation budget). ``None`` or a non-positive value admits every pair.
    :return: the admitted list of ``(i, j)`` feature-index pairs
    """
    pairs = list(combinations(range(num_features), 2))
    if max_pairs is not None and max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs


class PairwiseInteractionScorer(nn.Module):
    """Score explicit pairwise feature interactions on a finite grid.

    Each feature is softly assigned to ``num_bins`` bins via learnable,
    per-feature bin centers (adaptive discretization). For every admitted
    feature pair ``(i, j)`` a learnable ``num_bins x num_bins`` grid holds the
    interaction weight for each (bin_i, bin_j) cell; the pair's contribution is
    the bilinear form ``soft_i^T W_ij soft_j``. The scorer returns the sum of
    all admitted pair contributions as a per-example scalar, meant to be added
    to an existing additive score.

    The grids are zero-initialized, so at step 0 the scorer contributes exactly
    zero -- it augments a baseline additive scorer without perturbing it, and
    only learns interaction terms where they reduce the loss.

    :param num_features: number of scalar features feeding the scorer
    :param num_bins: number of bins in the finite grid per feature
    :param max_pairs: complexity budget -- max number of admitted pair terms
        (``None``/<=0 admits all pairs)
    :param feature_pairs: explicit ``(i, j)`` pairs to admit; overrides the
        default enumeration (still capped by ``max_pairs``)
    :param temperature: softness of the bin assignment (smaller -> harder bins)
    """

    def __init__(
        self,
        num_features: int,
        num_bins: int = 8,
        max_pairs: Optional[int] = None,
        feature_pairs: Optional[List[Tuple[int, int]]] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        if num_features < 0:
            raise ValueError(f"num_features must be non-negative, got {num_features}")
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.num_features = num_features
        self.num_bins = num_bins
        self.temperature = temperature

        pairs = feature_pairs if feature_pairs is not None else _default_pairs(num_features, max_pairs)
        if feature_pairs is not None and max_pairs is not None and max_pairs > 0:
            pairs = pairs[:max_pairs]
        self.num_pairs = len(pairs)

        # Adaptive per-feature discretization: learnable bin centers spread over a
        # normalized range; training adapts them per feature.
        centers = torch.linspace(-1.0, 1.0, num_bins).unsqueeze(0).repeat(max(num_features, 1), 1)
        self.bin_centers = nn.Parameter(centers)

        if self.num_pairs > 0:
            left = torch.tensor([i for i, _ in pairs], dtype=torch.long)
            right = torch.tensor([j for _, j in pairs], dtype=torch.long)
            # Zero init -> no contribution until interaction terms are learned.
            self.interaction_grids = nn.Parameter(torch.zeros(self.num_pairs, num_bins, num_bins))
        else:
            left = torch.zeros(0, dtype=torch.long)
            right = torch.zeros(0, dtype=torch.long)
            self.register_parameter("interaction_grids", None)
        self.register_buffer("pair_left", left)
        self.register_buffer("pair_right", right)

    def describe_pairs(self) -> List[Tuple[int, int]]:
        """:return: the admitted ``(i, j)`` feature-index pairs (for inspection)."""
        return list(zip(self.pair_left.tolist(), self.pair_right.tolist()))

    def _soft_bins(self, features: torch.Tensor) -> torch.Tensor:
        """Soft-assign each feature to bins.

        :param features: ``[batch, num_features]`` scalar features
        :return: ``[batch, num_features, num_bins]`` bin-assignment weights
        """
        # squared distance from each feature value to its learnable bin centers
        diff = features.unsqueeze(-1) - self.bin_centers.unsqueeze(0)
        return torch.softmax(-(diff * diff) / self.temperature, dim=-1)

    def pair_contributions(self, features: torch.Tensor) -> torch.Tensor:
        """Per-pair interaction scores.

        :param features: ``[batch, num_features]`` scalar features
        :return: ``[batch, num_pairs]`` interaction contribution of each pair
        """
        if self.num_pairs == 0:
            return features.new_zeros((features.shape[0], 0))
        soft = self._soft_bins(features)
        left = soft[:, self.pair_left]  # [batch, num_pairs, num_bins]
        right = soft[:, self.pair_right]  # [batch, num_pairs, num_bins]
        # bilinear form on the finite grid: sum_ij left_i * W_ij * right_j
        return torch.einsum("bpi,pij,bpj->bp", left, self.interaction_grids, right)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Total pairwise-interaction score.

        :param features: ``[batch, num_features]`` scalar features
        :return: ``[batch]`` summed interaction score to add to an additive head
        """
        if features.dim() != 2:
            raise ValueError(f"expected [batch, num_features] input, got shape {tuple(features.shape)}")
        return self.pair_contributions(features).sum(dim=-1)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, num_bins={self.num_bins}, num_pairs={self.num_pairs}"
