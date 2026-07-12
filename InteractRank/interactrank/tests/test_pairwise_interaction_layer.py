"""Integration tests for finite-grid pairwise cross-interaction scoring.

These tests wire the new ``PairwiseInteractionScorer`` against genuine
InteractRank modules -- ``LazyConcatInput`` (the exact op ``CrossLayer`` uses
to build its cross-feature tensor) and the real cross-feature constants -- so
we prove the scorer consumes the same representation the pre-ranking cross
layer feeds forward, not a self-contained stand-in.
"""
from __future__ import annotations

import pytest
import torch

# Non-new repo modules under test-integration (path set up in conftest.py).
from interactrank.common.lazy_concat import LazyConcatInput
from interactrank.constants.base_constants import CROSS_FEATURES
from interactrank.constants.base_constants import DOT_PRODUCT_FEATURE_FIELD
from interactrank.common.pairwise_interaction_layer import PairwiseInteractionScorer


def _cross_feature_batch(batch_size: int = 8) -> dict:
    """A formatted_data dict shaped exactly like CrossLayer's input."""
    torch.manual_seed(0)
    data = {name: torch.randn(batch_size) for name in CROSS_FEATURES}
    data[DOT_PRODUCT_FEATURE_FIELD] = torch.randn(batch_size)
    return data


def test_scorer_consumes_real_cross_concat():
    """The scorer accepts the tensor CrossLayer builds via LazyConcatInput."""
    data = _cross_feature_batch(batch_size=8)
    concat = LazyConcatInput(to_skip={DOT_PRODUCT_FEATURE_FIELD})
    cross = concat(data)  # [batch, num_cross_features] -- same op CrossLayer runs

    assert cross.shape == (8, len(CROSS_FEATURES))
    scorer = PairwiseInteractionScorer(num_features=cross.shape[1])
    out = scorer(cross)
    assert out.shape == (8,)
    assert torch.isfinite(out).all()


def test_zero_init_is_additive_noop():
    """Zero-initialized grids contribute exactly zero, so enabling the scorer
    leaves a baseline additive score untouched at step 0."""
    scorer = PairwiseInteractionScorer(num_features=len(CROSS_FEATURES))
    out = scorer(torch.randn(5, len(CROSS_FEATURES)))
    assert torch.count_nonzero(out) == 0


def test_complexity_budget_caps_admitted_pairs():
    """The explanation budget bounds the number of instantiated pair terms."""
    full = PairwiseInteractionScorer(num_features=5)
    assert full.num_pairs == 10  # C(5, 2)

    budgeted = PairwiseInteractionScorer(num_features=5, max_pairs=3)
    assert budgeted.num_pairs == 3
    assert len(budgeted.describe_pairs()) == 3
    assert budgeted.interaction_grids.shape == (3, 8, 8)


def test_scorer_is_differentiable():
    """Grids receive gradients so the scorer trains in the existing SGD path."""
    scorer = PairwiseInteractionScorer(num_features=4, num_bins=4)
    scorer(torch.randn(6, 4)).sum().backward()
    assert scorer.interaction_grids.grad is not None
    assert scorer.bin_centers.grad is not None


def test_captures_pure_interaction_signal():
    """Core IAIML result: an additive linear head cannot fit a label with pure
    pairwise-interaction structure and no marginal signal, but adding the
    finite-grid interaction terms recovers it.

    Label = XOR(sign(x0), sign(x1)): each feature is marginally uninformative,
    the signal lives only in the joint configuration -- exactly the case IAIML
    was built for and the case a linear CrossLayer is blind to.
    """
    torch.manual_seed(1)
    n = 512
    x = torch.randn(n, 4)
    label = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).float()

    def train(model_params, forward, steps=400):
        opt = torch.optim.Adam(model_params, lr=0.05)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(forward(), label)
            loss.backward()
            opt.step()
        return loss_fn(forward(), label).item()

    # Additive baseline: mirrors CrossLayer's nn.Linear over the cross features.
    linear = torch.nn.Linear(4, 1)
    additive_loss = train(list(linear.parameters()), lambda: linear(x).squeeze(-1))

    # Additive head + finite-grid pairwise interaction terms (the integration).
    linear2 = torch.nn.Linear(4, 1)
    scorer = PairwiseInteractionScorer(num_features=4, num_bins=6)
    interaction_loss = train(
        list(linear2.parameters()) + list(scorer.parameters()),
        lambda: linear2(x).squeeze(-1) + scorer(x),
    )

    # The interaction-aware scorer fits the joint signal the linear head cannot.
    assert additive_loss > 0.6  # near chance -- no marginal signal to exploit
    assert interaction_loss < additive_loss - 0.2


def _load_cross_layer():
    """Import the real CrossLayer, or skip if this environment can't build its
    (heavy, older-Python) import chain."""
    try:
        from interactrank.model import CrossLayer

        return CrossLayer
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CrossLayer import unavailable in this environment: {exc}")


def test_crosslayer_wires_interaction_scorer():
    """The CrossLayer edit instantiates the scorer only when enabled, and
    surfaces its magnitude in the summary metrics."""
    CrossLayer = _load_cross_layer()
    feature_map = {name: None for name in CROSS_FEATURES}

    disabled = CrossLayer(features=CROSS_FEATURES, feature_map=feature_map, enable_compute_average_navboost=False)
    assert disabled.interaction_scorer is None
    assert not any("interaction" in k for k in disabled.get_summary_metrics({}))

    enabled = CrossLayer(
        features=CROSS_FEATURES,
        feature_map=feature_map,
        enable_compute_average_navboost=False,
        enable_pairwise_interactions=True,
        interaction_max_pairs=3,
    )
    assert isinstance(enabled.interaction_scorer, PairwiseInteractionScorer)
    assert enabled.interaction_scorer.num_features == len(CROSS_FEATURES)
    assert enabled.interaction_scorer.num_pairs == 3
    assert "cross_interaction_grid_abs_mean" in enabled.get_summary_metrics({})
