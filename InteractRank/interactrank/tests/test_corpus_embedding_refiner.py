"""Tests for NUDGE-N corpus refinement wired into the lightweight evaluator.

Exercises the wiring edit in ``interactrank.eval`` (a non-new module): the
``TwoTowerLightweightEvaluator.refine_corpus_embeddings`` hook and the
``compute_all_ranks`` metric it is meant to improve.
"""

import numpy as np
import torch

from interactrank.constants.base_constants import LABEL_FIELD
from interactrank.eval import TwoTowerLightweightEvaluator
from interactrank.eval import compute_all_ranks
from interactrank.corpus_embedding_refiner import nudge_refine_corpus


ENGAGEMENT_LABEL = "engagement_" + LABEL_FIELD


def _rank_of_positive(query, pins, entity_ids, corpus_emb, corpus_ids):
    """Rank of the single positive record via the repo's own metric."""
    ranks = compute_all_ranks(
        metric_prefix="TEST",
        viewer_embeddings=query,
        entity_embeddings=pins,
        entity_ids=entity_ids,
        user_ids=torch.zeros(query.shape[0], dtype=torch.long),
        # save_artifacts_fn is None below, so image_sigs is only sliced, never
        # compared -- pass a tensor to avoid numpy-version boolean-index quirks.
        image_sigs=torch.zeros(query.shape[0], dtype=torch.long),
        labels=torch.ones(query.shape[0], dtype=torch.long),
        entity_embedding_corpus=corpus_emb,
        entity_id_corpus=corpus_ids,
        save_artifacts_fn=None,
    )
    return int(ranks[0])


def _build_evaluator(query_np, pins_np, entity_ids_np, corpus_emb, corpus_ids, step):
    evaluator = TwoTowerLightweightEvaluator(
        device=torch.device("cpu"),
        input_dir="",
        refine_corpus=True,
        nudge_step=step,
    )
    evaluator.query_embedding_np = query_np
    evaluator.pin_embedding_np = pins_np
    evaluator.entity_ids_np = entity_ids_np
    evaluator.corpus_embeddings = corpus_emb.clone()
    evaluator.corpus_entity_ids = corpus_ids.clone()
    evaluator.labels = {ENGAGEMENT_LABEL: np.ones(query_np.shape[0], dtype=np.int64)}
    return evaluator


def test_refinement_lifts_positive_above_a_negative():
    # One positive (query, entity 20) pair. Initially entity 10 outranks it.
    query_np = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pin20 = [0.6, 0.8, 0.0, 0.0]
    pins_np = np.array([pin20], dtype=np.float32)
    entity_ids_np = np.array([20], dtype=np.int64)

    corpus_ids = torch.tensor([10, 20, 30])
    corpus_emb = torch.tensor(
        [[0.7, 0.71, 0.0, 0.0], pin20, [0.2, 0.9, 0.0, 0.0]],
        dtype=torch.float32,
    )

    query_t = torch.from_numpy(query_np)

    rank_before = _rank_of_positive(
        query_t, torch.from_numpy(pins_np), torch.from_numpy(entity_ids_np), corpus_emb, corpus_ids
    )
    assert rank_before >= 1, "expected the positive to start outranked by a negative"

    evaluator = _build_evaluator(query_np, pins_np, entity_ids_np, corpus_emb, corpus_ids, step=0.3)
    evaluator.refine_corpus_embeddings()

    rank_after = _rank_of_positive(
        query_t,
        torch.from_numpy(evaluator.pin_embedding_np),
        torch.from_numpy(entity_ids_np),
        evaluator.corpus_embeddings,
        corpus_ids,
    )
    assert rank_after < rank_before
    assert rank_after == 0


def test_untouched_when_no_positive_matches_corpus():
    # Positive engages an entity absent from the corpus -> corpus unchanged.
    corpus_ids = torch.tensor([10, 30])
    corpus_emb = torch.tensor([[0.7, 0.71, 0.0, 0.0], [0.2, 0.9, 0.0, 0.0]], dtype=torch.float32)
    refined = nudge_refine_corpus(
        corpus_embeddings=corpus_emb,
        corpus_entity_ids=corpus_ids,
        query_embeddings=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        positive_entity_ids=torch.tensor([20]),
        labels=torch.tensor([1]),
        step=0.3,
    )
    # Rows normalized but directions preserved (no positive touched either row).
    assert torch.allclose(
        refined / refined.norm(dim=1, keepdim=True),
        corpus_emb / corpus_emb.norm(dim=1, keepdim=True),
        atol=1e-5,
    )


def test_refined_positive_moves_toward_query():
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    corpus_emb = torch.tensor([[0.6, 0.8, 0.0, 0.0]], dtype=torch.float32)
    corpus_ids = torch.tensor([20])
    refined = nudge_refine_corpus(
        corpus_embeddings=corpus_emb,
        corpus_entity_ids=corpus_ids,
        query_embeddings=query,
        positive_entity_ids=torch.tensor([20]),
        labels=torch.tensor([1]),
        step=0.3,
    )
    q = query[0] / query[0].norm()
    before = float(corpus_emb[0] @ q / corpus_emb[0].norm())
    after = float(refined[0] @ q / refined[0].norm())
    assert after > before
