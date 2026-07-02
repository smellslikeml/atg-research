"""Non-parametric refinement of the retrieval corpus embeddings.

This module raises the inner-product rank of known positive (viewer, entity)
pairs by nudging each corpus entity embedding toward the mean direction of the
viewer/query embeddings that positively engaged with it. It touches only the
fixed corpus embeddings -- no model weights, no gradients, no extra training
step -- so it slots between corpus construction and ``compute_all_ranks`` in the
lightweight two-tower evaluator.

Adapted from "NUDGE: Lightweight Non-Parametric Fine-Tuning of Embeddings for
Retrieval" (Kaddour et al., arXiv:2409.02343). We implement the NUDGE-N variant
(unit-norm constrained), which matches InteractRank's ``LpNormalize`` towers that
score by dot product on L2-normalized embeddings. The paper's validation-driven
optimal step-size search is intentionally left out: a single bounded step is
enough to move the corpus in the paper's direction for an evaluation hook.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Bounded per-entity move size. Small relative to the unit-norm embeddings so a
# single refinement step nudges positives up without collapsing the geometry.
DEFAULT_NUDGE_STEP = 0.1


def nudge_refine_corpus(
    corpus_embeddings: Tensor,
    corpus_entity_ids: Tensor,
    query_embeddings: Tensor,
    positive_entity_ids: Tensor,
    labels: Tensor = None,
    step: float = DEFAULT_NUDGE_STEP,
    normalize: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """NUDGE-N refinement of a fixed retrieval corpus.

    For every corpus entity, aggregate the query embeddings of the training
    records for which that entity is the labeled positive, and move the corpus
    embedding one bounded step toward that aggregate direction. Corpus entities
    with no positive query stay untouched.

    :param corpus_embeddings: [C, D] fixed corpus (entity/pin) embeddings
    :param corpus_entity_ids: [C] entity id for each corpus row
    :param query_embeddings: [B, D] viewer/query embeddings of eval records
    :param positive_entity_ids: [B] entity id each record engaged with
    :param labels: optional [B] mask; when given, only records with a truthy
        label contribute (mirrors the positive-only convention of
        ``compute_all_ranks``). When None, every record is treated as positive.
    :param step: bounded move size toward the aggregated query direction
    :param normalize: when True, re-project refined rows onto the unit sphere so
        dot-product scoring stays consistent with the L2-normalized towers
    :param eps: numerical floor for norm divisions
    :return: [C, D] refined corpus embeddings (a new tensor; input untouched)
    """
    if corpus_embeddings.ndim != 2:
        raise ValueError(f"corpus_embeddings must be [C, D], got {tuple(corpus_embeddings.shape)}")
    if query_embeddings.shape[0] != positive_entity_ids.shape[0]:
        raise ValueError("query_embeddings and positive_entity_ids must have the same number of records")

    device = corpus_embeddings.device
    query_embeddings = query_embeddings.to(device=device, dtype=corpus_embeddings.dtype)
    positive_entity_ids = positive_entity_ids.to(device)

    if labels is not None:
        mask = labels.to(device).bool()
        query_embeddings = query_embeddings[mask]
        positive_entity_ids = positive_entity_ids[mask]

    num_corpus = corpus_embeddings.shape[0]
    # Map each positive record onto its corpus row (or -1 if the entity is not
    # in the corpus, e.g. it was pruned by the entity_limit cut).
    id_to_row = {int(entity_id): row for row, entity_id in enumerate(corpus_entity_ids.tolist())}
    record_rows = torch.tensor(
        [id_to_row.get(int(entity_id), -1) for entity_id in positive_entity_ids.tolist()],
        device=device,
        dtype=torch.long,
    )
    keep = record_rows >= 0
    record_rows = record_rows[keep]
    contributing_queries = query_embeddings[keep]

    accum = torch.zeros_like(corpus_embeddings)
    counts = torch.zeros(num_corpus, device=device, dtype=corpus_embeddings.dtype)
    if record_rows.numel() > 0:
        accum.index_add_(0, record_rows, contributing_queries)
        counts.index_add_(0, record_rows, torch.ones_like(record_rows, dtype=corpus_embeddings.dtype))

    has_positive = counts > 0
    refined = corpus_embeddings.clone()
    if has_positive.any():
        mean_query = accum[has_positive] / counts[has_positive].unsqueeze(1)
        direction = mean_query / (mean_query.norm(dim=1, keepdim=True) + eps)
        refined[has_positive] = corpus_embeddings[has_positive] + step * direction

    if normalize:
        refined = refined / (refined.norm(dim=1, keepdim=True) + eps)

    logger.info(
        "NUDGE-N refined %d/%d corpus entities (step=%.4g, normalize=%s)",
        int(has_positive.sum()),
        num_corpus,
        step,
        normalize,
    )
    return refined
