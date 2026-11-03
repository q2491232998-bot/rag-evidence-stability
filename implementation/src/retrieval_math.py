import numpy as np


def stable_topk(scores, indices, k):
    scores = np.asarray(scores)
    indices = np.asarray(indices, dtype=np.int64)
    assert scores.ndim == indices.ndim == 1
    assert scores.shape == indices.shape
    assert np.isfinite(scores).all()
    assert k > 0
    if len(scores) <= k:
        chosen = np.arange(len(scores))
    else:
        partial = np.argpartition(scores, len(scores) - k)[-k:]
        boundary = scores[partial].min()
        above = np.flatnonzero(scores > boundary)
        tied = np.flatnonzero(scores == boundary)
        tied = tied[np.argsort(indices[tied], kind='stable')[:k - len(above)]]
        chosen = np.concatenate([above, tied])
    order = np.lexsort((indices[chosen], -scores[chosen]))
    chosen = chosen[order]
    return scores[chosen], indices[chosen]


def merge_topk(previous_scores, previous_indices, new_scores, first_index, k):
    local_scores, local_indices = stable_topk(new_scores, np.arange(first_index, first_index + len(new_scores)), k)
    return stable_topk(np.concatenate([previous_scores, local_scores]), np.concatenate([previous_indices, local_indices]), k)
