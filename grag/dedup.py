import numpy as np


def residual_novelty(new_emb: np.ndarray, existing_embs: np.ndarray) -> tuple[float, float]:
    """
    Return (max_cosine, residual_norm).

    max_cosine: highest cosine similarity between new_emb and any single existing_emb.
    residual_norm: L2 norm of new_emb after projecting onto the subspace spanned by
      existing_embs, computed via QR for numerical stability.

    Both inputs assumed L2-normalised. Returns (0.0, 1.0) if existing_embs is empty.
    """
    if existing_embs.size == 0 or existing_embs.shape[0] == 0:
        return 0.0, 1.0

    cosines = existing_embs @ new_emb
    max_cosine = float(np.max(cosines))

    Q, _ = np.linalg.qr(existing_embs.T)
    proj = Q @ (Q.T @ new_emb)
    residual = new_emb - proj
    residual_norm = float(np.linalg.norm(residual))

    return max_cosine, residual_norm
