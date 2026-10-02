import numpy as np
from ba_core import kl_divergence

def accelerated_ba(W, alpha=0.7, tol=1e-6, max_iter=10_000):
    """
    Practical accelerated Blahut–Arimoto using exponent squeezing (Yu, 2010).
    """
    W = np.array(W, dtype=float)
    X, Y = W.shape

    p = np.ones(X) / X
    capacity_history = []

    for it in range(max_iter):
        q = p @ W

        D = np.array([kl_divergence(W[x], q) for x in range(X)])

        C_lower = np.dot(p, D)
        C_upper = np.max(D)
        gap = C_upper - C_lower

        capacity_history.append(C_lower)

        if gap < tol:
            break

        D_tilde = alpha * (D - C_lower)
        p = p * np.exp(D_tilde)
        p /= np.sum(p)

    return {
        "capacity": C_lower / np.log(2),
        "p": p,
        "iterations": it + 1,
        "gap": gap,
        "history": np.array(capacity_history) / np.log(2)
    }
