import numpy as np

def kl_divergence(p, q):
    eps = 1e-15
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

def blahut_arimoto(W, tol=1e-6, max_iter=10_000):
    """
    Standard Blahut–Arimoto algorithm with duality-gap stopping rule.
    W: channel matrix of shape (|X|, |Y|)
    """
    W = np.array(W, dtype=float)
    X, Y = W.shape

    p = np.ones(X) / X
    capacity_history = []

    for it in range(max_iter):
        q = p @ W

        D = np.zeros(X)
        for x in range(X):
            D[x] = kl_divergence(W[x], q)

        C_lower = np.dot(p, D)
        C_upper = np.max(D)
        gap = C_upper - C_lower

        capacity_history.append(C_lower)

        if gap < tol:
            break

        p = p * np.exp(D)
        p /= np.sum(p)

    return {
        "capacity": C_lower / np.log(2),
        "p": p,
        "iterations": it + 1,
        "gap": gap,
        "history": np.array(capacity_history) / np.log(2)
    }
