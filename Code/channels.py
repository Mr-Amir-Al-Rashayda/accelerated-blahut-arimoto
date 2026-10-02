import numpy as np

def BSC(p):
    return np.array([
        [1 - p, p],
        [p, 1 - p]
    ])

def BEC(e):
    return np.array([
        [1 - e, 0, e],
        [0, 1 - e, e]
    ])
