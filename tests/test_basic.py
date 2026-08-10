import numpy as np
from scipy.linalg import expm

def test_symmetric_matrix_exponential():
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    C = expm(A)
    assert C.shape == (2, 2)
    assert np.allclose(C, C.T)

def test_pair_mask():
    n = 5
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    assert mask.sum() == 10
