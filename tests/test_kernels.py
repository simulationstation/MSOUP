import numpy as np
import pandas as pd

from msoup_purist_closure.kernels import KernelWeights, kernel_weights


def test_kernel_weights_k1_uniform():
    df = pd.DataFrame({"z": [0.1, 0.2, 0.3]})
    kw = kernel_weights(df, "K1")
    assert np.allclose(kw.weights, np.ones(3) / 3)


def test_kernel_weights_k2_inverse_variance():
    df = pd.DataFrame({"z": [0.1, 0.2, 0.3], "sigma": [1.0, 2.0, 1.0]})
    kw = kernel_weights(df, "K2")
    expected = np.array([1.0, 0.25, 1.0])
    expected = expected / expected.sum()
    assert np.allclose(kw.weights, expected)


def test_kernel_weights_invalid():
    df = pd.DataFrame({"z": [0.1]})
    kw = KernelWeights(df["z"].values, np.array([1.0]), "K1")
    assert kw.weights.sum() == 1.0
