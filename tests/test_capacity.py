"""Independent numerical checks for the original supplied solvers."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from ba_core import blahut_arimoto
from ba_accelerated import accelerated_ba
from channels import BSC, BEC


class CapacityTests(unittest.TestCase):
    def assert_result(self, result, expected):
        self.assertAlmostEqual(result["capacity"], expected, places=7)
        self.assertAlmostEqual(float(result["p"].sum()), 1.0, places=12)
        self.assertTrue(np.all(result["p"] >= 0))
        self.assertLess(result["gap"], 1e-9)

    def test_binary_symmetric_analytic_capacity(self):
        for solver in (blahut_arimoto, accelerated_ba):
            for p in (0.0, 0.01, 0.1, 0.3, 0.5, 1.0):
                with self.subTest(solver=solver.__name__, p=p):
                    entropy = 0.0 if p in (0.0, 1.0) else -p*np.log2(p)-(1-p)*np.log2(1-p)
                    self.assert_result(solver(BSC(p), tol=1e-9), 1-entropy)

    def test_binary_erasure_analytic_capacity(self):
        for solver in (blahut_arimoto, accelerated_ba):
            for e in (0.0, 0.2, 0.7, 1.0):
                with self.subTest(solver=solver.__name__, e=e):
                    self.assert_result(solver(BEC(e), tol=1e-9), 1-e)

    def test_identical_rows_have_zero_capacity(self):
        W = np.tile([0.2, 0.3, 0.5], (4, 1))
        for solver in (blahut_arimoto, accelerated_ba):
            self.assert_result(solver(W, tol=1e-9), 0.0)

    def test_noiseless_capacity(self):
        for solver in (blahut_arimoto, accelerated_ba):
            self.assert_result(solver(np.eye(4), tol=1e-9), 2.0)

    def test_asymmetric_channel_against_independent_grid(self):
        W = np.array([[0.9, 0.1], [0.3, 0.7]])
        weights = np.linspace(0.00001, 0.99999, 100001)
        inputs = np.column_stack([weights, 1-weights])
        outputs = inputs @ W
        divergence = (W[None, :, :] * np.log2(W[None, :, :] / outputs[:, None, :])).sum(axis=2)
        reference = (inputs * divergence).sum(axis=1).max()
        for solver in (blahut_arimoto, accelerated_ba):
            result = solver(W, tol=1e-10)
            self.assert_result(result, reference)
            self.assertGreater(result["iterations"], 1)
            self.assertTrue(np.all(np.diff(result["history"]) >= -1e-12))

    def test_alpha_one_matches_standard(self):
        W = np.array([[0.9, 0.1], [0.3, 0.7]])
        standard = blahut_arimoto(W, tol=1e-10)
        relaxed = accelerated_ba(W, alpha=1, tol=1e-10)
        np.testing.assert_allclose(standard["p"], relaxed["p"], atol=1e-12)
        self.assertAlmostEqual(standard["capacity"], relaxed["capacity"], places=12)


if __name__ == "__main__":
    unittest.main()
