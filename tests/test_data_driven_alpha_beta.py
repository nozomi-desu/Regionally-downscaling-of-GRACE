from __future__ import annotations

import unittest

import numpy as np

from scripts.preprocess.build_data_driven_alpha_beta import (
    assert_train_only,
    combine_beta,
    compute_alpha,
    forward_gradient,
    gradient_disagreement,
    gradient_scale,
)


class DataDrivenAlphaTests(unittest.TestCase):
    def test_identical_wghm_and_jpl_give_alpha_one(self) -> None:
        values = np.array([1.0, 3.0, -2.0])[:, None, None]
        alpha, disagreement, _, _ = compute_alpha(values, values, np.ones((1, 1), dtype=bool))
        self.assertAlmostEqual(float(disagreement[0, 0]), 0.0)
        self.assertAlmostEqual(float(alpha[0, 0]), 1.0)

    def test_equal_magnitude_opposite_series_give_alpha_zero(self) -> None:
        wghm = np.array([-1.0, 1.0, -2.0, 2.0])[:, None, None]
        jpl = -wghm
        alpha, disagreement, _, _ = compute_alpha(wghm, jpl, np.ones((1, 1), dtype=bool))
        self.assertAlmostEqual(float(disagreement[0, 0]), 1.0)
        self.assertAlmostEqual(float(alpha[0, 0]), 0.0)

    def test_zero_denominator_uses_conservative_alpha_zero(self) -> None:
        zeros = np.zeros((4, 1, 1), dtype=float)
        alpha, disagreement, _, _ = compute_alpha(zeros, zeros, np.ones((1, 1), dtype=bool))
        self.assertAlmostEqual(float(disagreement[0, 0]), 1.0)
        self.assertAlmostEqual(float(alpha[0, 0]), 0.0)

    def test_training_timestamp_leakage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_train_only(np.array(["2016-12", "2017-01"], dtype="datetime64[M]"), "2016-12-01")


class DataDrivenBetaTests(unittest.TestCase):
    @staticmethod
    def gradients(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        valid = np.ones_like(field, dtype=bool)
        gx, gy, vector_valid = forward_gradient(field, valid)
        scale = gradient_scale(gx, gy, vector_valid, "synthetic")
        return gx, gy, vector_valid, scale

    def test_equal_normalized_gradients_give_zero_disagreement(self) -> None:
        field = np.array([[[0.0, 1.0], [2.0, 4.0]]])
        gx, gy, valid, scale = self.gradients(field)
        disagreement = gradient_disagreement(gx, gy, valid, scale, gx, gy, valid, scale)
        self.assertAlmostEqual(float(disagreement[0, 0, 0]), 0.0)

    def test_one_side_zero_other_nonzero_gives_one(self) -> None:
        wgx = np.array([[[1.0]]])
        wgy = np.array([[[0.0]]])
        sgx = np.array([[[0.0]]])
        sgy = np.array([[[0.0]]])
        valid = np.ones((1, 1, 1), dtype=bool)
        disagreement = gradient_disagreement(wgx, wgy, valid, 1.0, sgx, sgy, valid, 1.0)
        self.assertAlmostEqual(float(disagreement[0, 0, 0]), 1.0)

    def test_both_sides_zero_give_zero(self) -> None:
        zeros = np.zeros((1, 1, 1))
        valid = np.ones_like(zeros, dtype=bool)
        disagreement = gradient_disagreement(zeros, zeros, valid, 1.0, zeros, zeros, valid, 1.0)
        self.assertAlmostEqual(float(disagreement[0, 0, 0]), 0.0)

    def test_missing_source_is_excluded_from_median(self) -> None:
        disagreements = np.array(
            [
                [[[[0.2]]], [[[0.4]]]],
                [[[[np.nan]]], [[[0.8]]]],
                [[[[0.6]]], [[[np.nan]]]],
            ]
        ).reshape(3, 2, 1, 1)
        beta, d_beta, count = combine_beta(disagreements)
        # time medians: 0.4 and 0.6; temporal median is 0.5
        self.assertAlmostEqual(float(d_beta[0, 0]), 0.5)
        self.assertAlmostEqual(float(beta[0, 0]), 0.5)
        self.assertEqual(int(count[0, 0]), 2)

    def test_time_with_all_sources_missing_is_excluded(self) -> None:
        disagreements = np.array([0.2, np.nan, 0.6], dtype=float).reshape(1, 3, 1, 1)
        beta, d_beta, count = combine_beta(disagreements)
        self.assertAlmostEqual(float(d_beta[0, 0]), 0.4)
        self.assertAlmostEqual(float(beta[0, 0]), 0.6)
        self.assertEqual(int(count[0, 0]), 2)

    def test_pixel_without_any_evidence_gets_beta_zero(self) -> None:
        disagreements = np.full((3, 4, 1, 1), np.nan)
        beta, d_beta, count = combine_beta(disagreements)
        self.assertAlmostEqual(float(d_beta[0, 0]), 1.0)
        self.assertAlmostEqual(float(beta[0, 0]), 0.0)
        self.assertEqual(int(count[0, 0]), 0)

    def test_alpha_beta_are_within_unit_interval(self) -> None:
        disagreements = np.array([0.0, 0.5, 1.0], dtype=float).reshape(1, 3, 1, 1)
        beta, d_beta, _ = combine_beta(disagreements)
        self.assertTrue(np.all((beta >= 0) & (beta <= 1)))
        self.assertTrue(np.all((d_beta >= 0) & (d_beta <= 1)))


if __name__ == "__main__":
    unittest.main()

