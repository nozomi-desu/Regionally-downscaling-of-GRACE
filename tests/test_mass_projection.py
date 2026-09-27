from __future__ import annotations

import unittest

import numpy as np
import torch

from utils.mass_closure import (
    latitude_area_weights,
    masked_pool2d_mean,
    masked_pool2d_mean_numpy,
    project_to_coarse_constraint_numpy,
    project_to_coarse_constraint_torch,
)


class StrictMassProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.factor = 2
        self.lat = np.array([-75.0, -25.0, 25.0, 75.0])
        self.area = latitude_area_weights(self.lat, width=4)
        self.pred = np.array(
            [
                [1.0, 4.0, 2.0, 8.0],
                [3.0, 7.0, 5.0, 9.0],
                [2.0, 6.0, 1.0, 3.0],
                [8.0, 4.0, 7.0, 5.0],
            ],
            dtype=np.float64,
        )
        self.mask = np.ones_like(self.pred)
        self.mask[0, 0] = 0.0
        self.mask[3, 3] = 0.0
        self.target = np.array([[10.0, -2.0], [4.0, 12.0]], dtype=np.float64)

    def test_numpy_projection_closes_area_weighted_blocks(self) -> None:
        projected, diagnostics = project_to_coarse_constraint_numpy(
            self.pred,
            self.target,
            self.mask,
            factor=self.factor,
            area_weights=self.area,
        )
        aggregated, valid = masked_pool2d_mean_numpy(
            projected,
            self.mask * self.area,
            factor=self.factor,
            eps=1e-12,
        )
        np.testing.assert_allclose(aggregated[valid], self.target[valid], rtol=0.0, atol=1e-10)
        self.assertLess(diagnostics["mass_closure_max_abs_after"], 1e-10)
        self.assertEqual(diagnostics["operator"], "coslat_area_weighted_block_mean")
        self.assertEqual(projected[0, 0], self.pred[0, 0])
        self.assertEqual(projected[3, 3], self.pred[3, 3])

    def test_torch_projection_is_differentiable_and_strict(self) -> None:
        pred = torch.tensor(self.pred, dtype=torch.float64, requires_grad=True)
        target = torch.tensor(self.target, dtype=torch.float64)
        mask = torch.tensor(self.mask, dtype=torch.float64)
        area = torch.tensor(self.area, dtype=torch.float64)
        projected, diagnostics = project_to_coarse_constraint_torch(
            pred,
            target,
            mask,
            factor=self.factor,
            area_weights=area,
        )
        pooled, valid = masked_pool2d_mean(
            projected,
            mask * area,
            kernel_size=self.factor,
            eps=1e-12,
        )
        np.testing.assert_allclose(
            pooled.detach().numpy()[valid.detach().numpy().astype(bool)],
            target.numpy().reshape(1, 1, 2, 2)[valid.detach().numpy().astype(bool)],
            rtol=0.0,
            atol=1e-10,
        )
        self.assertLess(float(diagnostics["mass_closure_max_abs_after"].detach()), 1e-10)
        (projected.square().mean()).backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_latitude_weights_reduce_toward_poles(self) -> None:
        weights = latitude_area_weights(np.array([0.0, 60.0, 80.0]))
        self.assertAlmostEqual(float(weights[0, 0]), 1.0)
        self.assertGreater(float(weights[1, 0]), float(weights[2, 0]))
        self.assertGreater(float(weights[0, 0]), float(weights[1, 0]))


if __name__ == "__main__":
    unittest.main()
