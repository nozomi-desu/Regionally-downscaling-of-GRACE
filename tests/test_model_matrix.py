from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.train.losses import build_total_loss
from scripts.train.model import (
    BaselineUNet,
    ScaleSeparationUNet,
    VariationalUNet,
    build_model_from_config,
)
from utils.mass_closure import latitude_area_weights, masked_pool2d_mean


class ModelMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.batch_size = 2
        self.channels = 13
        self.height = 24
        self.width = 24
        self.factor = 6
        self.inputs = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.mask = torch.ones(self.batch_size, 1, self.height, self.width)
        self.mask[:, :, :2, :2] = 0.0
        lat = torch.linspace(-70.0, 70.0, self.height)
        area = latitude_area_weights(lat, width=self.width)
        self.area = area.reshape(1, 1, self.height, self.width).expand(self.batch_size, -1, -1, -1)
        self.target_coarse = torch.randn(
            self.batch_size,
            1,
            self.height // self.factor,
            self.width // self.factor,
        )
        self.coarse_context = self.target_coarse.repeat_interleave(self.factor, -2).repeat_interleave(
            self.factor, -1
        )

    def test_factory_builds_registered_architectures(self) -> None:
        common = {"base_channels": 4, "dropout": 0.0, "residual_to_coarse": True}
        self.assertIsInstance(
            build_model_from_config(self.channels, {**common, "architecture": "baseline_unet"}),
            BaselineUNet,
        )
        self.assertIsInstance(
            build_model_from_config(self.channels, {**common, "architecture": "variational_unet"}),
            VariationalUNet,
        )
        self.assertIsInstance(
            build_model_from_config(
                self.channels,
                {
                    **common,
                    "architecture": "scale_separation_unet",
                    "coarse_factor": self.factor,
                    "gate_mode": "learned",
                },
            ),
            ScaleSeparationUNet,
        )

    def test_scale_separation_without_gate_has_zero_coarse_residual(self) -> None:
        model = ScaleSeparationUNet(
            self.channels,
            base_channels=4,
            coarse_factor=self.factor,
            gate_mode="none",
        )
        prediction = model(
            self.inputs,
            coarse_context=self.coarse_context,
            valid_mask=self.mask,
            area_weight=self.area,
        )
        pooled, valid = masked_pool2d_mean(
            prediction,
            self.mask * self.area,
            kernel_size=self.factor,
            eps=1e-12,
        )
        np.testing.assert_allclose(
            pooled.detach().numpy()[valid.detach().numpy().astype(bool)],
            self.target_coarse.numpy()[valid.detach().numpy().astype(bool)],
            rtol=0.0,
            atol=2e-5,
        )

    def test_learned_gate_is_bounded_and_trainable(self) -> None:
        model = ScaleSeparationUNet(
            self.channels,
            base_channels=4,
            coarse_factor=self.factor,
            gate_mode="learned",
            gate_smoothing_kernel=7,
        )
        prediction = model(
            self.inputs,
            coarse_context=self.coarse_context,
            valid_mask=self.mask,
            area_weight=self.area,
        )
        self.assertIsNotNone(model.last_gate)
        assert model.last_gate is not None
        self.assertGreaterEqual(float(model.last_gate.min().detach()), 0.0)
        self.assertLessEqual(float(model.last_gate.max().detach()), 1.0)
        prediction.square().mean().backward()
        self.assertIsNotNone(model.gate_head.weight.grad)
        self.assertTrue(torch.isfinite(model.gate_head.weight.grad).all())

    def test_variational_model_exposes_kl_and_uz_loss(self) -> None:
        model = VariationalUNet(self.channels, base_channels=4)
        prediction = model(self.inputs, coarse_context=self.coarse_context)
        self.assertIsNotNone(model.last_kl_loss)
        assert model.last_kl_loss is not None
        batch = {
            "target_fine": self.coarse_context,
            "target_coarse": self.target_coarse,
            "wghm_fine": torch.randn_like(prediction),
            "valid_mask": self.mask,
            "model_kl_loss": model.last_kl_loss,
        }
        loss, components = build_total_loss(
            prediction,
            batch,
            {
                "mode": "uz_dynamic",
                "coarse_factor": self.factor,
                "kl_weight": 1.0,
                "uz_gradient_weight": 1.0,
            },
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(components["dynamic_lambda"]), 0.0)
        self.assertLessEqual(float(components["dynamic_lambda"]), 1.0)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
