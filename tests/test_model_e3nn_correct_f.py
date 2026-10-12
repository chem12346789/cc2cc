import unittest

import torch

from cc2cc.utils.env_var import CUBE_MIDDLE, EDGE_LEN, EDGE_SIZE
from cc2cc.utils.model.model_utils.model_e3nn_correct_f import E3nn


class DistanceAndAngleAwareE3nnTests(unittest.TestCase):
    def make_model(self, lmax=0):
        model = E3nn("cube", EDGE_SIZE**3, 2, lmax)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.5)
        return model

    def point_index(self, x, y, z):
        return (
            ((CUBE_MIDDLE + x) * EDGE_SIZE + CUBE_MIDDLE + y) * EDGE_SIZE
            + CUBE_MIDDLE
            + z
        )

    def test_normalized_radii_and_shapes(self):
        for lmax in (0, 1, 2):
            model = self.make_model(lmax)
            torch.testing.assert_close(
                model.radii,
                torch.linalg.vector_norm(
                    model.edge_vec / EDGE_LEN, dim=-1, keepdim=True
                ),
            )
            self.assertTrue(torch.isfinite(model.sh).all())
            for shape in (
                (EDGE_SIZE**3, 2),
                (2, EDGE_SIZE**3, 2),
                (2, 3, EDGE_SIZE**3, 2),
            ):
                with self.subTest(lmax=lmax, shape=shape):
                    output = model(
                        torch.ones(shape, dtype=torch.float64, device="cpu")
                    )
                    self.assertEqual(output.shape, shape[:-2] + (1, EDGE_SIZE**3))
                    self.assertEqual(output.dtype, torch.float64)
                    self.assertEqual(output.device.type, "cpu")
                    self.assertTrue(torch.isfinite(output).all())

    def test_distance_dependence_at_lmax_zero(self):
        model = self.make_model()
        at_center = torch.zeros(
            (1, EDGE_SIZE**3, 2), dtype=torch.float64, device="cpu"
        )
        at_neighbor = torch.zeros_like(at_center)
        at_center[:, self.point_index(0, 0, 0)] = 1
        at_neighbor[:, self.point_index(1, 0, 0)] = 1
        torch.testing.assert_close(at_center.mean(dim=-2), at_neighbor.mean(dim=-2))
        self.assertGreater(
            (model(at_center) - model(at_neighbor)).abs().max(), 1e-8
        )

    def test_angles_distinguish_equal_radius_inputs(self):
        adjacent = torch.zeros(
            (1, EDGE_SIZE**3, 2), dtype=torch.float64, device="cpu"
        )
        opposite = torch.zeros_like(adjacent)
        adjacent[:, self.point_index(1, 0, 0)] = 1
        adjacent[:, self.point_index(0, 1, 0)] = 1
        opposite[:, self.point_index(1, 0, 0)] = 1
        opposite[:, self.point_index(-1, 0, 0)] = 1
        torch.testing.assert_close(adjacent.mean(dim=-2), opposite.mean(dim=-2))

        scalar_model = self.make_model(0)
        torch.testing.assert_close(scalar_model(adjacent), scalar_model(opposite))
        angular_model = self.make_model(1)
        self.assertGreater(
            (angular_model(adjacent) - angular_model(opposite)).abs().max(), 1e-8
        )
        with torch.no_grad():
            angular_model.mixer[0].weight[:, :angular_model.cube_size].zero_()
        torch.testing.assert_close(angular_model(adjacent), angular_model(opposite))

    def test_nonlinear_head_merges_scalar_features(self):
        model = self.make_model(2)
        self.assertIsInstance(model.mixer, torch.nn.Sequential)
        self.assertIsInstance(model.mixer[1], torch.nn.SiLU)
        self.assertEqual(model.mixer[0].in_features, 2 * model.cube_size)
        self.assertEqual(model.mixer[0].out_features, model.cube_size)
        self.assertEqual(model.mixer[2].out_features, model.cube_size)
        inputs = torch.ones(
            (1, EDGE_SIZE**3, 2), dtype=torch.float64, device="cpu"
        )
        hidden = model.tp1(inputs, model.sh, model.radial(model.radii))
        hidden = hidden.mean(dim=-2, keepdim=True)
        features = torch.cat(
            [model.tensor_square(hidden), model.readout(hidden)], dim=-1
        )
        expected = model.mixer[2](
            torch.nn.functional.silu(model.mixer[0](features))
        )
        torch.testing.assert_close(model(inputs), expected)
        with torch.no_grad():
            model.mixer[0].weight[:, model.cube_size:].zero_()
        self.assertGreater((model(inputs) - expected).abs().max(), 1e-8)

    def test_cube_rotations_and_inversion_preserve_outputs(self):
        model = self.make_model(2)
        grid = torch.arange(
            EDGE_SIZE**3 * 2, dtype=torch.float64, device="cpu"
        ).reshape(EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 2)
        inputs = grid.reshape(1, EDGE_SIZE**3, 2)
        expected = model(inputs)
        for transformed in (
            grid.rot90(1, dims=(0, 1)),
            grid.transpose(1, 2),
            grid.flip((0, 1, 2)),
        ):
            torch.testing.assert_close(
                model(transformed.reshape(1, EDGE_SIZE**3, 2)), expected
            )

    def test_gradients_reach_input_and_both_readout_paths(self):
        model = self.make_model(2)
        inputs = torch.arange(
            2 * EDGE_SIZE**3 * 2, dtype=torch.float64, device="cpu"
        ).reshape(2, EDGE_SIZE**3, 2).requires_grad_()
        model(inputs).square().sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum(), 0)
        for name, module in (
            ("radial", model.radial),
            ("linear", model.readout),
            ("quadratic", model.tensor_square),
            ("mixer", model.mixer),
        ):
            for parameter in module.parameters():
                with self.subTest(module=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
