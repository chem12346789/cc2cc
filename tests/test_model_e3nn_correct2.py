import unittest

import torch

from cc2cc.utils.env_var import CUBE_MIDDLE, EDGE_SIZE
from cc2cc.utils.model.model_utils.model_e3nn_correct2 import E3nn


class DistanceAwareE3nnTests(unittest.TestCase):
    def make_model(self, lmax=0):
        model = E3nn("cube", EDGE_SIZE**3, 2, lmax)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.5)
        return model

    def test_shapes_dtype_and_finite_center(self):
        for lmax in (0, 2):
            model = self.make_model(lmax)
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
            self.assertEqual(model.radii.min(), 0)
            self.assertTrue(torch.isfinite(model.sh).all())

    def test_distinguishes_radii_with_equal_input_mean(self):
        model = self.make_model()
        center = (CUBE_MIDDLE * EDGE_SIZE + CUBE_MIDDLE) * EDGE_SIZE + CUBE_MIDDLE
        at_center = torch.zeros(
            (1, EDGE_SIZE**3, 2), dtype=torch.float64, device="cpu"
        )
        at_corner = torch.zeros_like(at_center)
        at_center[:, center, :] = 1
        at_corner[:, 0, :] = 1
        torch.testing.assert_close(at_center.mean(dim=-2), at_corner.mean(dim=-2))
        self.assertGreater(
            (model(at_center) - model(at_corner)).abs().max(), 1e-8
        )

    def test_octahedral_symmetry(self):
        model = self.make_model(lmax=2)
        field = torch.arange(
            EDGE_SIZE**3 * 2, dtype=torch.float64, device="cpu"
        ).reshape(1, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 2)
        expected = model(field.reshape(1, EDGE_SIZE**3, 2))
        for transformed in (field.flip(1), field.transpose(1, 2)):
            torch.testing.assert_close(
                model(transformed.reshape(1, EDGE_SIZE**3, 2)), expected
            )

    def test_input_and_radial_gradients(self):
        model = self.make_model()
        inputs = torch.ones(
            (1, EDGE_SIZE**3, 2),
            dtype=torch.float64,
            device="cpu",
            requires_grad=True,
        )
        self.assertTrue(torch.autograd.gradcheck(model, (inputs,)))
        model(inputs).sum().backward()
        for tensor in (inputs, *model.parameters()):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum(), 0)

    def test_state_dict_round_trip(self):
        model = self.make_model()
        restored = E3nn("cube", EDGE_SIZE**3, 2, 0)
        restored.load_state_dict(model.state_dict())
        inputs = torch.ones(
            (1, EDGE_SIZE**3, 2), dtype=torch.float64, device="cpu"
        )
        torch.testing.assert_close(restored(inputs), model(inputs))

    def test_rejects_unsupported_geometry(self):
        with self.assertRaisesRegex(NotImplementedError, "Only cube"):
            E3nn("sphere", EDGE_SIZE**3, 2, 0)


if __name__ == "__main__":
    unittest.main()
