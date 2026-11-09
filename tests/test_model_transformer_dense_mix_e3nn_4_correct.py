import importlib
import io
import unittest

import torch

from cc2cc.utils.helper.runtime.env_var import EDGE_SIZE

MODEL_NAME = "transformer+dense_mix_e3nn_4_correct"


def make_model(model_name=MODEL_NAME, seed=0):
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.random.set_rng_state(generator.get_state())
        module = importlib.import_module(f"cc2cc.utils.model.{model_name}")
        model = module.Model()
    return model.to(device="cpu", dtype=torch.float64).eval()


def make_inputs(batch_size=2, requires_grad=False):
    return torch.linspace(
        0.1,
        0.9,
        batch_size * 4 * EDGE_SIZE**3,
        dtype=torch.float64,
        device="cpu",
    ).reshape(batch_size, 4, EDGE_SIZE**3).requires_grad_(requires_grad)


class TransformerDenseMixE3nnCorrectTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model()

    def test_batched_predictions_match_individual_predictions(self):
        inputs = make_inputs()
        with torch.no_grad():
            outputs = self.model(inputs)
            individual = torch.cat(
                [self.model(sample.unsqueeze(0)) for sample in inputs], dim=0
            )

        self.assertEqual(outputs.shape, (2, 1))
        self.assertEqual(outputs.dtype, torch.float64)
        self.assertEqual(outputs.device.type, "cpu")
        self.assertTrue(torch.isfinite(outputs).all())
        torch.testing.assert_close(outputs, individual)

    def test_input_gradient_matches_directional_finite_difference(self):
        inputs = make_inputs(requires_grad=True)
        direction = torch.linspace(
            -1.0, 1.0, inputs.numel(), dtype=torch.float64, device="cpu"
        ).reshape_as(inputs)
        gradient = torch.autograd.grad(self.model(inputs).sum(), inputs)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum(), 0)

        step = 1e-5
        with torch.no_grad():
            difference = (
                self.model(inputs + step * direction).sum()
                - self.model(inputs - step * direction).sum()
            ) / (2 * step)
        torch.testing.assert_close(
            (gradient * direction).sum(), difference, rtol=1e-5, atol=1e-8
        )

    def test_rotation_preserves_predictions_and_rotates_input_gradients(self):
        grid = make_inputs().reshape(2, 4, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE)
        rotated = grid.rot90(1, dims=(2, 3)).clone().requires_grad_(True)
        grid.requires_grad_(True)

        outputs = self.model(grid.flatten(start_dim=2))
        rotated_outputs = self.model(rotated.flatten(start_dim=2))
        gradient = torch.autograd.grad(outputs.sum(), grid)[0]
        rotated_gradient = torch.autograd.grad(rotated_outputs.sum(), rotated)[0]

        torch.testing.assert_close(rotated_outputs, outputs)
        torch.testing.assert_close(rotated_gradient, gradient.rot90(1, dims=(2, 3)))

    def test_force_like_loss_backpropagates_through_input_gradients(self):
        self.model.train()
        inputs = make_inputs(requires_grad=True)
        density_gradient = torch.autograd.grad(
            self.model(inputs).sum(), inputs, create_graph=True
        )[0]
        response = torch.linspace(
            -1.0, 1.0, inputs.numel() * 3, dtype=torch.float64, device="cpu"
        ).reshape(*inputs.shape, 3)
        force = torch.einsum("bcn,bcnk->k", density_gradient, response)
        self.assertTrue(torch.isfinite(force).all())
        self.assertGreater(force.abs().sum(), 0)
        force.square().sum().backward()

        gradients = [parameter.grad for parameter in self.model.parameters()]
        self.assertTrue(
            all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            )
        )
        self.assertTrue(
            any(
                gradient is not None and torch.count_nonzero(gradient) > 0
                for gradient in gradients
            )
        )

    def test_checkpoint_round_trip_preserves_predictions_and_gradients(self):
        checkpoint_file = io.BytesIO()
        torch.save(
            {"model": MODEL_NAME, "state_dict": self.model.state_dict()},
            checkpoint_file,
        )
        checkpoint_file.seek(0)
        checkpoint = torch.load(
            checkpoint_file, map_location="cpu", weights_only=True
        )
        restored = make_model(checkpoint["model"], seed=1)
        inputs = make_inputs(requires_grad=True)
        expected = self.model(inputs)
        self.assertFalse(torch.allclose(restored(inputs), expected))

        restored.load_state_dict(checkpoint["state_dict"])
        actual = restored(inputs)
        expected_gradient = torch.autograd.grad(expected.sum(), inputs)[0]
        actual_gradient = torch.autograd.grad(actual.sum(), inputs)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)


if __name__ == "__main__":
    unittest.main()
