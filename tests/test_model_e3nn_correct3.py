import importlib
import unittest

import torch

from cc2cc.utils.env_var import EDGE_SIZE
from cc2cc.utils.model.model_utils.model_e3nn_correct3 import E3nn


class NormalizedDistanceE3nnTests(unittest.TestCase):
    def test_radii_are_normalized_by_grid_spacing(self):
        model = E3nn("cube", EDGE_SIZE**3, 2, 0)
        expected = torch.tensor(
            [0, 1, 2**0.5, 3**0.5], dtype=torch.float64, device="cpu"
        )
        torch.testing.assert_close(model.radii.unique(), expected)

    def test_correct3_architecture_uses_normalized_module(self):
        module = importlib.import_module(
            "cc2cc.utils.model.transformer+dense_mix_e3nn_4_correct3"
        )
        model = module.Model()
        model.double()
        self.assertIsInstance(model.conv1, E3nn)
        output = model(
            torch.ones((1, 4, EDGE_SIZE**3), dtype=torch.float64, device="cpu")
        )
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(torch.isfinite(output).all())
