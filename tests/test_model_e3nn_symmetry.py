import importlib
import unittest

import torch

from cc2cc.utils.env_var import EDGE_SIZE

MODEL_MODULES = (
    "cc2cc.utils.model.transformer+dense_mix_e3nn_4_correct",
    "cc2cc.utils.model.transformer+dense_mix_e3nn_4_correct3",
    "cc2cc.utils.model.transformer+dense_mix_e3nn_4_correct_f",
)


class ModelSymmetryTests(unittest.TestCase):
    def test_octahedral_transformations_preserve_outputs(self):
        grid = torch.arange(
            4 * EDGE_SIZE**3, dtype=torch.float64, device="cpu"
        ).reshape(1, 4, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE)
        grid = grid / grid.numel()
        transformations = (
            ("xy rotation", grid.rot90(1, dims=(2, 3))),
            ("xz rotation", grid.rot90(1, dims=(2, 4))),
            ("axis reflection", grid.flip(2)),
            ("inversion", grid.flip((2, 3, 4))),
        )

        for module_name in MODEL_MODULES:
            model = importlib.import_module(module_name).Model().double().eval()
            with torch.no_grad():
                expected = model(grid.flatten(start_dim=2))
                for transformation, transformed in transformations:
                    with self.subTest(
                        model=module_name, transformation=transformation
                    ):
                        torch.testing.assert_close(
                            model(transformed.flatten(start_dim=2)), expected
                        )


if __name__ == "__main__":
    unittest.main()
