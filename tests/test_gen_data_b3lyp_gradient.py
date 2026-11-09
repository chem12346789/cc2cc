from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pyscf

from cc2cc.utils.reference.reference_b3lyp import (
    converged_b3lyp,
    b3lyp_finite_difference_gradient,
)
from cc2cc.utils.reference.reference_vv10 import post_vv10_finite_difference_gradient


class TestB3lypGradient(TestCase):
    def test_finite_difference_matches_analytical(self):
        original_threads = pyscf.lib.num_threads()
        pyscf.lib.num_threads(1)
        self.addCleanup(pyscf.lib.num_threads, original_threads)
        for spin, atoms in (
            (0, "H 0 0 0; H 0.2 0 1.4"),
            (1, "He 0 0 0; H 0.3 0 1.5"),
        ):
            with self.subTest(spin=spin):
                mol = pyscf.M(
                    atom=atoms, basis="sto-3g", unit="Bohr", spin=spin, verbose=0
                )
                coords = mol.atom_coords().copy()
                mf = converged_b3lyp(mol)
                analytical = mf.nuc_grad_method().kernel()
                numerical = b3lyp_finite_difference_gradient(mol)
                self.assertEqual(numerical.shape, (mol.natm, 3))
                self.assertEqual(numerical.dtype, np.float64)
                np.testing.assert_allclose(analytical, numerical, rtol=0, atol=5e-5)
                np.testing.assert_array_equal(mol.atom_coords(), coords)

    def test_post_vv10_reference_preserves_grid_level_and_step(self):
        mol = pyscf.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
        with patch(
            "cc2cc.utils.reference.reference_vv10.post_vv10_total_energy",
            side_effect=lambda displaced, level: np.sum(displaced.atom_coords() ** 2)
            + level,
        ) as energy:
            result = post_vv10_finite_difference_gradient(mol, 2, 0.002)
        np.testing.assert_array_equal(result, np.zeros((1, 3)))
        self.assertEqual(energy.call_count, 6)
        self.assertTrue(all(call.args[1] == 2 for call in energy.call_args_list))
