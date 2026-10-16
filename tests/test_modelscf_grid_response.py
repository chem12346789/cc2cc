import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyscf
import torch

from cc2cc.utils.Grids import Grid
from cc2cc.utils.env_var import EDGE_SIZE
from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.modelscf_rks import get_veff_grad_modified_rks
from cc2cc.utils.modelscf_uks import get_veff_grad_modified_uks


class ToyFunctional:
    dtype = torch.float64
    device = "cpu"
    eval_xc_eff = ModelClass.eval_xc_eff

    def __init__(self, cube_size, before_weight=False):
        self.model = SimpleNamespace(cube_size=cube_size)
        self.before_weight = before_weight

    def model_output(self, inputs, weights):
        if self.before_weight:
            inputs = inputs * weights[:, :, None]
        energy = (inputs ** 2).sum(dim=(1, 2))[:, None] * 0.01
        return inputs, energy if self.before_weight else energy * weights

    def get_b3lyp_ene(self, inputs):
        return inputs[:, 0, 0] * 0.2


class TestModelGridResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = pyscf.lib.num_threads()
        pyscf.lib.num_threads(1)

    @classmethod
    def tearDownClass(cls):
        pyscf.lib.num_threads(cls.threads)

    def test_b3lyp_limit_matches_pyscf_in_both_modes(self):
        for spin in (0, 1):
            with self.subTest(spin=spin):
                mol = pyscf.M(
                    atom="He 0 0 0; H 0.3 0 1.5", spin=spin, charge=1 - spin,
                    basis="sto-3g", unit="Bohr", verbose=0,
                )
                mf = (pyscf.dft.RKS(mol) if spin == 0
                      else pyscf.dft.UKS(mol)).density_fit()
                mf.xc = "b3lyp"
                mf.grids = Grid(mol, 0, cube_type="center", cube_size=1)
                mf.conv_tol = 1e-12
                mf.kernel()
                self.assertTrue(mf.converged)
                functional = ToyFunctional(1)
                functional.model_output = lambda inputs, weights: (
                    inputs, inputs[:, 0, :1] * 0
                )
                functional.get_b3lyp_ene = lambda inputs: (
                    inputs[:, :4, 0]
                    * ModelClass.b3lyp_weights(inputs)
                ).sum(-1)
                for enabled in (False, True):
                    reference = mf.Gradients()
                    reference.grid_response = enabled
                    expected = reference.kernel()
                    custom = mf.Gradients()
                    custom.grids = mf.grids
                    custom.grid_response = enabled
                    modify = (get_veff_grad_modified_rks if spin == 0
                              else get_veff_grad_modified_uks)
                    modify(custom, functional)
                    np.testing.assert_allclose(
                        custom.kernel(), expected, rtol=0, atol=1e-10
                    )

    def test_grid_response_against_displaced_xc_energy(self):
        for spin, cube_type, cube_size, before_weight in (
            (0, "center", 1, False),
            (1, "center", 1, False),
            (0, "cube", EDGE_SIZE ** 3, False),
            (1, "cube", EDGE_SIZE ** 3, False),
            (0, "cube", EDGE_SIZE ** 3, True),
            (1, "cube", EDGE_SIZE ** 3, True),
        ):
            with self.subTest(spin=spin, cube=cube_type, weighted=before_weight):
                mol = pyscf.M(
                    atom="He 0 0 0; H 0.3 0 1.5", spin=spin,
                    charge=1 - spin, basis="sto-3g", unit="Bohr", verbose=0,
                )
                mf = pyscf.dft.RKS(mol) if spin == 0 else pyscf.dft.UKS(mol)
                mf.xc = "b88"
                dm = mf.get_init_guess()
                functional = ToyFunctional(cube_size, before_weight)
                grids = Grid(mol, 0, cube_type=cube_type, cube_size=cube_size)
                grad = mf.Gradients()
                grad.grids = grids
                grad.auxbasis_response = False
                grad.get_j = lambda *args: np.zeros(
                    (3, mol.nao, mol.nao) if spin == 0
                    else (2, 3, mol.nao, mol.nao)
                )
                modify = (get_veff_grad_modified_rks if spin == 0
                          else get_veff_grad_modified_uks)
                original_grad_elec = grad.grad_elec.__func__
                modify(grad, functional)
                self.assertIs(grad.grad_elec.__func__, original_grad_elec)
                self.assertFalse(grad.grid_response)
                with patch.object(functional, "eval_xc_eff",
                                  wraps=functional.eval_xc_eff) as evaluate:
                    default = grad.get_veff(mol, dm)
                    self.assertTrue(all(not c.kwargs for c in evaluate.call_args_list))
                if spin == 0:
                    self.assertIsNone(default.exc1_grid)
                else:
                    np.testing.assert_array_equal(default.exc1_grid, 0)
                grad.grid_response = True
                vxc = grad.get_veff(mol, dm)
                analytical = vxc.exc1_grid.copy()
                for atom, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
                    if spin == 0:
                        analytical[atom] += 2 * np.einsum(
                            "xij,ij->x", vxc[:, p0:p1], dm[p0:p1]
                        )
                    else:
                        analytical[atom] += 2 * np.einsum(
                            "sxij,sij->x", vxc[:, :, p0:p1], dm[:, p0:p1]
                        )

                def energy(displaced):
                    grid = Grid(displaced, 0, cube_type=cube_type, cube_size=cube_size)
                    cube = grid.gen_cube(displaced, dm, grid.coords)
                    evaluate = (cube.gen_cube_rho_rks if spin == 0
                                else cube.gen_cube_rho_uks)
                    rho = evaluate(mf._numint, dm)[0]
                    return functional.eval_xc_eff(rho, grid.weights)[0].sum()

                numerical = np.empty((mol.natm, 3))
                coords = mol.atom_coords()
                step = 1e-4
                for atom in range(mol.natm):
                    for axis in range(3):
                        values = []
                        for sign in (1, -1):
                            displaced_coords = coords.copy()
                            displaced_coords[atom, axis] += sign * step
                            displaced = mol.copy().set_geom_(
                                displaced_coords, unit="Bohr"
                            )
                            values.append(energy(displaced))
                        numerical[atom, axis] = (values[0] - values[1]) / (2 * step)
                np.testing.assert_allclose(analytical, numerical, rtol=0, atol=2e-6)
                np.testing.assert_allclose(analytical.sum(axis=0), 0, atol=1e-10)
                np.testing.assert_array_equal(mol.atom_coords(), coords)
