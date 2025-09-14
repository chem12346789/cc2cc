from timeit import default_timer as timer
import os
import json
import warnings

import numpy as np

import pyscf

from cc2cc.utils.env_var import DATA_TEST_PATH


class TestDataDFT:
    """
    Class to generate and store test data for DFT calculations.
    It generates 1-RDM, energy, dipole, and gradient for a given molecule.
    The data is saved in a compressed npz file for later use.
    Note:
        1) If the data already exists, it will be loaded instead of recomputed.
        2) If the molecule coordinates are different from the saved data, it will raise an error.
        3) If disp is not None, it will generate data for the dispersion-corrected DFT calculation (Will store the data in the same file).
    Args:
        mol (pyscf.Mole): The molecule object.
        name (str): The name of the molecule, used for saving/loading data.
        xc_code (str): The exchange-correlation functional code for DFT calculations.
        disp (str or None): Dispersion correction method, if any. Default is None.
    Raises:
        ValueError: If the molecule coordinates are different from the saved data.
    """

    def __init__(
        self,
        mol: pyscf.M,
        name: str,
        xc_code: str,
        disp: str,
    ) -> None:
        self.mol = mol
        xc_code_disp = xc_code if disp is None else f"{xc_code}-{disp}"
        print(f"Testing DFT {xc_code_disp} for {name}")
        path_to_data = DATA_TEST_PATH / f"{name}_cc.npz"

        if (path_to_data).exists():
            print(f"Data for {name} loaded from file.")
            data_frame = dict(np.load(path_to_data, allow_pickle=True).items())
        else:
            data_frame = {"mol_corr": mol.atom_coords()}

        if_update = False
        if f"e_dft-{xc_code_disp}" not in data_frame:
            if mol.spin == 0:
                data_frame_ks = self.test_mol_rks(xc_code_disp)
            else:
                data_frame_ks = self.test_mol_uks(xc_code_disp)
            data_frame.update(data_frame_ks)
            if_update = True

        mol_corr = data_frame["mol_corr"]
        if np.linalg.norm(mol.atom_coords() - mol_corr, ord=1) > 1e-6:
            print("Molecule coordinates are different.")
            warnings.warn(
                f"Coordinates of {name} are different from the saved data. "
                "Please check the coordinates or regenerate the data."
            )
            if mol.spin == 0:
                data_frame_ks = self.test_mol_rks(xc_code_disp)
            else:
                data_frame_ks = self.test_mol_uks(xc_code_disp)
            data_frame.update(data_frame_ks)

        self.dm1_dft = data_frame["dm1_dft"]
        self.grad_dft = data_frame[f"grad_dft-{xc_code_disp}"]
        self.e_dft = data_frame[f"e_dft-{xc_code_disp}"]
        self.dft_dipole = data_frame[f"dft_dipole-{xc_code_disp}"]

        print(f"Data for {name} loaded.")
        if if_update:
            print(f"Data for {name} saved to file.")
            np.savez_compressed(path_to_data, **data_frame)

    def test_mol_rks(self, xc_code_disp):
        """
        Generate 1-RDM, energy, dipole, and gradient for the dft dispersion-corrected RKS molecule.
        """
        time_start = timer()
        mdft = pyscf.scf.RKS(self.mol)
        mdft.xc = xc_code_disp
        mdft.verbose = 4
        mdft.grids.level = 4
        mdft.level_shift = 0.1
        mdft.kernel()
        if mdft.converged is False:
            raise ValueError("RKS not converged.")
        dm1_dft = mdft.make_rdm1(ao_repr=True)
        e_dft = mdft.e_tot
        dft_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=dm1_dft,
            unit="A.U.",
        )
        g = mdft.Gradients()
        grad_dft = g.kernel()
        time_dft = timer() - time_start

        dict_ = {
            f"e_dft-{xc_code_disp}": e_dft,
            f"dft_dipole-{xc_code_disp}": dft_dipole,
            f"time_dft-{xc_code_disp}": time_dft,
            f"grad_dft-{xc_code_disp}": grad_dft,
        }
        if xc_code_disp == "b3lyp":
            dict_.update({"dm1_dft": dm1_dft})
        return dict_

    def test_mol_uks(self, xc_code_disp):
        """
        Generate 1-RDM, energy, dipole, and gradient for the dft dispersion-corrected UKS molecule.
        """
        time_start = timer()
        mdft = pyscf.scf.UKS(self.mol)
        mdft.xc = xc_code_disp
        mdft.verbose = 4
        mdft.grids.level = 4
        mdft.level_shift = 0.1
        mdft.kernel()
        if mdft.converged is False:
            raise ValueError("UKS not converged.")
        dm1_dft = mdft.make_rdm1(ao_repr=True)
        e_dft = mdft.e_tot
        dft_dipole = pyscf.scf.hf.dip_moment(
            mol=self.mol,
            dm=dm1_dft,
            unit="A.U.",
        )
        g = mdft.Gradients()
        grad_dft = g.kernel()
        time_dft = timer() - time_start

        dict_ = {
            f"e_dft-{xc_code_disp}": e_dft,
            f"dft_dipole-{xc_code_disp}": dft_dipole,
            f"time_dft-{xc_code_disp}": time_dft,
            f"grad_dft-{xc_code_disp}": grad_dft,
        }
        if xc_code_disp == "b3lyp":
            dict_.update({"dm1_dft": dm1_dft})
        return dict_
