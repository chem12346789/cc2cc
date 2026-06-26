from timeit import default_timer as timer
import warnings

import numpy as np

import pyscf
import pyscf.dft

from cc2cc.utils.env_var import DATA_TEST_PATH


def _to_numpy(array):
    """Convert cupy/numpy-like arrays to numpy arrays."""
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def diff_rho(mol, dm1_compare1, dm1_compare2, grids):
    """
    Calculate the difference between two density matrices.
    """
    coords = _to_numpy(grids.coords)
    weights = _to_numpy(grids.weights)
    dm1_compare1 = _to_numpy(dm1_compare1)
    dm1_compare2 = _to_numpy(dm1_compare2)
    ao = pyscf.dft.numint.eval_ao(mol, coords, deriv=0)
    if len(np.shape(dm1_compare1)) != len(np.shape(dm1_compare2)):
        raise ValueError("dm1_compare1 and dm1_compare2 must have the same dimension.")
    if len(np.shape(dm1_compare1)) == 3:
        dm1_compare1 = dm1_compare1[0] + dm1_compare1[1]
        dm1_compare2 = dm1_compare2[0] + dm1_compare2[1]
    ddm = dm1_compare1 - dm1_compare2
    drho = pyscf.dft.numint.eval_rho(mol, ao, ddm, xctype="LDA")

    return np.sum(np.abs(drho) * weights)


def diff_I_value(mol, dm1_compare1, dm1_compare2, grids):
    r"""
    Calculate the difference between two density.
    I = \frac{\int |rho1 - rho2|^2 \d r}{\int |rho1|^2 \d r + \int |rho2|^2 \d r}
    """
    coords = _to_numpy(grids.coords)
    weights = _to_numpy(grids.weights)
    dm1_compare1 = _to_numpy(dm1_compare1)
    dm1_compare2 = _to_numpy(dm1_compare2)
    ao = pyscf.dft.numint.eval_ao(mol, coords, deriv=0)
    if len(np.shape(dm1_compare1)) != len(np.shape(dm1_compare2)):
        raise ValueError("dm1_compare1 and dm1_compare2 must have the same dimension.")
    if len(np.shape(dm1_compare1)) == 3:
        dm1_compare1 = dm1_compare1[0] + dm1_compare1[1]
        dm1_compare2 = dm1_compare2[0] + dm1_compare2[1]
    rho1 = pyscf.dft.numint.eval_rho(mol, ao, dm1_compare1, xctype="LDA")
    rho2 = pyscf.dft.numint.eval_rho(mol, ao, dm1_compare2, xctype="LDA")
    drho = rho1 - rho2
    I_value = np.sum(np.abs(drho) ** 2 * weights) / (
        np.sum(np.abs(rho1) ** 2 * weights)
        + np.sum(np.abs(rho2) ** 2 * weights)
    )

    return I_value


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
        dm1_dft = data_frame["dm1_dft"] if "dm1_dft" in data_frame else None
        if f"e_dft-{xc_code_disp}" not in data_frame:
            if mol.spin == 0:
                data_frame_ks = self.test_mol_rks(dm1_dft, xc_code_disp)
            else:
                data_frame_ks = self.test_mol_uks(dm1_dft, xc_code_disp)
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
                data_frame_ks = self.test_mol_rks(dm1_dft, xc_code_disp)
            else:
                data_frame_ks = self.test_mol_uks(dm1_dft, xc_code_disp)
            data_frame.update(data_frame_ks)

        self.dm1_dft = data_frame["dm1_dft"]
        self.grad_dft = data_frame[f"grad_dft-{xc_code_disp}"]
        self.e_dft = data_frame[f"e_dft-{xc_code_disp}"]
        self.dft_dipole = data_frame[f"dft_dipole-{xc_code_disp}"]

        print(f"Data for {name} loaded.")
        if if_update:
            print(f"Data for {name} saved to file.")
            np.savez(path_to_data, **data_frame)

    def test_mol_rks(self, dm1_dft, xc_code_disp):
        """
        Generate 1-RDM, energy, dipole, and gradient for the dft dispersion-corrected RKS molecule.
        """
        time_start = timer()
        mdft = pyscf.scf.RKS(self.mol).density_fit()
        mdft.xc = xc_code_disp
        mdft.verbose = 4
        mdft.grids.level = 4
        mdft.level_shift = 0.1
        if dm1_dft is None:
            mdft.kernel()
        else:
            mdft.kernel(dm0=dm1_dft)
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

    def test_mol_uks(self, dm1_dft, xc_code_disp):
        """
        Generate 1-RDM, energy, dipole, and gradient for the dft dispersion-corrected UKS molecule.
        """
        time_start = timer()
        mdft = pyscf.scf.UKS(self.mol).density_fit()
        mdft.xc = xc_code_disp
        mdft.verbose = 4
        mdft.grids.level = 4
        mdft.level_shift = 0.1
        mdft.kernel(dm0=dm1_dft)
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
