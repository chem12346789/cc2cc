# pylint: disable=W0212

import numpy as np
import pyscf

# from pyscf.grad import ccsd as ccsd_grad
import opt_einsum as oe

from pyscf.cc import uccsd_t_lambda
from pyscf.cc import uccsd_t_rdm
from pyscf.cc import uccsd_t
from pyscf.cc import uccsd_rdm
from pyscf.cc.uccsd_t_rdm import _gamma1_intermediates as u_gamma1_intermediates
from pyscf.cc.uccsd_t_rdm import _gamma2_intermediates as u_gamma2_intermediates

from cc2cc.utils import DATA_PATH, AU2KCALMOL


def ucc_append(mol, grids, name, cc_triple=False):
    """
    Append data for the UCCSD method.
    """

    print(f"Append data for {name}, spin {mol.spin}")

    mf = pyscf.scf.UHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    if mf.converged is False:
        raise ValueError("UHF not converged.")
    mdft = pyscf.scf.UKS(mol)
    mdft.max_cycle = 200
    mdft.xc = "b3lyp"
    mdft.kernel(mf.make_rdm1())
    if mdft.converged is False:
        raise ValueError("UKS not converged.")
    dm1_dft = mdft.make_rdm1(ao_repr=True)

    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)

    rho_dft = [
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[0], xctype="GGA"),
        pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[1], xctype="GGA"),
    ]
    rho_cube_dft = grids.gen_cube_rho_uks(mol, dm1_dft, rho_dft, ni=mdft._numint)

    saved_data = np.load(DATA_PATH / f"data_{name}.npz")
    e_cc = saved_data["e_cc"]
    dm_cc = saved_data["dm_cc"]
    rho_cube = saved_data["rho_cube"]
    weights = saved_data["weights"]
    exc_cc_grids = saved_data["exc_cc_grids"]
    error_energy = saved_data["error_energy"]

    if (np.linalg.norm(grids.weights - weights)) > 1e-6:
        raise ValueError(
            f"Grids weights do not match for {name}. "
            "Please regenerate the grids or use the correct grids."
        )

    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm_cc=dm_cc,
        rho_cube_cc=rho_cube,
        rho_cube_dft=rho_cube_dft,
        weights=grids.weights,
        exc_cc_grids=exc_cc_grids,
        error_energy=error_energy,
        mol=mol.tostring(format="xyz"),
        charge=mol.charge,
        spin=mol.spin,
    )
