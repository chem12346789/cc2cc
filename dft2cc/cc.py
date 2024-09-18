from timeit import default_timer as timer
import numpy as np
import pyscf
from pyscf.grad import ccsd as ccsd_grad

import opt_einsum as oe

from dft2cc.utils import gen_basis, process_input, Grid
from dft2cc.utils import DATA_PATH

ORIENTATION_NUMBER_DICT = {"x": 0, "y": 1, "z": 2}


def cc(molecular, name, args):
    """
    Generate data for the CCSD method. (Restrict scenario to spin 0).
    """
    mol = pyscf.M(
        atom=molecular,
        basis=gen_basis(
            molecular,
            args.basis,
            args.if_basis_str,
        ),
        spin=0,
    )

    print(mol.atom)
    print(f"Generate data for {name}")

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.incore_complete = True
    mycc.async_io = False
    mycc.direct = True
    mycc.kernel()
    dm1_cc = mycc.make_rdm1(ao_repr=True)
    e_cc = mycc.e_tot
    g = ccsd_grad.Gradients(mycc)
    grad_cc = g.kernel()

    mdft = pyscf.scf.RKS(mol)
    mdft.xc = "b3lyp"
    mdft.kernel()
    dm1_dft = mdft.make_rdm1(ao_repr=True)
    g = mdft.nuc_grad_method()
    grad_dft = g.kernel()

    dm1_cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc)
    dm1_dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft)
    error_dipole = (dm1_cc_dipole - dm1_dft_dipole) / 2.541746

    grids = Grid(mol, level=1, period=2)
    ao_1 = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)
    dft_r_3 = pyscf.dft.numint.eval_rho(mol, ao_1, dm1_dft, xctype="GGA")
    data_grids_norm = process_input(dft_r_3, grids)

    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm_cc=dm1_cc,
        rho_inv_4_norm=data_grids_norm,
        error_energy=e_cc - mdft.e_tot,
        error_grad=grad_cc - grad_dft,
        error_dipole=error_dipole,
    )
