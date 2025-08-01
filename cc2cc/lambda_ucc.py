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

from cc2cc.ucc import get_dft_energy
from cc2cc.utils import get_veff_modified_uks, diff_rho
from cc2cc.utils import DATA_PATH


def lambda_ucc(mol, grids, name, modeldict, args):
    """
    Generate data for the UCCSD method.
    """
    print(f"Generate data for {name}, spin {mol.spin}")

    mf = pyscf.scf.UHF(mol)
    mf.max_cycle = 200
    mf.kernel()

    mycc = pyscf.cc.UCCSD(mf)
    _, t1, t2 = mycc.kernel()
    if args.cc_triple:
        eris = mycc.ao2mo()
        e3ref = uccsd_t.kernel(mycc, eris, t1, t2)
        l1, l2 = uccsd_t_lambda.kernel(mycc, eris, t1, t2)[1:]
        dm1_cc = uccsd_t_rdm.make_rdm1(mycc, t1, t2, l1, l2, eris=eris, ao_repr=True)
        dm1_cc_mo = uccsd_t_rdm.make_rdm1(
            mycc, t1, t2, l1, l2, eris=eris, ao_repr=False
        )
        d1 = u_gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
        d2 = u_gamma2_intermediates(mycc, t1, t2, l1, l2, eris)
        dm2_cc = uccsd_rdm._make_rdm2(mycc, d1, d2, True, True, ao_repr=True)
        del d1, d2
        e_cc = mycc.e_tot + e3ref
    else:
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        dm1_cc_mo = mycc.make_rdm1(ao_repr=False)
        dm2_cc = mycc.make_rdm2(ao_repr=True)
        e_cc = mycc.e_tot
    dm1_cc = np.array(dm1_cc)
    dm2_cc = np.array(dm2_cc)

    mdft = pyscf.scf.UKS(mol)
    mdft.conv_tol = 1e-6
    mdft.max_cycle = 50
    mdft.xc = "b3lyp"
    mdft.grids = grids
    mdft.verbose = 4
    get_veff_modified_uks(mdft, modeldict, lambda_rho=1, dm_tar=dm1_cc)
    mdft.kernel(mf.make_rdm1())
    dm1_dft = mdft.make_rdm1(ao_repr=True)

    mdft_ene = pyscf.scf.UKS(mol)
    mdft_ene.xc = "b3lyp"
    e_dft = mdft_ene.energy_tot(dm1_dft)

    print(f"{diff_rho(mol, dm1_cc, dm1_dft, grids):.6f} (CCSD vs DFT)")
    cc_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_cc, unit="A.U.")
    dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm1_dft, unit="A.U.")
    print(f"{np.linalg.norm(cc_dipole - dft_dipole)} (CCSD vs DFT)")

    error_energy_dft, exc_cc_grids_dft, rho_cc, rho_dft = get_dft_energy(
        mol,
        grids,
        mf.mo_coeff,
        dm1_dft,
        mdft.mo_coeff,
        e_dft,
        dm1_cc,
        dm1_cc_mo,
        dm2_cc,
        e_cc,
    )

    rho_cube_cc = grids.gen_cube_rho_uks(rho_cc, mdft._numint, dm1_cc)
    rho_cube_dft = grids.gen_cube_rho_uks(rho_dft, mdft._numint, dm1_dft)
    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm1_cc=dm1_cc,
        rho_cube_cc=rho_cube_cc,
        rho_cube_dft=rho_cube_dft,
        weights=grids.weights,
        exc_cc_grids=exc_cc_grids_dft,
        error_energy=error_energy_dft,
        mol=mol.tostring(format="xyz"),
        charge=mol.charge,
        spin=mol.spin,
    )
