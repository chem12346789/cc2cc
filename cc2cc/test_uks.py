from pathlib import Path

import pandas as pd
import numpy as np

import pyscf
from pyscf.grad import ccsd as ccsd_grad

from cc2cc.utils import DATA_TEST_PATH, AU2KCALMOL, AU2DEBYE
from cc2cc.utils import gen_basis
from cc2cc.utils import Grid


class TEST_DATA:

    def __init__(
        self,
        molecular,
        name="methane",
        basis="sto-3g",
        if_basis_str=False,
        spin=0,
    ):
        self.name = name
        self.basis = basis
        self.if_basis_str = if_basis_str

        rotate(molecular)

        self.mol = pyscf.M(
            atom=molecular,
            basis=gen_basis(
                molecular,
                self.basis,
                self.if_basis_str,
            ),
            verbose=4,
            spin=spin,
        )

    # pylint: disable=W0201
    def test_mol(self):
        """
        Generate 1-RDM.
        """
        # if False:
        if (DATA_TEST_PATH / f"data_{self.name}.npz").exists():
            print(f"Load data from {DATA_TEST_PATH}/data_{self.name}.npz")
            data_saved = np.load(f"{DATA_TEST_PATH}/data_{self.name}.npz")
            self.cc_dipole = data_saved["cc_dipole"]
            self.e_cc = data_saved["e_cc"]
            self.grad_ccsd = data_saved["grad_ccsd"]
            self.dft_dipole = data_saved["dft_dipole"]
            self.e_dft = data_saved["e_dft"]
            self.grad_dft = data_saved["grad_dft"]
        else:
            print(f"Generate data for {self.name}")

            mdft = pyscf.scf.RKS(self.mol)
            mdft.xc = "b3lyp"
            mdft.max_cycle = 250
            mdft.kernel()
            dm1_dft = mdft.make_rdm1(ao_repr=True)
            self.e_dft = mdft.e_tot
            g = mdft.nuc_grad_method()
            self.grad_dft = g.kernel()
            self.dft_dipole = pyscf.scf.hf.dip_moment(
                mol=self.mol,
                dm=dm1_dft,
                unit="A.U.",
            )

            mf = pyscf.scf.RHF(self.mol)
            mf.kernel()
            mycc = pyscf.cc.CCSD(mf)
            mycc.incore_complete = True
            mycc.async_io = False
            mycc.direct = True
            mycc.kernel()
            dm1_cc = mycc.make_rdm1(ao_repr=True)
            self.e_cc = mycc.e_tot
            g = ccsd_grad.Gradients(mycc)
            self.grad_ccsd = g.kernel()
            self.cc_dipole = pyscf.scf.hf.dip_moment(
                mol=self.mol,
                dm=dm1_cc,
                unit="A.U.",
            )

            np.savez_compressed(
                DATA_TEST_PATH / f"data_{self.name}.npz",
                cc_dipole=self.cc_dipole,
                e_cc=self.e_cc,
                grad_ccsd=self.grad_ccsd,
                dft_dipole=self.dft_dipole,
                e_dft=self.e_dft,
                grad_dft=self.grad_dft,
            )


def test_uks(
    args,
    molecular,
    name,
    modeldict,
    df_dict: dict,
    df_dict_path: Path,
):
    """
    Test the model. Restrict Khon-Sham (no spin).
    """
    # 2.0 Prepare
    test_data = TEST_DATA(
        molecular,
        name=name,
        basis=args.basis,
        if_basis_str=args.if_basis_str,
    )
    test_data.test_mol()

    grids = Grid(test_data.mol)
