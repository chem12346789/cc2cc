import sys
from itertools import product
import os

os.environ["PYSCF_MAX_MEMORY"] = "25000"
os.environ["OMP_NUM_THREADS"] = "64"

import pyscf
from pyscf.cc import ccsd_t, uccsd_t
from pyscf import gto
from pyscf import lib

sys.path.append("../")
from cc2cc.utils import gen_mole

lib.num_threads(64)

name_mol_list = [
    "W4_11-p",
    "W4_11-p4",
    # "W4_11-alh3",
    # "W4_11-al",
    # "W4_11-h",
]
# do basis set extrapolation cc-pVTZ and cc-pVQZ
scf_energies = {}
ccsd_t_corr_energies = {}
# basis_iter_list = ["aug-cc-pVTZ", "aug-cc-pVQZ"]
basis_iter_list = ["aug-cc-pV(Q+d)Z", "aug-cc-pV(5+d)Z"]
# basis_iter_list = ["aug-cc-pCVTZ", "aug-cc-pCVQZ"]

for basis_iter, name_mol in product(range(len(basis_iter_list)), name_mol_list):
    mol = gen_mole(
        name_mol,
        basis_iter_list[basis_iter],
        ma_basis=False,
        dataset_name="gmtkn-def2",
        if_rotate=True,
        if_rotate_random=True,
        solve_symmetry=True,
        verbose=1,
    )
    mol.max_memory = 25000

    if mol.spin == 0:
        mf = pyscf.scf.RHF(mol)
        mf.max_cycle = 50
        mf.kernel()

        mycc = pyscf.cc.CCSD(mf)
        mycc.set_frozen()
        mycc.direct = True
        mycc.verbose = 4

        _, t1, t2 = mycc.kernel()
        eris = mycc.ao2mo()
        e3ref = ccsd_t.kernel(mycc, eris, t1, t2)
    else:
        mf = pyscf.scf.UHF(mol)
        mf.max_cycle = 50
        mf.kernel()

        mycc = pyscf.cc.UCCSD(mf)
        mycc.set_frozen()
        mycc.direct = True
        mycc.verbose = 4

        _, t1, t2 = mycc.kernel()
        eris = mycc.ao2mo()
        e3ref = uccsd_t.kernel(mycc, eris, t1, t2)

    scf_energies[f"{name_mol}_{basis_iter}"] = mf.e_tot
    ccsd_t_corr_energies[f"{name_mol}_{basis_iter}"] = mycc.e_tot - mf.e_tot
    # ccsd_t_corr_energies[f"{name_mol}_{basis_iter}"] = mycc.e_tot + e3ref - mf.e_tot
print(scf_energies, ccsd_t_corr_energies)
