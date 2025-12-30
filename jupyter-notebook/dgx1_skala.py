import sys
from itertools import product

import pyscf
from pyscf.cc import ccsd_t, uccsd_t
from pyscf import gto
from skala.pyscf import SkalaKS

sys.path.append("../")
from cc2cc.utils import gen_mole


name_mol_list = [
    "W4_11-p",
    "W4_11-p4",
]
# do basis set extrapolation cc-pVTZ and cc-pVQZ
scf_energies = {}
# basis_iter_list = ["aug-cc-pVTZ", "aug-cc-pVQZ"]
# basis_iter_list = ["cc-pVTZ-F12", "cc-pVQZ-F12"]
basis_iter_list = ["def2-TZVPPD", "def2-QZVPPD"]
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

    if mol.spin == 0:
        ks = SkalaKS(mol, xc="skala")
        ks.kernel()
    else:
        ks = SkalaKS(mol, xc="skala")
        ks.kernel()

    scf_energies[f"{name_mol}_{basis_iter}"] = ks.e_tot
print(scf_energies)
