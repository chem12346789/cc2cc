import basis_set_exchange
from pyscf import gto
from pyscf.gto.basis.bse import _ecp_basis, get_basis

from cc2cc.utils.helper.chemistry.addon_basis import addon_basis

def_ecp_list = [
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
]
aug_cc_atom_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
]
aug_cc_pp_atom_list = def_ecp_list.copy()
aug_cc_pp_atom_list.remove("Rb")
aug_cc_pp_atom_list.remove("Sr")
aug_cc_pp_atom_list.remove("Cs")
aug_cc_pp_atom_list.remove("Ba")
aug_cc_pp_atom_list.remove("La")
aug_cc_atom_list.remove("K")
aug_cc_atom_list.remove("Ca")
special_atom_list = ["K", "Ca"]
cc_atom_list = aug_cc_atom_list.copy()
cc_pp_atom_list = aug_cc_pp_atom_list.copy()


def get_ecp(name, elements):
    """
    Obtain the effective core potential (ECP) from Basis Set Exchange.

    Args:
        name : str
            Name of the basis set, case insensitive.
        elements : str, int or list

    Returns:
        A dict of ECP data for the specified elements.
    """
    basis = basis_set_exchange.api.get_basis(name, elements)
    return _ecp_basis(basis)


def gen_basis(basis: str, atom_list: list):
    """Generate the basis set definition for the requested atoms."""
    if "def2" in basis:
        return basis
    if "cc-" in basis:
        dict_ = {}
        for atom in atom_list:
            if atom in dict_:
                continue
            if "aug-" in basis:
                if atom in aug_cc_atom_list:
                    dict_.update(get_basis(basis, atom))
                if atom in aug_cc_pp_atom_list:
                    dict_.update(get_basis(basis + "-PP", atom))
                if atom in special_atom_list:
                    dict_.update(addon_basis[atom][basis])
            else:
                if atom in cc_atom_list:
                    dict_.update(get_basis(basis, atom))
                if atom in cc_pp_atom_list:
                    dict_.update(get_basis(basis + "-PP", atom))
                if atom in special_atom_list:
                    dict_.update(addon_basis[atom][basis])
        return dict_


def gmtkn55_qzvp_basis(atom_list: list, subset: str):
    basis = {}
    augmented_basis = False
    for atom in dict.fromkeys(atom_list):
        if subset in ("G21EA", "AHB21", "IL16"):
            angular_momenta = (0,) if atom == "H" else (0, 1)
        elif subset == "WATER27" and atom == "O":
            angular_momenta = (0, 1)
        else:
            basis[atom] = "def2-qzvp"
            continue
        augmented_basis = True
        shells = gto.basis.load("def2-qzvp", atom)
        augmented = gto.basis.load("aug-cc-pvqz", atom)
        unaugmented = gto.basis.load("cc-pvqz", atom)
        for angular_momentum in angular_momenta:
            plain_exponents = {
                row[0]
                for shell in unaugmented if shell[0] == angular_momentum
                for row in shell[1:]
                if len(row) == 2 and row[1] == 1.0
            }
            diffuse = [
                shell for shell in augmented
                if shell[0] == angular_momentum and len(shell) == 2
                and len(shell[1]) == 2 and shell[1][1] == 1.0
                and shell[1][0] not in plain_exponents
            ]
            if len(diffuse) != 1:
                raise ValueError(
                    f"Expected one diffuse shell for {atom}, l={angular_momentum}; "
                    f"found {len(diffuse)}."
                )
            shells.extend(diffuse)
        basis[atom] = shells
    return basis if augmented_basis else "def2-qzvp"


def gen_ecp(basis: str, atom_list: list) -> dict:
    """Generate the ECP definition for the requested atoms."""
    if "def2" in basis:
        dict_ = {}
        for atom in def_ecp_list:
            dict_[atom] = basis
        return dict_
    if "cc-" in basis:
        dict_ = {}
        for atom in atom_list:
            if atom in dict_:
                continue
            if "aug-" in basis:
                if atom in aug_cc_pp_atom_list:
                    dict_.update(get_ecp(basis + "-PP", atom))
            else:
                if atom in cc_pp_atom_list:
                    dict_.update(get_ecp(basis + "-PP", atom))
        return dict_
