import pyscf
import basis_set_exchange

BASIS = {
    # aug-cc-pwcv
    "aug-cc-pwcv6z": "aug-cc-pv6z",
    "aug-cc-pwcv5z": "aug-cc-pv5z",
    "aug-cc-pwcvqz": "aug-cc-pvqz",
    "aug-cc-pwcvtz": "aug-cc-pvtz",
    "aug-cc-pwcvdz": "aug-cc-pvdz",
    # aug-cc-pwcv
    "augccpwcv6z": "aug-cc-pv6z",
    "augccpwcv5z": "aug-cc-pv5z",
    "augccpwcvqz": "aug-cc-pvqz",
    "augccpwcvtz": "aug-cc-pvtz",
    "augccpwcvdz": "aug-cc-pvdz",
    # aug-cc-pcv
    "aug-cc-pcv6z": "aug-cc-pv6z",
    "aug-cc-pcv5z": "aug-cc-pv5z",
    "aug-cc-pcvqz": "aug-cc-pvqz",
    "aug-cc-pcvtz": "aug-cc-pvtz",
    "aug-cc-pcvdz": "aug-cc-pvdz",
    # ccpwpcv
    "ccpwpcv6z": "cc-pv6z",
    "ccpwpcv5z": "cc-pv5z",
    "ccpwpcvqz": "cc-pvqz",
    "ccpwpcvtz": "cc-pvtz",
    "ccpwpcvdz": "cc-pvdz",
    # cc-pcv
    "cc-pcv6z": "cc-pv6z",
    "cc-pcv5z": "cc-pv5z",
    "cc-pcvqz": "cc-pvqz",
    "cc-pcvtz": "cc-pvtz",
    "cc-pcvdz": "cc-pvdz",
}


def gen_basis(molecular, basis_name, if_basis_str):
    """
    Generate the basis set for the molecular system.
    Use the basis_set_exchange basis.
    Avoid no core correlation basis set (such as cc-pcvdz) for H atom; See https://github.com/pyscf/pyscf/issues/1795
    """
    basis_name = basis_name.lower()
    basis = {}
    for i_atom in molecular:
        if if_basis_str:
            basis[i_atom[0]] = pyscf.gto.load(
                (
                    basis_set_exchange.api.get_basis(
                        BASIS[basis_name],
                        elements=i_atom[0],
                        fmt="nwchem",
                    )
                    if ((i_atom[0] == "H") and (basis_name in BASIS))
                    else basis_set_exchange.api.get_basis(
                        basis_name, elements=i_atom[0], fmt="nwchem"
                    )
                ),
                i_atom[0],
            )
        else:
            basis[i_atom[0]] = (
                BASIS[basis_name]
                if ((i_atom[0] == "H") and (basis_name in BASIS))
                else basis_name
            )
    return basis
