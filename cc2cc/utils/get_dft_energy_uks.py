def get_dft_energy(
    mol,
    grids,
    dft_mo_coeff,
    dm1_dft,
    evaluate=False,
):
    """Calculate unrestricted B3LYP energy-density components."""
    from cc2cc.utils.reference_b3lyp import get_dft_energy as shared_dft_energy

    return shared_dft_energy(
        mol, grids, dft_mo_coeff, dm1_dft,
        evaluate=evaluate, unrestricted=True,
    )
