"""Legacy entry points; inversion and diagnostics live in zmp_reference."""

from cc2cc.utils.zmp_reference import LEVEL_SHIFT, MAX_CYCLE, get_zmp


def get_zmp_rks(
    mol, dm_tar, dm_dft, grids, max_l=4, verbose=0, start_l=0, max_cycle=MAX_CYCLE
):
    return get_zmp(
        mol, dm_tar, dm_dft, grids, max_l, verbose, start_l, max_cycle
    )


def get_zmp_uks(
    mol, dm_tar, dm_dft, grids, max_l=4, verbose=0, start_l=0, max_cycle=MAX_CYCLE
):
    return get_zmp(
        mol, dm_tar, dm_dft, grids, max_l, verbose, start_l, max_cycle
    )
