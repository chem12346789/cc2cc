"""Benchmark dft. Restrict Khon-Sham (no spin)."""

from cc2cc.utils import TestDataDFT


def benchmark_rks(mol, name, data_record):
    """
    Benchmark dft. Restrict Khon-Sham (no spin).
    """
    dict_ = {}
    for xc_code, disp in [
        # ("b3lyp", None),
        # ("b3lyp", "d3bj"),
        ("b3lyp", "d3zero"),
        # ("blyp", None),
        # ("blyp", "d3bj"),
    ]:
        test_data = TestDataDFT(
            mol,
            name,
            xc_code=xc_code,
            disp=disp,
        )
        xc_code_disp = xc_code if disp is None else f"{xc_code}-{disp}"

        dict_.update(
            {
                "name": name,
                f"{xc_code_disp}_ene": test_data.e_dft,
                f"{xc_code_disp}_dipole_x": test_data.dft_dipole[0],
                f"{xc_code_disp}_dipole_y": test_data.dft_dipole[1],
                f"{xc_code_disp}_dipole_z": test_data.dft_dipole[2],
            }
        )

    data_record.add_data(dict_)
    data_record.save_csv()
