import numpy as np

import pyscf
from pyscf import lib

from cc2cc.utils.modelscf_rks import get_veff_grad_modified_zeros
from cc2cc.utils.env_var import CUBE_MIDDLE, EDGE_SIZE


def get_dft_input(mol, grids, dm1_dft, data_dict, max_memory=8000):
    """
    Calculate the input of (exchange-correlation energy - DFT energy) on the grids.
    """
    mdft = pyscf.scf.UKS(mol)
    mdft.grids = grids
    mdft.xc = "b3lyp"
    mdft.verbose = 4
    mdft.kernel(dm1_dft)

    rho_cube_dft = np.zeros((len(grids.coords), grids.input_level, EDGE_SIZE**3))

    ni = mdft._numint
    step = int(max_memory * 1024**2 / (dm1_dft.shape[-1] * EDGE_SIZE**3 * 32 * 8))
    # 32 is the number of elements in the ao_array and ao_mat, 8 is the size of float64 in bytes
    print(f"Step size: {step}")
    for p0, p1 in lib.prange(0, len(grids.coords), step):
        if grids.screen_index is None:
            mask = None
        else:
            mask = grids.screen_index[p0:p1]
        coords_ = grids.coords[p0:p1]
        gridcube = grids.gen_cube(mol, dm1_dft, coords_, mask)
        rho_cube_dft_part, wv, ao_value = gridcube.gen_cube_rho_uks(
            ni, dm1_dft, ao_deriv=2, require_vxc=True
        )
        rho_cube_dft[p0:p1] = rho_cube_dft_part

    data_dict["rho_cube_dft"] = rho_cube_dft.reshape(
        len(grids.coords), grids.input_level, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE
    )


def get_dft_grad(mol, grids, dm1_dft, data_dict, max_memory=8000):
    """
    Calculate the gradient of (exchange-correlation energy - DFT energy) on the grids.
    Note the max_memory is hard to predict (a large memory usage is due to grad2force and grad_mat), so just set it to a relative small value to avoid OOM.
    """
    mdft = pyscf.scf.UKS(mol)
    mdft.grids = grids
    mdft.xc = "b3lyp"
    mdft.verbose = 4
    mdft.kernel(dm1_dft)
    gdft = mdft.Gradients()
    grad_dft = gdft.kernel()
    get_veff_grad_modified_zeros(gdft)
    grad_dft_zeros = gdft.kernel()

    atmlst = range(mol.natm)
    grad2force = np.zeros(
        (
            len(atmlst),
            grids.input_level,
            len(grids.coords),
            EDGE_SIZE,
            EDGE_SIZE,
            EDGE_SIZE,
            3,
        )
    )

    rho_cube_dft = np.zeros((len(grids.coords), grids.input_level, EDGE_SIZE**3))

    ni = mdft._numint
    step = int(max_memory * 1024**2 / (dm1_dft.shape[-1] * EDGE_SIZE**3 * 32 * 8))
    # 32 is the number of elements in the ao_array and ao_mat, 8 is the size of float64 in bytes
    print(f"Step size: {step}")
    for p0, p1 in lib.prange(0, len(grids.coords), step):
        if grids.screen_index is None:
            mask = None
        else:
            mask = grids.screen_index[p0:p1]
        coords_ = grids.coords[p0:p1]
        gridcube = grids.gen_cube(mol, dm1_dft, coords_, mask)
        rho_cube_dft_part, wv, ao_value = gridcube.gen_cube_rho_uks(
            ni, dm1_dft, ao_deriv=2, require_vxc=True
        )
        rho_cube_dft[p0:p1] = rho_cube_dft_part

        wv = wv.reshape(gridcube.input_level, 2, 4, len(gridcube.coords))
        wv[:, :, 0, :] *= 0.5

        ao_array = np.array([ao_value[0], ao_value[1], ao_value[2], ao_value[3]])
        ao_mat = np.array(
            [
                [ao_value[1], ao_value[2], ao_value[3]],
                [ao_value[4], ao_value[5], ao_value[6]],
                [ao_value[5], ao_value[7], ao_value[8]],
                [ao_value[6], ao_value[8], ao_value[9]],
            ]
        )
        for k, ia in enumerate(atmlst):
            ao0, ao1 = mol.aoslice_by_atom()[ia, 2:]
            print(
                f"size of ao_array: {ao_array.shape} elements, size of ao_mat: {ao_mat.shape} elements, size of ao_value: {ao_value.shape} elements, size of grad2force: {grad2force.shape} elements, size of wv: {wv.shape} elements",
                flush=True,
            )
            grad2force_part = np.einsum(
                "isnp,xpu,npv,suv->ipx",
                wv,
                ao_value[1:4, :, ao0:ao1],
                ao_array,
                dm1_dft[:, ao0:ao1],
                optimize=True,
            ) + np.einsum(
                "isnp,nxpu,pv,suv->ipx",
                wv,
                ao_mat[:, :, :, ao0:ao1],
                ao_value[0],
                dm1_dft[:, ao0:ao1],
                optimize=True,
            )
            grad2force_part = -grad2force_part * 2
            grad2force[k, :, p0:p1] = np.reshape(
                grad2force_part,
                (grids.input_level, p1 - p0, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
            )
        print(
            f"current p0: {p0}, p1: {p1}, current size: {lib.current_memory()[0] / 1024:.2f} GB, max size: {max_memory / 1024:.2f} GB",
        )
        print(
            f"size of ao_array: {ao_array.shape} elements, size of ao_mat: {ao_mat.shape} elements, size of ao_value: {ao_value.shape} elements, size of grad2force: {grad2force.shape} elements",
            flush=True,
        )

    data_dict["rho_cube_dft"] = rho_cube_dft.reshape(
        len(grids.coords), grids.input_level, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE
    )
    data_dict["grad2force"] = grad2force

    # Test force
    grad_mat = np.zeros(
        (grids.input_level, len(grids.coords), EDGE_SIZE, EDGE_SIZE, EDGE_SIZE)
    )
    grad_mat[0, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.08
    grad_mat[1, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.19
    grad_mat[2, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.72
    grad_mat[3, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.81
    force = np.einsum(
        "p,ipabc,tipabcx->tx",
        grids.weights,
        grad_mat,
        data_dict["grad2force"],
        optimize=True,
    )
    print("Error force DFT: ", np.linalg.norm(force - (grad_dft - grad_dft_zeros)))
    data_dict["grad_dft_zeros"] = grad_dft_zeros
