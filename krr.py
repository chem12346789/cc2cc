import time
import types
import argparse
from itertools import product
from pathlib import Path

import numpy as np
from numba import njit, prange

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split

from cc2cc.utils import DATA_PATH, AU2KCALMOL, CUBE_USE, CUBE_MIDDLE, CUBE_USE_MIDDLE

DATA_PATH = Path("data/grids_dft")
BASIS = "cc-pVDZ"
HASHLEN = 21
HASHSIZE = 10
# pylint: disable=W0621


def load_data(molecular_list, extend_atom, extend_xyz, distance_list):
    """
    Load the data.
    """
    input_dict = {}
    output_dict = {}
    weights_dict = {}
    coords_dict = {}
    keys_list = []
    for (
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
    ) in product(
        molecular_list,
        extend_atom,
        extend_xyz,
        distance_list,
    ):
        name = f"{name_mol}_{BASIS}_{extend_atom}_{extend_xyz}_{distance:.4f}"
        data_path = Path(f"{DATA_PATH}") / f"data_{name}.npz"
        if not (data_path).exists():
            print(f"No file: {data_path}")
            continue

        data = np.load(data_path)

        output_1 = data["exc_over_dm_b3lyp_grids"]
        output_2 = data["exc_over_dm_cc_2_grids"]
        output_3 = data["exc_over_dm_cc_1_j_grids"]
        output_4 = data["exc_over_dm_cc_1_k_grids"]
        output_ = np.array(
            [
                output_1,
                output_2,
                output_3,
                output_4,
            ]
        ).T
        weights_ = data["weights"]
        coords_cube = data["coor_cube"]
        coords_cube = coords_cube[
            :,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            :,
        ]
        input_ = data["rho_cube"]
        input_ = input_[
            :,
            :,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
            CUBE_MIDDLE - CUBE_USE_MIDDLE : CUBE_MIDDLE + CUBE_USE_MIDDLE + 1,
        ]
        if CUBE_USE > 1:
            swap_ = input_[:, :, 0, 0, 0].copy()
            input_[:, :, 0, 0, 0] = input_[:, :, 1, 1, 1].copy()
            input_[:, :, 1, 1, 1] = swap_.copy()
        input_ = input_.reshape(-1, (CUBE_USE) ** 3 * 4)
        print(f"max input: {np.max(input_)}, min input: {np.min(input_)}")
        input_dict[name] = input_

        output_dict[name] = output_
        weights_dict[name] = weights_

        if CUBE_USE > 1:
            swap_ = coords_cube[:, 0, 0, 0, :].copy()
            coords_cube[:, 0, 0, 0, :] = coords_cube[:, 1, 1, 1, :].copy()
            coords_cube[:, 1, 1, 1, :] = swap_.copy()
        coords_cube = coords_cube.reshape(-1, (CUBE_USE) ** 3, 3)
        coords_dict[name] = coords_cube

        keys_list.append(name)
        if len(np.shape(output_dict[name])) == 1:
            print(
                AU2KCALMOL
                * np.sum(
                    np.abs(
                        input_dict[name][:, 0] * output_dict[name] * weights_dict[name]
                    )
                ),
                AU2KCALMOL
                * np.sum(
                    input_dict[name][:, 0] * output_dict[name] * weights_dict[name]
                ),
            )
    return input_dict, output_dict, weights_dict, coords_dict, keys_list


def evaluate(krr, x_train, y_train, w_train, x_all, y_all, w_all):
    """
    Evaluate the model.
    """
    print("Krr perdict:")
    train_error = np.sum(
        (np.abs(y_train - krr.predict_data(x_train)) * w_train * x_train[:, 0])
    )
    print("train", AU2KCALMOL * train_error, "KCAL/MOL", flush=True)
    error_krr = AU2KCALMOL * np.sum(
        (np.abs(y_all - krr.predict_data(x_all)) * w_all * x_all[:, 0])
    )
    print(f"test, {error_krr} KCAL/MOL", flush=True)

    print("B3lyp perdict:")
    b3lyp_error = AU2KCALMOL * np.sum(np.abs(y_train * w_train * x_train[:, 0]))
    print("train", b3lyp_error, "KCAL/MOL", flush=True)
    b3lyp_error = AU2KCALMOL * np.sum(np.abs(y_all * w_all * x_all[:, 0]))
    print("test", b3lyp_error, "KCAL/MOL", flush=True)
    print("End of evaluate.\n", flush=True)
    return np.abs(error_krr)


def add_data(krr, x_train, y_train, w_train, x_test, y_test, w_test):
    """
    Add data which has large error.
    """
    print("Add data:", flush=True)
    error_test = (y_test - krr.predict_data(x_test)) * w_test * x_test[:, 0]

    index_add = (
        np.array([True] * len(error_test))
        if len(error_test) < 51
        else (error_test**2 > np.sort(error_test**2, axis=0)[-51])
    )
    print(AU2KCALMOL * error_test[index_add])
    print(np.max(krr.kernel_matrix[index_add], axis=1))
    x_train = np.concatenate([x_train, x_test[index_add]])
    y_train = np.concatenate([y_train, y_test[index_add]])
    w_train = np.concatenate([w_train, w_test[index_add]])
    x_test = x_test[~index_add]
    y_test = y_test[~index_add]
    w_test = w_test[~index_add]

    print(
        "Length of x_train:",
        len(x_train),
        "Length of x_test:",
        len(x_test),
        flush=True,
    )
    print("End of add data.\n", flush=True)
    return x_train, y_train, w_train, x_test, y_test, w_test


def gen_logger(distance_list):
    """
    Function to distance list and generate logger
    """
    if len(distance_list) == 3:
        distance_l = np.linspace(
            distance_list[0], distance_list[1], int(distance_list[2])
        )
    else:
        distance_l = distance_list
    return distance_l


args_parse = argparse.ArgumentParser()
args_parse.add_argument(
    "--gamma",
    type=float,
    default=10,
)
args_parse.add_argument(
    "--alpha",
    type=float,
    default=1e-8,
)
args_parse.add_argument(
    "--kernel",
    type=str,
    default="rbf",
)
args_parse.add_argument(
    "--molecular_list",
    nargs="+",
    type=str,
    default=[
        "methane",
        "ethane",
        "ethylene",
        "acetylene",
    ],
    help="Name of molecular.",
)
args_parse.add_argument(
    "--distance_list",
    nargs="+",
    type=float,
    help="Distance between atom H to the origin. Default is 1.0.",
    default=[0.0],
)
args_parse.add_argument(
    "--extend_atom",
    type=str,
    nargs="+",
    default=["0-1"],
    help="Number of atoms to extend. Default is 0.",
)
args_parse.add_argument(
    "--extend_xyz",
    type=int,
    nargs="+",
    default=[0],
    help="Number of xyz to extend. 0 for x, 1 for y, 2 for z. Default is 0.",
)
args = args_parse.parse_args()
for i in range(len(args.extend_xyz)):
    args.extend_xyz[i] += 1
args.distance_list = gen_logger(args.distance_list)


print("gamma:", args.gamma)
print("alpha:", args.alpha)
print("kernel:", args.kernel, flush=True)

(
    input_dict,
    output_dict,
    weights_dict,
    coords_dict,
    keys_list,
) = load_data(
    args.molecular_list,
    args.extend_atom,
    args.extend_xyz,
    args.distance_list,
)

krr = KernelRidge(
    alpha=args.alpha,
    gamma=args.gamma,
    kernel="precomputed",
)


@njit(parallel=True)
def get_kernel(x1, x2, gamma=100.0, kernel_type="rbf"):
    """
    Precompute the kernel.
    Rbf kernel.
    Using numba to speed up.
    """
    kernel = np.zeros((x1.shape[0], x2.shape[0]))
    for j in prange(x2.shape[0]):
        for i in range(x1.shape[0]):
            if kernel_type == "rbf":
                kernel[i, j] = np.exp(-gamma * np.sum((x1[i] - x2[j]) ** 2))
            elif kernel_type == "linear":
                kernel[i, j] = np.dot(x1[i], x2[j])
            elif kernel_type == "laplacian":
                kernel[i, j] = np.exp(-gamma * np.sum(np.abs(x1[i] - x2[j])))
    return kernel


def fit_data(self, x, y):
    """
    Modify the fit method to use the precomputed kernel.
    """
    self.x_fit = x.copy()
    self.kernel_matrix = get_kernel(x, x, self.gamma, self.kernel_type)
    self.fit(self.kernel_matrix, y)


def predict_data(self, x):
    """
    Modify the predict method to use the precomputed kernel.
    """
    self.kernel_matrix = get_kernel(x, self.x_fit, self.gamma, self.kernel_type)
    return self.kernel_matrix @ self.dual_coef_


krr.fit_data = types.MethodType(fit_data, krr)
krr.predict_data = types.MethodType(predict_data, krr)

x_all = {}
y_all = {}
w_all = {}
# x_neighborhood = {}
# y_neighborhood = {}
# w_neighborhood = {}
coor_all = {}
name_all = {}
dual_coef = {}
x_fit = {}

shape_matrix = (20, 194, 40)

for key in keys_list:
    print(f"Key: {key}", flush=True)
    for index_ in range(len(input_dict[key])):
        if index_ % (len(input_dict[key]) / 10) == 0:
            print(
                f"Processing: {index_ / (len(input_dict[key]) / 10) * 10:.1f}%",
                flush=True,
            )

        index_round0 = int(input_dict[key][index_, 0] * HASHSIZE)
        index_round1 = int(input_dict[key][index_, 1] * HASHSIZE)
        index_round2 = int(input_dict[key][index_, 2] * HASHSIZE)
        index_round3 = int(input_dict[key][index_, 3] * HASHSIZE)

        if np.abs(index_round0) >= HASHLEN // 2:
            index_round0 = np.sign(index_round0) * HASHLEN // 2
        if np.abs(index_round1) >= HASHLEN // 2:
            index_round1 = np.sign(index_round1) * HASHLEN // 2
        if np.abs(index_round2) >= HASHLEN // 2:
            index_round2 = np.sign(index_round2) * HASHLEN // 2
        if np.abs(index_round3) >= HASHLEN // 2:
            index_round3 = np.sign(index_round3) * HASHLEN // 2
        index_round1 += HASHLEN // 2
        index_round2 += HASHLEN // 2
        index_round3 += HASHLEN // 2

        index_round = int(
            index_round0 * HASHLEN * HASHLEN * HASHLEN
            + index_round1 * HASHLEN * HASHLEN
            + index_round2 * HASHLEN
            + index_round3
        )

        iatm, iang, irad = np.unravel_index(index_, shape_matrix)

        if index_round not in x_all:
            x_all[index_round] = [input_dict[key][index_]]
            y_all[index_round] = [output_dict[key][index_]]
            w_all[index_round] = [weights_dict[key][index_]]
            coor_all[index_round] = [coords_dict[key][index_]]
            name_all[index_round] = [f"{key}_{iatm}_{irad}_{iang}"]
        else:
            x_all[index_round].append(input_dict[key][index_])
            y_all[index_round].append(output_dict[key][index_])
            w_all[index_round].append(weights_dict[key][index_])
            coor_all[index_round].append(coords_dict[key][index_])
            name_all[index_round].append(f"{key}_{iatm}_{irad}_{iang}")

for index_ in np.sort(list(x_all.keys())):
    x_all[index_] = np.array(x_all[index_])
    y_all[index_] = np.array(y_all[index_])
    w_all[index_] = np.array(w_all[index_])
    coor_all[index_] = np.array(coor_all[index_])
    name_all[index_] = np.array(name_all[index_])

train_error_sum = 0
energy_correct_sum = 0

for index_ in np.sort(list(x_all.keys())):
    index_round0 = index_ // (HASHLEN * HASHLEN * HASHLEN)
    index_round1 = (
        index_ % (HASHLEN * HASHLEN * HASHLEN) // (HASHLEN * HASHLEN) - HASHLEN // 2
    )
    index_round2 = index_ % (HASHLEN * HASHLEN) // HASHLEN - HASHLEN // 2
    index_round3 = index_ % HASHLEN - HASHLEN // 2

    print(
        index_round0 / HASHSIZE,
        index_,
        index_round0,
        index_round1,
        index_round2,
        index_round3,
    )

    if index_round0 > 0:
        break
    np.savez_compressed(
        f"data/save/train_test-{index_round0}-{index_round1}-{index_round2}-{index_round3}.npz",
        x=x_all[index_],
        y=y_all[index_],
        w=w_all[index_],
        coor=coor_all[index_],
        name=name_all[index_],
    )

    # if index_round0 >= HASHLEN // 2:
    #     krr.alpha = args.alpha
    #     krr.gamma = 0.1
    #     krr.kernel_type = "rbf"
    # else:
    #     krr.alpha = args.alpha
    #     krr.gamma = args.gamma
    #     krr.kernel_type = "rbf"

    # if len(x_all[index_]) < 500:
    #     krr.fit_data(x_all[index_], y_all[index_])
    # else:
    #     x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
    #         x_all[index_],
    #         y_all[index_],
    #         w_all[index_],
    #         train_size=50,
    #         random_state=42,
    #     )

    #     CONVERGE_STEP = 0
    #     for i_step in range(500):
    #         print(f"Step {i_step}:", flush=True)
    #         krr.fit_data(x_train, y_train)
    #         error_krr = evaluate(
    #             krr,
    #             x_train,
    #             y_train,
    #             w_train,
    #             x_all[index_],
    #             y_all[index_],
    #             w_all[index_],
    #         )
    #         if error_krr < 0.25:
    #             print(f"Error is small: {error_krr}", flush=True)
    #             CONVERGE_STEP += 1
    #             if CONVERGE_STEP == 1:
    #                 print("Converge.")
    #                 break
    #         x_train, y_train, w_train, x_test, y_test, w_test = add_data(
    #             krr, x_train, y_train, w_train, x_test, y_test, w_test
    #         )

    # train_error = np.sum(
    #     (y_all[index_] - krr.predict_data(x_all[index_]))
    #     * w_all[index_]
    #     * x_all[index_][:, 0]
    # )

    # if (
    #     index_round0 == 0
    #     and index_round1 == 0
    #     and index_round2 == 0
    #     and index_round3 == 0
    # ):
    #     krr.alpha = args.alpha
    #     krr.kernel_type = "rbf"
    # else:
    #     krr.alpha = 1e-8
    #     krr.kernel_type = "rbf"

    # if len(x_all[index_]) < 500:
    #     krr.fit_data(x_all[index_], y_all[index_])
    # else:
    #     x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
    #         x_all[index_],
    #         y_all[index_],
    #         w_all[index_],
    #         train_size=50,
    #         random_state=42,
    #     )

    #     CONVERGE_STEP = 0
    #     for i_step in range(500):
    #         print(f"Step {i_step}:", flush=True)
    #         krr.fit_data(x_train, y_train)
    #         error_krr = evaluate(
    #             krr,
    #             x_train,
    #             y_train,
    #             w_train,
    #             x_all[index_],
    #             y_all[index_],
    #             w_all[index_],
    #         )
    #         if error_krr < 0.1:
    #             print(f"Error is small: {error_krr}", flush=True)
    #             CONVERGE_STEP += 1
    #             if CONVERGE_STEP == 1:
    #                 print("Converge.")
    #                 break
    #         x_train, y_train, w_train, x_test, y_test, w_test = add_data(
    #             krr, x_train, y_train, w_train, x_test, y_test, w_test
    #         )

    # train_error = np.sum(
    #     (y_all[index_] - krr.predict_data(x_all[index_]))
    #     * w_all[index_]
    #     * x_all[index_][:, 0]
    # )
    # energy_correct = np.sum(x_all[index_][:, 0] * y_all[index_] * w_all[index_])

    # train_error_sum += train_error
    # energy_correct_sum += energy_correct

    # if np.abs(AU2KCALMOL * train_error) > 0.01:
    #     print(
    #         f"Round {index_round0} {index_round1} {index_round2} {index_round3}",
    #         flush=True,
    #     )
    #     print(
    #         f"Length of x_all: {len(x_all[index_])}",
    #         flush=True,
    #     )
    #     print(
    #         f"Train error: {AU2KCALMOL * train_error} KCAL/MOL",
    #         flush=True,
    #     )
    #     print(
    #         f"Energy correct: {AU2KCALMOL * energy_correct} KCAL/MOL",
    #         flush=True,
    #     )

    # dual_coef[index_] = krr.dual_coef_
    # x_fit[index_] = krr.x_fit
    # print()

# save the model
np.savez_compressed(
    f"data/save/dual_coef-final-{args.alpha}-{args.gamma}.npz",
    dual_coef=dual_coef,
    x_fit=x_fit,
)

print(
    f"Energy correct sum: {AU2KCALMOL * energy_correct_sum} KCAL/MOL",
    f"Train error sum: {AU2KCALMOL * train_error_sum} KCAL/MOL",
    flush=True,
)
