import time
import types
import argparse
from itertools import product
from pathlib import Path

import numpy as np
from numba import njit, prange

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/grids_dft")
AUTOKCALMOL = 627.509
BASIS = "cc-pVDZ"
HASHLEN = 10000
HASHSIZE = 10
# pylint: disable=W0621


def load_data(molecular_list, extend_atom, extend_xyz, distance_list):
    """
    Load the data.
    """
    input_dict = {}
    output_dict = {}
    weights_dict = {}
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

        input_ = data["rho_inv_4_norm"]
        output_ = data["exc_over_dm_cc_grids"]
        weights_ = data["weights"]
        input_dict[name] = np.transpose(input_, (1, 0))
        for index_ in range(4):
            input_[index_, :] = input_[index_, :] / (1 if index_ == 0 else np.sqrt(20))
            print(np.var(input_[index_, :]))
            print(np.max(input_[index_, :]), np.min(input_[index_, :]))
        output_dict[name] = output_
        weights_dict[name] = weights_
        keys_list.append(name)
    return input_dict, output_dict, weights_dict, keys_list


args_parse = argparse.ArgumentParser()
args_parse.add_argument(
    "--gamma",
    type=float,
    default=100,
)
args_parse.add_argument(
    "--alpha",
    type=float,
    default=0.01,
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

print("gamma:", args.gamma)
print("alpha:", args.alpha)
print("kernel:", args.kernel, flush=True)

(input_dict, output_dict, weights_dict, keys_list) = load_data(
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
def get_kernel(x1, x2, gamma=100.0):
    """
    Precompute the kernel.
    Rbf kernel.
    Using numba to speed up.
    """
    kernel = np.zeros((x1.shape[0], x2.shape[0]))
    for j in prange(x2.shape[0]):
        for i in range(x1.shape[0]):
            kernel[i, j] = np.exp(-gamma * np.sum((x1[i] - x2[j]) ** 2))
    return kernel


def fit_data(self, x, y):
    """
    Modify the fit method to use the precomputed kernel.
    """
    self.x_fit = x.copy()
    self.fit(get_kernel(x, x, self.gamma), y)


def predict_data(self, x):
    """
    Modify the predict method to use the precomputed kernel.
    """
    return self.predict(get_kernel(x, self.x_fit, self.gamma))


krr.fit_data = types.MethodType(fit_data, krr)
krr.predict_data = types.MethodType(predict_data, krr)

x_all = {}
y_all = {}
w_all = {}
dual_coef = {}

for key in keys_list:
    print(f"Key: {key}", flush=True)
    for index_ in range(len(input_dict[key])):
        if index_ % (len(input_dict[key]) / 10) == 0:
            print(
                f"Processing: {index_ / (len(input_dict[key]) / 10) * 10:.1f}%",
                flush=True,
            )
        index_round0 = int(input_dict[key][index_, 0] * HASHSIZE)
        index_round1 = int(input_dict[key][index_, 1] * HASHSIZE) + HASHLEN // 2
        index_round2 = int(input_dict[key][index_, 2] * HASHSIZE) + HASHLEN // 2
        index_round3 = int(input_dict[key][index_, 3] * HASHSIZE) + HASHLEN // 2
        index_round = int(
            index_round0 * HASHLEN * HASHLEN * HASHLEN
            + index_round1 * HASHLEN * HASHLEN
            + index_round2 * HASHLEN
            + index_round3
        )

        if index_round not in x_all:
            x_all[index_round] = np.array([input_dict[key][index_]])
            y_all[index_round] = np.array([output_dict[key][index_]])
            w_all[index_round] = np.array([weights_dict[key][index_]])
        else:
            x_all[index_round] = np.append(
                x_all[index_round],
                [input_dict[key][index_]],
                axis=0,
            )
            y_all[index_round] = np.append(
                y_all[index_round],
                output_dict[key][index_],
            )
            w_all[index_round] = np.append(
                w_all[index_round],
                weights_dict[key][index_],
            )

train_error_sum = 0
energy_correct_sum = 0

for index_ in np.sort(list(x_all.keys())):
    index_round0 = index_ // (HASHLEN * HASHLEN * HASHLEN)
    index_round1 = (
        index_ % (HASHLEN * HASHLEN * HASHLEN) // (HASHLEN * HASHLEN) - HASHLEN // 2
    )
    index_round2 = index_ % (HASHLEN * HASHLEN) // HASHLEN - HASHLEN // 2
    index_round3 = index_ % HASHLEN - HASHLEN // 2

    print(index_round0 / HASHSIZE, index_)

    np.savez_compressed(
        f"train_test-final-{index_}.npz",
        x=x_all[index_],
        y=y_all[index_],
        w=w_all[index_],
    )

    if len(x_all[index_]) < 500:
        krr.fit_data(x_all[index_], y_all[index_])
    else:
        x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
            x_all[index_],
            y_all[index_],
            w_all[index_],
            train_size=500,
            random_state=0,
        )
        krr.fit_data(x_train, y_train)
        test_error = np.sum(
            np.abs(y_test - krr.predict_data(x_test)) * w_test * x_test[:, 0]
        )

    train_error = np.sum(
        np.abs(
            (y_all[index_] - krr.predict_data(x_all[index_]))
            * w_all[index_]
            * x_all[index_][:, 0]
        )
    )
    energy_correct = np.sum(np.abs(x_all[index_][:, 0] * y_all[index_] * w_all[index_]))

    train_error_sum += train_error
    energy_correct_sum += energy_correct

    if AUTOKCALMOL * train_error > 0.1 / HASHSIZE:
        print(
            f"Round {index_round0} {index_round1} {index_round2} {index_round3}",
            flush=True,
        )
        print("Length of x_all:", len(x_all[index_]), flush=True)

        print(
            f"Energy correct: {AUTOKCALMOL * energy_correct} KCAL/MOL",
            flush=True,
        )
        print(
            f"Train error: {AUTOKCALMOL * train_error} KCAL/MOL",
            flush=True,
        )
    dual_coef[index_] = krr.dual_coef_

    print()

print(
    f"Energy correct sum: {AUTOKCALMOL * energy_correct_sum} KCAL/MOL",
    f"Train error sum: {AUTOKCALMOL * train_error_sum} KCAL/MOL",
    flush=True,
)
