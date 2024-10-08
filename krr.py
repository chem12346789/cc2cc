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

# pylint: disable=W0621


def load_data(molecular_list, extend_atom, extend_xyz, distance_list, if_normal=True):
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

        for index_ in range(len(input_)):
            print(input_[index_].shape)

        output_dict[name] = output_
        weights_dict[name] = weights_
        keys_list.append(name)
    return input_dict, output_dict, weights_dict, keys_list


def evaluate(
    krr, x_train, input_dict, y_train, output_dict, w_train, weights_dict, keys_list
):
    """
    Evaluate the model.
    """
    train_error = np.sum(
        (y_train - krr.predict_data(x_train)) * w_train * x_train[:, 0]
    )
    print("Krr perdict:")
    print("train", AUTOKCALMOL * train_error, "KCAL/MOL", flush=True)
    for key in keys_list:
        error_krr = AUTOKCALMOL * np.sum(
            (output_dict[key] - krr.predict_data(input_dict[key]))
            * weights_dict[key]
            * input_dict[key][:, 0]
        )
        print(f"{key} test, {error_krr} KCAL/MOL", flush=True)
    print("B3lyp perdict:")
    b3lyp_error = np.sum(y_train * w_train * x_train[:, 0])
    print("train", AUTOKCALMOL * b3lyp_error, "KCAL/MOL", flush=True)
    for key in keys_list:
        b3lyp_error = np.sum(
            output_dict[key] * weights_dict[key] * input_dict[key][:, 0]
        )
        print(key, "test", AUTOKCALMOL * b3lyp_error, "KCAL/MOL", flush=True)
    print("End of evaluate.\n", flush=True)
    return np.abs(error_krr)


def add_data(krr, x_train, y_train, w_train, x_test, y_test, w_test):
    """
    Add data which has large error.
    """
    print("Add data:", flush=True)
    error_test = y_test - krr.predict_data(x_test)
    print("error_test:", np.mean(error_test**2), flush=True)

    index_add = error_test**2 > np.sort(error_test**2, axis=0)[-51]
    print(index_add)
    print(error_test[index_add])
    print("Length of x_train:", len(x_train), flush=True)
    x_train = np.concatenate([x_train, x_test[index_add]])
    y_train = np.concatenate([y_train, y_test[index_add]])
    w_train = np.concatenate([w_train, w_test[index_add]])
    x_test = x_test[~index_add]
    y_test = y_test[~index_add]
    w_test = w_test[~index_add]
    print("End of add data.\n", flush=True)
    return x_train, y_train, w_train, x_test, y_test, w_test


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

i_step = 0

for training_set in [0, 1, 2]:
    if training_set == 0:
        x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
            input_dict[keys_list[training_set]],
            output_dict[keys_list[training_set]],
            weights_dict[keys_list[training_set]],
            train_size=50,
            random_state=42,
        )
        np.savez_compressed(
            f"train_test-{time.strftime('%Y%m%d-%H%M%S')}.npz",
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            w_train=w_train,
            w_test=w_test,
        )
    else:
        x_test = input_dict[keys_list[training_set]].copy()
        y_test = output_dict[keys_list[training_set]].copy()
        w_test = weights_dict[keys_list[training_set]].copy()

    CONVERGE_STEP = 0
    for i_step in range(i_step, i_step + 250):
        print(f"Step {i_step}:", flush=True)
        krr.fit_data(x_train, y_train)
        error_krr = evaluate(
            krr,
            x_train,
            input_dict,
            y_train,
            output_dict,
            w_train,
            weights_dict,
            keys_list[: training_set + 1],
        )
        if error_krr < 0.5:
            print(f"Error is small: {error_krr}", flush=True)
            CONVERGE_STEP += 1
            if CONVERGE_STEP == 2:
                print("Converge.")
                break
        x_train, y_train, w_train, x_test, y_test, w_test = add_data(
            krr, x_train, y_train, w_train, x_test, y_test, w_test
        )


np.savez_compressed(
    f"train_test-final-{time.strftime('%Y%m%d-%H%M%S')}.npz",
    x_train=x_train,
    y_train=y_train,
    w_train=w_train,
    dual_coef=krr.dual_coef_,
)
