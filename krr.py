import numpy as np
import argparse
from itertools import product
from pathlib import Path

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/grids_dft")
AUTOKCALMOL = 627.509
BASIS = "cc-pVDZ"

# pylint: disable=W0621


def load_data(
    molecular_list,
    extend_atom,
    extend_xyz,
    distance_list,
):
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

        if not (Path(f"{DATA_PATH}") / f"data_{name}.npz").exists():
            print(f"No file: {name:>40}")
            continue

        data = np.load(Path(f"{DATA_PATH}") / f"data_{name}.npz")

        input_ = data["rho_inv_4_norm"]
        output_ = data["exc_over_dm_cc_grids"]
        weights_ = data["weights"]
        input_dict[name] = np.transpose(input_, (1, 0))
        output_dict[name] = output_
        weights_dict[name] = weights_
        keys_list.append(name)
    return input_dict, output_dict, weights_dict, keys_list


def evaluate(
    krr,
    x_train,
    input_dict,
    y_train,
    output_dict,
    w_train,
    weights_dict,
    keys_list,
):
    """
    Evaluate the model.
    """
    print("Krr perdict:")
    print(
        "train",
        AUTOKCALMOL
        * np.sum((y_train - krr.predict(x_train)) * w_train * x_train[:, 0]),
        flush=True,
    )
    for key in keys_list:
        print(
            key,
            "test",
            AUTOKCALMOL
            * np.sum(
                (output_dict[key] - krr.predict(input_dict[key]))
                * weights_dict[key]
                * input_dict[key][:, 0]
            ),
            "KCAL/MOL",
            flush=True,
        )
    print("B3lyp perdict:")
    print(
        "train",
        AUTOKCALMOL * np.sum(y_train * w_train * x_train[:, 0]),
        "KCAL/MOL",
        flush=True,
    )
    for key in keys_list:
        print(
            key,
            "test",
            AUTOKCALMOL
            * np.sum(output_dict[key] * weights_dict[key] * input_dict[key][:, 0]),
            "KCAL/MOL",
            flush=True,
        )


def add_data(
    krr,
    x_train,
    y_train,
    w_train,
    x_text,
    y_text,
    w_text,
    input_dict,
    output_dict,
    weights_dict,
    keys_list,
    training_set,
):
    """
    Add data which has large error.
    """
    error_train = y_train - krr.predict(x_train)
    benchmark_error = np.mean(error_train**2)
    print("error_train:", benchmark_error, flush=True)
    print("keys_list:", keys_list, flush=True)
    for key in keys_list:
        if training_set == 0:
            error_test = y_test - krr.predict(x_test)
            print("error_test:", np.mean(error_test**2), flush=True)
            index_add = error_test**2 > benchmark_error * 2
            print("length of index_add:", np.sum(index_add), flush=True)
            x_train = np.concatenate([x_train, x_test[index_add]])
            y_train = np.concatenate([y_train, y_test[index_add]])
            w_train = np.concatenate([w_train, w_test[index_add]])
        else:
            error_test = output_dict[key] - krr.predict(input_dict[key])
            print(key, "error_test:", np.mean(error_test**2), flush=True)
            index_add = error_test**2 > benchmark_error * 2
            print("length of index_add:", np.sum(index_add), flush=True)
            x_train = np.concatenate([x_train, input_dict[key][index_add]])
            y_train = np.concatenate([y_train, output_dict[key][index_add]])
            w_train = np.concatenate([w_train, weights_dict[key][index_add]])
    return x_train, y_train, w_train


args_parse = argparse.ArgumentParser()
args_parse.add_argument(
    "--gamma",
    type=float,
    default=100,
)
args_parse.add_argument(
    "--alpha",
    type=float,
    default=0.001,
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
        # "ethylene",
        # "acetylene",
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
    default=[0],
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

krr = GridSearchCV(
    KernelRidge(),
    param_grid={
        "kernel": [args.kernel],
        "gamma": [args.gamma],
        "alpha": [args.alpha],
    },
)

x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
    input_dict[keys_list[0]],
    output_dict[keys_list[0]],
    weights_dict[keys_list[0]],
    train_size=0.2,
)

for training_set in range(len(keys_list) + 1):
    krr.fit(x_train, y_train)
    evaluate(
        krr,
        x_train,
        input_dict,
        y_train,
        output_dict,
        w_train,
        weights_dict,
        keys_list[: training_set + 1],
    )
    x_train, y_train, w_train = add_data(
        krr,
        x_train,
        y_train,
        w_train,
        x_test,
        y_test,
        w_test,
        input_dict,
        output_dict,
        weights_dict,
        keys_list[training_set : training_set + 1],
        training_set,
    )
