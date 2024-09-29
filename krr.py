import numpy as np
import argparse
from itertools import product
from pathlib import Path

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/grids_dft")


def load_data(
    molecular_list,
    extend_atom,
    extend_xyz,
    distance_list,
    input_mat,
    output_mat,
    weights_mat,
):
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
        name = f"{name_mol}_{basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"

        if not (Path(f"{DATA_PATH}") / f"data_{name}.npz").exists():
            print(f"No file: {name:>40}")
            continue

        data = np.load(Path(f"{DATA_PATH}") / f"data_{name}.npz")

        input_ = data["rho_inv_4_norm"]
        output_ = data["exc_over_dm_cc_grids"]
        weights_ = data["weights"]

        for i_coord in range(len(output_)):
            input_mat.append(input_[:, i_coord])
            output_mat.append(output_[i_coord])
            weights_mat.append(weights_[i_coord])


args_parse = argparse.ArgumentParser()
args_parse.add_argument("--gamma", type=float, default=100)
args_parse.add_argument("--alpha", type=float, default=0.01)
args_parse.add_argument("--kernel", type=str, default="rbf")
args = args_parse.parse_args()

print("gamma:", args.gamma)
print("alpha:", args.alpha)
print("kernel:", args.kernel, flush=True)

input_mat = []
output_mat = []
weights_mat = []
basis = "cc-pVDZ"
molecular_list = [
    "methane",
    # "ethane",
    # "ethylene",
    # "acetylene",
]
extend_atom = ["0"]
extend_xyz = [1]
distance_list = [0]
load_data(
    molecular_list,
    extend_atom,
    extend_xyz,
    distance_list,
    input_mat,
    output_mat,
    weights_mat,
)


krr = GridSearchCV(
    KernelRidge(),
    param_grid={
        "kernel": [args.kernel],
        "gamma": [args.gamma],
        "alpha": [args.alpha],
    },
)

input_mat = np.array(input_mat)
output_mat = np.array(output_mat)
weights_mat = np.array(weights_mat)

print("input_mat.shape:", input_mat.shape, flush=True)

x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
    input_mat, output_mat, weights_mat, train_size=0.005
)
# x_train = input_mat
# y_train = output_mat
# w_train = weights_mat

krr.fit(x_train, y_train)
autokcalmol = 627.509

print("Krr perdict:")
print(
    autokcalmol
    * np.sum((output_mat - krr.predict(input_mat)) * weights_mat * input_mat[:, 0])
)

print("B3lyp perdict:")
print(autokcalmol * np.sum(output_mat * weights_mat * input_mat[:, 0]), flush=True)


input_mat = []
output_mat = []
weights_mat = []
basis = "cc-pVDZ"
molecular_list = ["ethane"]
extend_atom = ["0"]
extend_xyz = [1]
distance_list = [0]

load_data(
    molecular_list,
    extend_atom,
    extend_xyz,
    distance_list,
    input_mat,
    output_mat,
    weights_mat,
)

input_mat = np.array(input_mat)
output_mat = np.array(output_mat)
weights_mat = np.array(weights_mat)

x_test = np.array(input_mat)
y_test = np.array(output_mat)
w_test = np.array(weights_mat)

print("Krr perdict:")
print(autokcalmol * np.sum((y_test - krr.predict(x_test)) * w_test * x_test[:, 0]))

print("B3lyp perdict:")
print(autokcalmol * np.sum(y_test * w_test * x_test[:, 0]))
