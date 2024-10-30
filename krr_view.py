import argparse
from pathlib import Path

import numpy as np

from sklearn.kernel_ridge import KernelRidge

from krr import add_args, load_data, hash_value

DATA_PATH = Path("data/grids_dft")
BASIS = "cc-pVDZ"

parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
args = add_args(parser)

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

        index_round = hash_value(input_dict[key][index_])
        (iatm, iang, irad) = np.unravel_index(index_, shape_matrix)

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

x_keys = list(x_all.keys())
for index_ in np.sort(list(x_all.keys())):
    x_all[index_] = np.array(x_all[index_])
    y_all[index_] = np.array(y_all[index_])
    w_all[index_] = np.array(w_all[index_])
    coor_all[index_] = np.array(coor_all[index_])
    name_all[index_] = np.array(name_all[index_])

train_error_sum = 0
energy_correct_sum = 0

for index_ in x_keys:
    if int(index_.split("_")[0]) != 0:
        break
    np.savez_compressed(
        f"data/save/train_test_new-{index_}.npz",
        x=x_all[index_],
        y=y_all[index_],
        w=w_all[index_],
        coor=coor_all[index_],
        name=name_all[index_],
    )
