import argparse
from itertools import product
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np

import faiss

from sklearn.kernel_ridge import KernelRidge

from krr import add_args, load_data, hash_value, append
from cc2cc.utils import ARRAY_USE_MIDDLE

parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
args = add_args(parser)

print("gamma:", args.gamma)
print("alpha:", args.alpha)
print("kernel:", args.kernel, flush=True)

view_dict = {
    "cc": "exc_over_dm_cc_grids",
}

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
    args.basis,
    view_dict=view_dict,
)

krr = KernelRidge(alpha=args.alpha, gamma=args.gamma, kernel="precomputed")

x_all = {}
y_all = {}
w_all = {}
coor_all = {}
name_all = {}
dual_coef = {}
x_fit = {}

center_piont = [-0.01, -0.001, 0.001, 0.01]
hashtable = append(np.linspace(-1, 0, 11)[:-1], center_piont, np.linspace(0, 1, 11)[1:])
list_1 = list(range(-len(center_piont) // 2 + 1, len(center_piont) // 2, 1))
list_2 = [-len(hashtable) // 2, len(hashtable) // 2]
print(hashtable, list_1, list_2)
shape_matrix = (20, 194, 40)

for key in keys_list:
    print(f"Key: {key}", flush=True)
    for index_ in range(len(input_dict[key])):
        if index_ % (len(input_dict[key]) / 10) == 0:
            print(
                f"Processing: {index_ / (len(input_dict[key]) / 10) * 10:.1f}%",
                flush=True,
            )

        index_round = hash_value(input_dict[key][index_], hashtable=hashtable)
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
    if index_ not in [
        "0_0",
        "1_1",
        "2_2",
        f"{len(hashtable)//2}_{len(hashtable)//2}",
    ]:
        continue

    for name_plot_i, name_plot in enumerate(view_dict.keys()):
        name_plot = name_plot + "-" + index_
        x = x_all[index_]
        y = y_all[index_]
        w = w_all[index_]
        coor = coor_all[index_]
        name = name_all[index_]
        y = y[:, name_plot_i]

        name_plot = f"{name_plot}-{x.shape[1]}"
        print(x.shape, flush=True)

        (num_sample, dim_sample) = x.shape
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, dim_sample, flat_config)
        index.add(x)
        b = x.copy()
        distances, indices = index.search(b, 20)
        var_y = np.var(
            np.einsum("ij,i->ij", y[indices], x[:, ARRAY_USE_MIDDLE] * w), axis=1
        )

        for max_x_ in [1e-2, 1e-3, 1e-4, 1e-5]:
            print("\n begin of print", flush=True)
            Path(f"plot/{name_plot}-{max_x_}/plot/").mkdir(parents=True, exist_ok=True)
            max_x = max_x_**2 * x.shape[1]

            mean_of_distances = np.max(distances, axis=1)
            mean_of_distances[mean_of_distances < max_x] = -1
            mean_of_distances[mean_of_distances > max_x] = 0
            mean_of_distances = -mean_of_distances

            plot_number = 16
            argsort_ = np.argsort(mean_of_distances * var_y)[::-1][:plot_number]

            energy = np.einsum(
                "ij,i->ij", y[indices[argsort_]], (x[:, ARRAY_USE_MIDDLE] * w)[argsort_]
            )
            distances_ = distances[argsort_]
            name_ = name[indices[argsort_]]
            max_y = np.max(np.abs(energy - energy[:, [0]])) * 627.509
            print()
            print(
                np.sum(
                    (x[indices[argsort_]] - x[indices[argsort_]][[0], :, :]) ** 2,
                    axis=1,
                ),
                flush=True,
            )
            print(distances_, flush=True)
            print(np.max(np.abs((energy - energy[0]) * 627.509)), flush=True)
            print(name_, flush=True)

            INDEX_CHECK = 0
            plt.rcParams["figure.figsize"] = np.array([0.5, 0.5]) * 520 / 72
            name_energy_dict = {}

            for i in range(plot_number):
                name_energy_dict[name_[INDEX_CHECK][i]] = (
                    energy[INDEX_CHECK][i] - energy[INDEX_CHECK][0]
                ) * 627.509

            x_name_dict = {}
            x_energy_dict = {}

            for name_i, energy_i in name_energy_dict.items():
                x_ = (
                    name_i.split("_")[0]
                    + name_i.split("_")[5]
                    + name_i.split("_")[6]
                    + name_i.split("_")[7]
                )
                if x_ in x_name_dict:
                    x_name_dict[x_].append([name_i, energy_i])
                else:
                    x_name_dict[x_] = [[name_i, energy_i]]
                plt.scatter(x_, energy_i, c="b")

            for key_, value_ in x_name_dict.items():
                for j in value_:
                    plt.scatter(float(j[0].split("_")[4]), j[1], c="r")
                plt.xticks(rotation=90)
                plt.xlabel(r"$\Delta$ x ($\AA$)")
                plt.ylabel(r"Energy (kcal/mol)")
                plt.xlim(-0.575, 0.575)
                plt.xticks(np.arange(-0.5, 0.6, 0.1))
                plt.savefig(
                    f"plot/{name_plot}-{max_x_}/plot/{key_}_{len(value_)}.pdf",
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"plot/{name_plot}-{max_x_}/plot/{key_}_{len(value_)}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()
                print(key_, [i[0] for i in value_], flush=True)
            print("\n end of print", flush=True)

            color_dict = {
                "methane": "#004D40",
                "ethane": "#1A237E",
                "ethylene": "#212121",
                "acetylene": "#7B1FA2",
                "propane": "#FF6F00",
            }

            plt.rcParams["figure.figsize"] = np.array([0.5, 0.5]) * 520 / 72

            f, axes = plt.subplots(1, 1)
            axes = np.array(axes).reshape(1, 1)

            shapexy = np.shape(axes)
            inter_x = np.linspace(0.175, 0.995, shapexy[1] + 1)
            inter_y = np.linspace(0.125, 0.995, shapexy[0] + 1)
            delta_x = inter_x[1] - inter_x[0]
            delta_y = inter_y[1] - inter_y[0]

            for i in range(shapexy[0]):
                for j in range(shapexy[1]):
                    axes[i][j].set_position(
                        [
                            inter_x[j],
                            inter_y[i],
                            inter_x[j + 1] - inter_x[j],
                            inter_y[i + 1] - inter_y[i],
                        ]
                    )
                    axes[i][j].xaxis.set_tick_params(
                        direction="in", which="both", bottom=True, top=True
                    )
                    axes[i][j].yaxis.set_tick_params(
                        direction="in", which="both", left=True, right=True
                    )

                    axes[i, j].set_xlim(-max_x * 0.1, max_x * 1.1)
                    axes[i, j].set_ylim(-max_y * 1.1, max_y * 1.1)

                    if i != 0:
                        axes[i, j].set_xticks([])
                    else:
                        axes[i, j].set_xticks([0, max_x])
                        axes[i, j].set_xticklabels([0, max_x_])

            for j in range(indices.shape[1]):
                axes[0, 0].scatter(
                    distances_[INDEX_CHECK][j],
                    (energy[INDEX_CHECK][j] - energy[INDEX_CHECK][0]) * 627.509,
                    c=(color_dict[name_[INDEX_CHECK][j].split("_")[0]],),
                )

            for i, c in color_dict.items():
                axes[0, 0].scatter([-1], [-1], c=c, label=i)

            plt.xlabel("Distance of cube")
            plt.ylabel(r"$\Delta$ Energy (kcal/mol)")
            plt.legend(loc="best")

            plt.savefig(
                f"plot/{name_plot}-{max_x_}/sub_test.pdf", dpi=300, bbox_inches="tight"
            )
            plt.savefig(
                f"plot/{name_plot}-{max_x_}/sub_test.png", dpi=300, bbox_inches="tight"
            )
            plt.clf()

            plt.rcParams["figure.figsize"] = np.array([1.25, 1.25]) * 520 / 72
            f, axes = plt.subplots(4, 4)
            axes = axes.reshape(4, 4)

            shapexy = np.shape(axes)
            inter_x = np.linspace(0.025, 0.95, shapexy[1] + 1)
            inter_y = np.linspace(0.025, 0.95, shapexy[0] + 1)
            delta_x = inter_x[1] - inter_x[0]
            delta_y = inter_y[1] - inter_y[0]

            for i in range(shapexy[0]):
                for j in range(shapexy[1]):
                    axes[i][j].set_position(
                        [
                            inter_x[j],
                            inter_y[i],
                            inter_x[j + 1] - inter_x[j],
                            inter_y[i + 1] - inter_y[i],
                        ]
                    )
                    axes[i][j].xaxis.set_tick_params(
                        direction="in", which="both", bottom=True, top=True
                    )
                    axes[i][j].yaxis.set_tick_params(
                        direction="in", which="both", left=True, right=True
                    )

                    axes[i, j].set_xlim(-max_x * 0.1, max_x * 1.1)
                    axes[i, j].set_ylim(-max_y * 0.1, max_y * 1.1)

                    if i != 0:
                        axes[i, j].set_xticks([])
                    else:
                        axes[i, j].set_xticks([0, max_x])
                        axes[i, j].set_xticklabels([0, max_x_])
                    if j != 0:
                        axes[i][j].set_yticks([])

            for i in range(argsort_.shape[0]):
                axes_i, axes_j = np.unravel_index(i, (4, 4))
                for j in range(indices.shape[1]):
                    axes[axes_i, axes_j].scatter(
                        distances_[i][j],
                        np.abs(energy[i][j] - energy[i][0]) * 627.509,
                        c=(color_dict[name_[i][j].split("_")[0]],),
                    )

                axes[axes_i, axes_j].text(
                    0.01,
                    1 - 0.01,
                    f"{i}",
                    transform=axes[axes_i, axes_j].transAxes,
                    va="top",
                )

            for i, c in color_dict.items():
                axes[0, 0].scatter([-1], [-1], c=c, label=i)
            plt.legend()

            plt.savefig(
                f"plot/{name_plot}-{max_x_}/test.pdf", dpi=300, bbox_inches="tight"
            )
            plt.savefig(
                f"plot/{name_plot}-{max_x_}/test.png", dpi=300, bbox_inches="tight"
            )
            plt.clf()
