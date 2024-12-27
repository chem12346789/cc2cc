import argparse
import copy
from itertools import product
from pathlib import Path

from matplotlib import pyplot as plt

import numpy as np

import faiss

from sklearn.kernel_ridge import KernelRidge

from krr import add_args, load_data, hash_value
from cc2cc.utils import CUBE_SIZE, DATA_PATH

faiss.cvar.distance_compute_blas_threshold = 8000000

parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
args = add_args(parser)

print("gamma:", args.gamma)
print("alpha:", args.alpha)
print("kernel:", args.kernel, flush=True)

view_dict = {
    "cc": "exc_cc_grids",
    # "b3lyp": "exc_over_dm_b3lyp_grids",
    # "mrks": "exc_over_dm_mrks_grids",
}

(
    input_dict,
    output_dict,
    weights_dict,
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
name_all = {}
dual_coef = {}
x_fit = {}

# center_piont = []
# hashtable = append(np.linspace(-1, 0, 11)[:-1], center_piont, np.linspace(0, 1, 11)[1:])
hashtable = np.linspace(-1e10, 1e10, 2)
print(hashtable)
shape_matrix = (20, 302, 75)

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
            name_all[index_round] = [f"{key}_{iatm}_{irad}_{iang}"]
        else:
            x_all[index_round].append(input_dict[key][index_])
            y_all[index_round].append(output_dict[key][index_])
            w_all[index_round].append(weights_dict[key][index_])
            name_all[index_round].append(f"{key}_{iatm}_{irad}_{iang}")

x_keys = list(x_all.keys())
for index_ in np.sort(list(x_all.keys())):
    x_all[index_] = np.array(x_all[index_])
    y_all[index_] = np.array(y_all[index_])
    w_all[index_] = np.array(w_all[index_])
    name_all[index_] = np.array(name_all[index_])

train_error_sum = 0
energy_correct_sum = 0

color_dict_atom = {
    "C": "#283593",
    "H": "#7CB342",
    "O": "#D32F2F",
}

PLOT_NUMBER = 16
SEARCH_NUMBER = 100


# for index_ in []:
for index_ in x_keys:
    if index_ not in [
        "0",
        # "1_1_1_1",
        # "2_2_2_2",
        # f"{len(hashtable)//2}_{len(hashtable)//2}_{len(hashtable)//2}_{len(hashtable)//2}",
    ]:
        continue
    print(f"Index: {index_}", flush=True)

    for name_plot_i, name_plot in enumerate(view_dict.keys()):
        name_plot = name_plot + "-" + index_
        x = x_all[index_]
        y = y_all[index_]
        w = w_all[index_]
        name = name_all[index_]
        y = y[:, name_plot_i]

        name_plot = f"{name_plot}-{x.shape[1]}"
        print(x.shape, flush=True)

        (num_sample, dim_sample) = x.shape

        # res = faiss.StandardGpuResources()
        # flat_config = faiss.GpuIndexFlatConfig()
        # flat_config.device = 0
        # index = faiss.GpuIndexFlatL2(res, dim_sample, flat_config)

        index = faiss.IndexFlatL2(dim_sample)
        index.add(x)
        b = x.copy()
        distances, indices = index.search(b, SEARCH_NUMBER)

        for if_kcal, max_x_ in product(
            [
                False,
                # True,
            ],
            [
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
                1e-7,
            ],
        ):
            print("\n begin of print", flush=True)
            if if_kcal:
                plot_name = f"{name_plot}-{max_x_}-kcal"
            else:
                plot_name = f"{name_plot}-{max_x_}-au"
            plot_path = Path(f"plot-{Path(DATA_PATH).parts[-1]}/{plot_name}")
            print("Save to", plot_path, flush=True)
            (plot_path / "plot").mkdir(parents=True, exist_ok=True)
            max_x = max_x_**2 * x.shape[1]

            mean_of_distances = np.max(distances, axis=1)
            mean_of_distances[mean_of_distances < max_x] = -1
            mean_of_distances[mean_of_distances > max_x] = 0
            mean_of_distances = -mean_of_distances

            if if_kcal:
                energy = np.einsum(
                    "ij,i->ij",
                    y[indices],
                    x[:, CUBE_SIZE**3] * w,
                )
                var_energy = np.var(energy, axis=1)
                argsort_ = np.argsort(mean_of_distances * var_energy)[::-1][
                    :PLOT_NUMBER
                ]
                energy = energy[argsort_] * 627.509
            else:
                var_energy = np.var(y[indices], axis=1)
                argsort_ = np.argsort(mean_of_distances * var_energy)[::-1][
                    :PLOT_NUMBER
                ]
                energy = y[indices[argsort_]] * 627.509

            distances_ = distances[argsort_]
            name_ = name[indices[argsort_]]
            x_ = x[indices[argsort_]]
            print(np.sum((x_ - x_[:, [0], :]) ** 2, axis=2))
            print(distances_)
            print(x_.shape)
            print(energy.shape, distances_.shape, name_.shape, flush=True)
            print(y.shape, distances.shape, name.shape, flush=True)
            max_y = np.max(np.abs(energy - energy[:, [0]]))

            INDEX_CHECK = 0
            name_energy_dict = {}
            for i in range(SEARCH_NUMBER):
                if if_kcal:
                    name_energy_dict[name_[INDEX_CHECK][i]] = energy[INDEX_CHECK][i]
                else:
                    name_energy_dict[name_[INDEX_CHECK][i]] = energy[INDEX_CHECK][i]

            x_name_dict = {}
            for name_i, energy_i in name_energy_dict.items():
                x_ = f"{name_i.split("_")[0]}_{name_i.split("_")[5]}_{name_i.split("_")[6]}_{name_i.split("_")[7]}"
                if x_ in x_name_dict:
                    x_name_dict[x_].append([name_i, energy_i])
                else:
                    x_name_dict[x_] = [[name_i, energy_i]]

            plt.rcParams["figure.figsize"] = np.array([0.95, 0.5]) * 520 / 72
            for key_, value_ in x_name_dict.items():
                if if_kcal:
                    for j in value_:
                        print(f"{j[0]:20} {j[1]:10.5e}", flush=True)
                    print("kcal/mol")
                else:
                    for j in value_:
                        print(f"{j[0]:20} {j[1]:10.5e}", flush=True)
                    print("au")
                f, axes = plt.subplots(2, 1)
                axes[1].remove()
                # 3d plot
                axes[1] = plt.axes(projection="3d")
                axes = np.array(axes).reshape(1, 2)

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

                for j in value_:
                    axes[0, 0].scatter(float(j[0].split("_")[4]), j[1], c="r")
                axes[0, 0].tick_params(axis="x", which="both", rotation=70)
                axes[0, 0].set_xlabel(r"$\Delta$ x ($\AA$)")
                if if_kcal:
                    axes[0, 0].set_ylabel(r"Energy (kcal/mol)")
                else:
                    axes[0, 0].set_ylabel(r"Energy (au)")
                axes[0, 0].set_xlim(-0.575, 0.575)
                axes[0, 0].set_xticks(np.arange(-0.5, 0.6, 0.1))

                plt.savefig(
                    plot_path / f"plot/{key_}_{len(value_)}.pdf",
                    bbox_inches="tight",
                )
                plt.savefig(
                    plot_path / f"plot/{key_}_{len(value_)}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()
            print("\n end of print", flush=True)

            color_dict = {
                "H": "#7CB342",
                "Li": "#FFD600",
                "Be": "#FF6F00",
                "B": "#FF1744",
                "C": "#283593",
                "N": "#1976D2",
                "O": "#D32F2F",
                "F": "#388E3C",
                "Na": "#FFD600",
                "Al": "#FF6F00",
                "P": "#FF1744",
                "S": "#1976D2",
                "Cl": "#388E3C",
                "Si": "#283593",
                "H2": "#7CB342",
                "CO": "#FF1744",
                "NO": "#1976D2",
                "NH3": "#388E3C",
                "CH4": "#283593",
                "C2H6": "#FF1744",
                "CH3CN": "#1976D2",
                "C2H3": "#388E3C",
                "C3H9C": "#283593",
                "NaCl": "#FFD600",
                "SiH4": "#FF6F00",
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
                    energy[INDEX_CHECK][j] - energy[INDEX_CHECK][0],
                    c=(color_dict[name_[INDEX_CHECK][j].split("_")[0]],),
                )

            for i, c in color_dict.items():
                axes[0, 0].scatter([-1], [-1], c=c, label=i)

            if if_kcal:
                plt.ylabel(r"Energy (kcal/mol)")
            else:
                plt.ylabel(r"Energy (au)")
            plt.legend(loc="best")
            plt.xlabel("Distance of cube")
            plt.savefig(plot_path / "sub_test.pdf", dpi=300, bbox_inches="tight")
            plt.savefig(plot_path / "sub_test.png", dpi=300, bbox_inches="tight")
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
                    axes[i, j].set_ylim(-max_y * 1.1, max_y * 1.1)

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
                        energy[i][j] - energy[i][0],
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
                axes[-1, -1].scatter([-1], [-1], c=c, label=i)
            axes[-1, -1].legend(loc="best")

            plt.savefig(plot_path / "test.pdf", dpi=300, bbox_inches="tight")
            plt.savefig(plot_path / "test.png", dpi=300, bbox_inches="tight")
            plt.clf()
