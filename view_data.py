import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
import faiss


def plot():
    data = np.load("data/save/train_test-0-0-0-0.npz")

    x = np.array(data["x"], dtype=np.float32)
    y_all = np.array(data["y"], dtype=np.float32)
    y = y_all[:, 0]
    w = np.array(data["w"], dtype=np.float32)

    (_, dim_sample) = x.shape
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, dim_sample, flat_config)
    index.add(x)

    b = x.copy()
    min_number = 100
    distances, indices = index.search(b, min_number)
    name = data["name"]

    var_y = np.var(np.einsum("ij,i->ij", y[indices], x[:, 0] * w), axis=1)

    mean_of_distances = np.max(distances, axis=1)
    mean_of_distances[mean_of_distances < 0.0001] = -1
    mean_of_distances[mean_of_distances > 0.0001] = 0
    mean_of_distances = -mean_of_distances

    argsort_ = np.argsort(mean_of_distances * var_y)[::-1][:100]

    energy = np.einsum("ij,i->ij", y[indices[argsort_]], (x[:, 0] * w)[argsort_])
    distances = distances[argsort_]
    name = name[indices[argsort_]]
    x = x[indices[argsort_]]
    y = y[indices[argsort_]]
    w = w[indices[argsort_]]
    y_all = y_all[indices[argsort_]]

    color_dict = {
        "methane": "#004D40",
        "ethane": "#1A237E",
        "ethylene": "#212121",
        "acetylene": "#7B1FA2",
    }

    plt.rcParams["figure.figsize"] = np.array([3, 3]) * 520 / 72

    f, axes = plt.subplots(10, 10)
    axes = axes.reshape(10, 10)

    begin_y = 0.025
    end_y = 0.95
    int_y = 0.0
    begin_x = 0.025
    end_x = 0.95
    int_x = 0.0
    end_x += int_x
    end_y += int_y

    shapexy = np.shape(axes)
    inter_x = np.linspace(begin_x, end_x, shapexy[1] + 1)
    inter_y = np.linspace(begin_y, end_y, shapexy[0] + 1)

    for i in range(shapexy[0]):
        for j in range(shapexy[1]):
            axes[i][j].set_position(
                [
                    inter_x[j],
                    inter_y[i],
                    inter_x[j + 1] - inter_x[j] - int_x,
                    inter_y[i + 1] - inter_y[i] - int_y,
                ]
            )
            axes[i][j].xaxis.set_tick_params(
                direction="in", which="both", bottom=True, top=True
            )
            axes[i][j].yaxis.set_tick_params(
                direction="in", which="both", left=True, right=True
            )

            axes[i, j].set_xlim(-0.00001, 0.00011)
            axes[i, j].set_ylim(-0.002, 0.022)

            if i != 0:
                axes[i, j].set_xticks([])
            else:
                axes[i, j].set_xticks([0, 0.0001])
                axes[i, j].set_xticklabels([0, 0.0001])
            if j != 0:
                axes[i][j].set_yticks([])
            else:
                axes[i][j].set_yticks([0, 0.01, 0.02])
                axes[i][j].set_yticklabels([0, 0.01, 0.02])

    for i in range(argsort_.shape[0]):
        axes_i, axes_j = np.unravel_index(i, (10, 10))
        for j in range(indices.shape[1]):
            axes[axes_i, axes_j].scatter(
                distances[i][j],
                np.abs(energy[i][j] - energy[i][0]) * 627.509,
                c=(color_dict[name[i][j].split("_")[0]],),
            )

        axes[axes_i, axes_j].text(
            0.01,
            1 - 0.01,
            f"{i}",
            transform=axes[axes_i, axes_j].transAxes,
            va="top",
        )
    plt.savefig("test.pdf", dpi=300)
    plt.clf()
    return x, y, y_all, w, energy, distances, name


x, y, y_all, w, energy, distances, name = plot()

np.savez_compressed(
    "data/save/test.npz",
    x=x,
    y=y,
    y_all=y_all,
    w=w,
    energy=energy,
    distances=distances,
    name=name,
)

index_check = 23
print(np.sum((x[index_check] - x[index_check][0]) ** 2, axis=1))
print(np.max(distances[index_check]))
print(np.max(np.abs((energy[index_check] - energy[index_check][0]) * 627.509)))
print(name[index_check])

plt.rcParams["figure.figsize"] = np.array([0.5, 0.5]) * 520 / 72
name_energy_dict = {}

for i in range(energy.shape[1]):
    name_energy_dict[name[index_check][i]] = (
        energy[index_check][i] - energy[index_check][0]
    ) * 627.509

name_dict = {
    "methane": 0,
    "ethane": 1,
    "ethylene": 2,
    "acetylene": 3,
}

x_name_dict = {}
x_energy_dict = {}

for name_i, energy_i in name_energy_dict.items():
    x_ = name_dict[name_i.split("_")[0]] * 11 + float(name_i.split("_")[5])
    if x_ in x_name_dict:
        x_name_dict[x_].append([name_i, energy_i])
    else:
        x_name_dict[x_] = [[name_i, energy_i]]
    plt.scatter(x_, energy_i, c="b")


Path("./plot").mkdir(parents=True, exist_ok=True)

for key_, value_ in x_name_dict.items():
    for j in value_:
        plt.scatter(float(j[0].split("_")[4]), j[1], c="r")
    plt.xticks(rotation=90)
    plt.xlabel(r"$\Delta$ x ($\AA$)")
    plt.ylabel("Energy (kcal/mol)")
    plt.xlim(-0.575, 0.575)
    plt.xticks(np.arange(-0.5, 0.6, 0.1))
    plt.savefig(f"./plot/{key_}.pdf", bbox_inches="tight")
    plt.savefig(f"./plot/{key_}.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(key_, [i[0] for i in value_])

color_dict = {
    "methane": "#004D40",
    "ethane": "#1A237E",
    "ethylene": "#212121",
    "acetylene": "#7B1FA2",
}

plt.rcParams["figure.figsize"] = np.array([0.5, 0.5]) * 520 / 72

f, axes = plt.subplots(1, 1)
axes = np.array(axes).reshape(1, 1)

begin_y = 0.125
end_y = 0.995
int_y = 0.0
begin_x = 0.175
end_x = 0.995
int_x = 0.0
end_x += int_x
end_y += int_y

shapexy = np.shape(axes)
inter_x = np.linspace(begin_x, end_x, shapexy[1] + 1)
inter_y = np.linspace(begin_y, end_y, shapexy[0] + 1)

delta_x = inter_x[1] - inter_x[0] - int_x
delta_y = inter_y[1] - inter_y[0] - int_y

for i in range(shapexy[0]):
    for j in range(shapexy[1]):
        axes[i][j].set_position(
            [
                inter_x[j],
                inter_y[i],
                inter_x[j + 1] - inter_x[j] - int_x,
                inter_y[i + 1] - inter_y[i] - int_y,
            ]
        )
        axes[i][j].xaxis.set_tick_params(
            direction="in", which="both", bottom=True, top=True
        )
        axes[i][j].yaxis.set_tick_params(
            direction="in", which="both", left=True, right=True
        )

        axes[i, j].set_xlim(-0.000003, 0.000033)
        # axes[i, j].set_ylim(-0.022, 0.002)

        if i != 0:
            axes[i, j].set_xticks([])
        else:
            axes[i, j].set_xticks([0, 0.00003])
            axes[i, j].set_xticklabels([0, 0.00003])
        # if j != 0:
        #     axes[i][j].set_yticks([])
        # else:
        #     axes[i][j].set_yticks([-0.03, -0.02, -0.01, 0])
        #     axes[i][j].set_yticklabels([-0.03, -0.02, -0.01, 0])
        #     axes[i][j].set_yticks([0, 0.01, 0.02, 0.03])
        #     axes[i][j].set_yticklabels([0, 0.01, 0.02, 0.03])

for j in range(energy.shape[1]):
    axes[0, 0].scatter(
        distances[index_check][j],
        (energy[index_check][j] - energy[index_check][0]) * 627.509,
        c=(color_dict[name[index_check][j].split("_")[0]],),
    )

axes[0, 0].scatter(distances[index_check][0], 0, c="r", marker="x")

for i, c in color_dict.items():
    axes[0, 0].scatter([-1], [-1], c=c, label=i)

plt.xlabel("Distance of cube")
plt.ylabel(r"$\Delta$ Energy (kcal/mol)")
plt.legend()

plt.savefig("sub_test.pdf", dpi=300, bbox_inches="tight")
plt.savefig("sub_test.png", dpi=300, bbox_inches="tight")
plt.clf()
