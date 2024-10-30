import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from cc2cc.utils import AU2KCALMOL

from krr import load_data, evaluate, add_data, add_args, hash_value
from krr import KernelRidgeModified


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


krr = KernelRidgeModified(
    alpha=args.alpha,
    gamma=args.gamma,
    kernel_type="rbf",
)

x_all = {}
y_all = {}
w_all = {}
# x_neighborhood = {}
# y_neighborhood = {}
# w_neighborhood = {}
name_all = {}
dual_coef = {}
model_para = {
    "x_fit": {},
    "alpha": {},
    "gamma": {},
    "kernel_type": {},
}

for key in keys_list:
    print(f"Key: {key}", flush=True)
    for index_ in range(len(input_dict[key])):
        if index_ % (len(input_dict[key]) / 10) == 0:
            print(
                f"Processing: {index_ / (len(input_dict[key]) / 10) * 10:.1f}%",
                flush=True,
            )

        index_round = hash_value(input_dict[key][index_])

        if index_round not in x_all:
            x_all[index_round] = [input_dict[key][index_]]
            y_all[index_round] = [output_dict[key][index_]]
            w_all[index_round] = [weights_dict[key][index_]]
        else:
            x_all[index_round].append(input_dict[key][index_])
            y_all[index_round].append(output_dict[key][index_])
            w_all[index_round].append(weights_dict[key][index_])

x_keys = list(x_all.keys())
for index_ in x_keys:
    x_all[index_] = np.array(x_all[index_])
    y_all[index_] = np.array(y_all[index_])
    w_all[index_] = np.array(w_all[index_])

train_error_sum = 0
energy_correct_sum = 0

for index_ in x_keys:
    print(f"Round {index_}", flush=True)

    if (
        int(index_.split("-")[0]) == 0
        and int(index_.split("-")[1]) == 0
        and int(index_.split("-")[2]) == 0
        and int(index_.split("-")[3]) == 0
    ):
        krr.alpha = args.alpha
        krr.gamma = args.gamma
        krr.kernel_type = "rbf"
        ERROR_KRR_CONVERGE = 0.5
    elif int(index_.split("-")[0]) == 0:
        krr.alpha = args.alpha
        krr.gamma = 1
        krr.kernel_type = "rbf"
        ERROR_KRR_CONVERGE = 1e-2
    else:
        krr.alpha = args.alpha
        krr.gamma = 0.1
        krr.kernel_type = "rbf"
        ERROR_KRR_CONVERGE = 1e-2

    if len(x_all[index_]) < 500:
        krr.fit_data(x_all[index_], y_all[index_])
    else:
        x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
            x_all[index_],
            y_all[index_],
            w_all[index_],
            train_size=50,
            random_state=42,
        )

        CONVERGE_STEP = 0
        for i_step in range(250):
            print(f"Step {i_step}:", flush=True)
            krr.fit_data(x_train, y_train)
            error_krr, train_error = evaluate(
                krr,
                x_train,
                y_train,
                w_train,
                x_all[index_],
                y_all[index_],
                w_all[index_],
            )
            print(error_krr, train_error, error_krr < 2 * train_error, flush=True)
            if error_krr < ERROR_KRR_CONVERGE or error_krr < 2 * train_error:
                print(f"Error is small: {error_krr}", flush=True)
                CONVERGE_STEP += 1
                if CONVERGE_STEP == 1:
                    print("Converge.")
                    break
            x_train, y_train, w_train, x_test, y_test, w_test = add_data(
                krr, x_train, y_train, w_train, x_test, y_test, w_test
            )

    train_error = np.sum(
        np.abs(
            (y_all[index_] - krr.predict_data(x_all[index_]))
            * w_all[index_]
            * x_all[index_][:, 0]
        )
    )
    energy_correct = np.sum(np.abs(y_all[index_] * w_all[index_] * x_all[index_][:, 0]))

    train_error_sum += train_error
    energy_correct_sum += energy_correct

    if np.abs(AU2KCALMOL * train_error) > 0.001:
        print(
            f"Round {index_}\n",
            f"Length of x_all: {len(x_all[index_])}\n",
            f"Train error: {AU2KCALMOL * train_error} KCAL/MOL\n",
            f"Energy correct: {AU2KCALMOL * energy_correct} KCAL/MOL\n",
            flush=True,
        )

    dual_coef[index_] = krr.dual_coef_
    model_para["x_fit"][index_] = krr.x_fit
    model_para["alpha"][index_] = krr.alpha
    model_para["gamma"][index_] = krr.gamma
    model_para["kernel_type"][index_] = krr.kernel_type

# save the model
random_number = np.random.randint(10000)
print(random_number)
np.savez_compressed(
    f"data/save/dual_coef-final-{random_number}.npz",
    dual_coef=dual_coef,
    model_para=model_para,
)

print(
    f"Energy correct sum: {AU2KCALMOL * energy_correct_sum} KCAL/MOL",
    f"Train error sum: {AU2KCALMOL * train_error_sum} KCAL/MOL",
    flush=True,
)
