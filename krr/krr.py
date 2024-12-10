import numpy as np

from numba import njit, prange

from sklearn.kernel_ridge import KernelRidge

from cc2cc.utils import AU2KCALMOL, CUBE_SIZE


def cut_off(x, hashtable):
    """
    Cut off the value.
    """
    return np.sum(hashtable < x) - len(hashtable) // 2


def hash_value(input_, hashtable):
    """
    Hash the value.
    hashtable:
        shape [:, 3] (float1, float2, int1)
        meaning:
            float1: begin of the key range
            float2: end of the key range
            int1: value of the key
    """
    index_round0 = cut_off(input_[CUBE_SIZE **3 + CUBE_SIZE **3 * 0], hashtable)
    return f"{index_round0}"


@njit(parallel=True)
def get_kernel(x1, x2, gamma, kernel_type):
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


class KernelRidgeModified(KernelRidge):
    """
    Kernel Ridge Regression with precomputed kernel.
    """

    def __init__(self, alpha=1, gamma=1, kernel_type="rbf"):
        super().__init__(
            alpha=alpha,
            gamma=gamma,
            kernel="precomputed",
        )
        self.alpha = alpha
        self.gamma = gamma
        self.kernel_type = kernel_type
        self.x_fit = None
        self.kernel_matrix = None

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


def evaluate(krr, x_train, y_train, w_train, x_all, y_all, w_all):
    """
    Evaluate the model.
    """
    print("Krr perdict:")
    train_error = AU2KCALMOL * np.sum(
        (np.abs(y_train - krr.predict_data(x_train)) * w_train * x_train[:, CUBE_SIZE **3])
    )
    print(f"train, {train_error} KCAL/MOL", flush=True)
    error_krr = AU2KCALMOL * np.sum(
        (np.abs(y_all - krr.predict_data(x_all)) * w_all * x_all[:, CUBE_SIZE **3])
    )
    print(f"test, {error_krr} KCAL/MOL", flush=True)

    print("B3lyp perdict:")
    b3lyp_error = AU2KCALMOL * np.sum(np.abs(y_train * w_train * x_train[:, CUBE_SIZE **3]))
    print("train", b3lyp_error, "KCAL/MOL", flush=True)
    b3lyp_error = AU2KCALMOL * np.sum(np.abs(y_all * w_all * x_all[:, CUBE_SIZE **3]))
    print("test", b3lyp_error, "KCAL/MOL", flush=True)
    print("End of evaluate.\n", flush=True)
    return np.abs(error_krr), np.abs(train_error)


def add_data(
    krr,
    x_train,
    y_train,
    w_train,
    x_test,
    y_test,
    w_test,
    x_fitted,
    y_fitted,
    w_fitted,
):
    """
    Add data which has large error.
    """
    print("Add data:", flush=True)
    if len(x_test) == 0:
        print("No data to add.")
        return (
            x_train,
            y_train,
            w_train,
            x_test,
            y_test,
            w_test,
            x_fitted,
            y_fitted,
            w_fitted,
        )
    error_test = (y_test - krr.predict_data(x_test)) * w_test * x_test[:, CUBE_SIZE **3]

    index_add_test = (
        np.array([True] * len(error_test))
        if len(error_test) < 51
        else (error_test**2 > np.sort(error_test**2, axis=0)[-51])
    )
    print(
        np.array2string(
            AU2KCALMOL * error_test[index_add_test],
            formatter={"float_kind": lambda x: f"{x:.6f}"},
        ),
        np.array2string(
            AU2KCALMOL * (y_test - krr.predict_data(x_test))[index_add_test],
            formatter={"float_kind": lambda x: f"{x:.6f}"},
        ),
        np.array2string(
            np.max(krr.kernel_matrix[index_add_test], axis=1),
            formatter={"float_kind": lambda x: f"{x:.6f}"},
        ),
    )
    x_train = np.concatenate([x_train, x_test[index_add_test]])
    y_train = np.concatenate([y_train, y_test[index_add_test]])
    w_train = np.concatenate([w_train, w_test[index_add_test]])
    x_test = x_test[~index_add_test]
    y_test = y_test[~index_add_test]
    w_test = w_test[~index_add_test]

    if len(x_test) == 0:
        print("No data to add.")
        return (
            x_train,
            y_train,
            w_train,
            x_test,
            y_test,
            w_test,
            x_fitted,
            y_fitted,
            w_fitted,
        )
    error_test = (y_test - krr.predict_data(x_test)) * w_test * x_test[:, CUBE_SIZE **3]
    index_add_fitted = (
        np.array([True] * len(error_test))
        if len(error_test) < 51
        else (error_test**2 < 1e-10**2)
    )
    x_fitted = np.concatenate([x_fitted, x_test[index_add_fitted]])
    y_fitted = np.concatenate([y_fitted, y_test[index_add_fitted]])
    w_fitted = np.concatenate([w_fitted, w_test[index_add_fitted]])
    x_test = x_test[~index_add_fitted]
    y_test = y_test[~index_add_fitted]
    w_test = w_test[~index_add_fitted]

    print(
        "Length of x_train:",
        len(x_train),
        "Length of x_test:",
        len(x_test),
        "Length of x_fitted:",
        len(x_fitted),
        flush=True,
    )
    print("End of add data.\n", flush=True)

    return (
        x_train,
        y_train,
        w_train,
        x_test,
        y_test,
        w_test,
        x_fitted,
        y_fitted,
        w_fitted,
    )
