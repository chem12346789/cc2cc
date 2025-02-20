import numpy as np

MASS = {
    "H": 1.00782503207,
    "He": 4.00260325415,
    "Li": 6.938,
    "Be": 9.012183065,
    "B": 10.806,
    "C": 12.0096,
    "N": 14.006855,
    "O": 15.9994,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.989769282,
    "Mg": 24.304,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.0675,
    "Cl": 35.4515,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.9559085,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.901,
    "Kr": 83.798,
}


def get_barycenter(molecular):
    """
    Get the barycenter
    """
    barycenter = np.array([0, 0, 0], dtype=np.float64)
    mass = 0.0
    for mol in molecular:
        mass += MASS[mol[0]]
        barycenter[0] += mol[1] * MASS[mol[0]]
        barycenter[1] += mol[2] * MASS[mol[0]]
        barycenter[2] += mol[3] * MASS[mol[0]]
    return barycenter / mass


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if np.abs(s) < 1e-12:
        return np.eye(3)
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def get_inertia_moment(molecular):
    """
    Get the moment of inertia
    """
    I = np.zeros((3, 3), dtype=np.float64)
    for mol in molecular:
        I[0, 0] += MASS[mol[0]] * (mol[2] ** 2 + mol[3] ** 2)
        I[1, 1] += MASS[mol[0]] * (mol[1] ** 2 + mol[3] ** 2)
        I[2, 2] += MASS[mol[0]] * (mol[1] ** 2 + mol[2] ** 2)
        I[0, 1] -= MASS[mol[0]] * mol[1] * mol[2]
        I[0, 2] -= MASS[mol[0]] * mol[1] * mol[3]
        I[1, 2] -= MASS[mol[0]] * mol[2] * mol[3]
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
    return I


def rotate(molecular, rotation=None, degree=None, verbose=False):
    """
    Rotate the molecular to certain direction, center of mass is at the origin, and the (three) principal axis of charge is along the x, y, z axis.
    """
    if rotation is not None:
        verbose = True

    if verbose:
        print("Rotate the molecular to certain direction")
        print(f"before rotation {molecular}")
    # Get the barycenter
    barycenter = get_barycenter(molecular)

    for mol in molecular:
        mol[1] -= barycenter[0]
        mol[2] -= barycenter[1]
        mol[3] -= barycenter[2]

    I = get_inertia_moment(molecular)
    eig_val, eig_vec = np.linalg.eig(I)
    index1 = np.argsort(eig_val)[2]
    index2 = np.argsort(eig_val)[1]
    if np.abs((eig_val[index1] - eig_val[index2])) > 1e-12:
        list_max_eig = eig_vec[:, index1]
        rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [0, 0, 1])
        # rotate the molecule
        for mol in molecular:
            x_array = np.array(mol[1:])
            x_array = rotation_matrix @ x_array
            mol[1] = x_array[0]
            mol[2] = x_array[1]
            mol[3] = x_array[2]

        I = get_inertia_moment(molecular)
        eig_val, eig_vec = np.linalg.eig(I)
        index2 = np.argsort(eig_val)[1]
        list_max_eig = eig_vec[:, index2]
        rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [0, 1, 0])
        for mol in molecular:
            x_array = np.array(mol[1:])
            x_array = rotation_matrix @ x_array
            mol[1] = x_array[0]
            mol[2] = x_array[1]
            mol[3] = x_array[2]
    else:
        index1 = np.argsort(eig_val)[0]
        list_max_eig = eig_vec[:, index1]

        if (np.sqrt(list_max_eig[1] ** 2 + list_max_eig[2] ** 2)) > 1e-6:
            rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [1, 0, 0])

            # rotate the molecule
            for mol in molecular:
                x_array = np.array(mol[1:])
                x_array = rotation_matrix @ x_array
                mol[1] = x_array[0]
                mol[2] = x_array[1]
                mol[3] = x_array[2]

        for mol in molecular:
            x_array = np.array(mol[1:])
            index_ = np.argsort(np.abs(x_array))[-1]
            mol[1] = x_array[index_]

    I = get_inertia_moment(molecular)
    eig_val, eig_vec = np.linalg.eig(I)
    index2 = np.argsort(eig_val)[0]
    list_max_eig = eig_vec[:, index2]
    rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [1, 0, 0])
    for mol in molecular:
        x_array = np.array(mol[1:])
        x_array = rotation_matrix @ x_array
        mol[1] = x_array[0]
        mol[2] = x_array[1]
        mol[3] = x_array[2]
    if verbose:
        print(f"after rotation {molecular}")

    if rotation is not None:
        if degree is None:
            degree = np.pi / 2
        if rotation == "x":
            rotation = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(degree), -np.sin(degree)],
                    [0, np.sin(degree), np.cos(degree)],
                ]
            )
        elif rotation == "y":
            rotation = np.array(
                [
                    [np.cos(degree), 0, np.sin(degree)],
                    [0, 1, 0],
                    [-np.sin(degree), 0, np.cos(degree)],
                ]
            )
        elif rotation == "z":
            rotation = np.array(
                [
                    [np.cos(degree), -np.sin(degree), 0],
                    [np.sin(degree), np.cos(degree), 0],
                    [0, 0, 1],
                ]
            )
        else:
            if verbose:
                print("random rotation")
            random_axis = np.random.uniform(-1, 1, 3)
            random_axis = random_axis / np.linalg.norm(random_axis)
            rotation = rotation_matrix_from_vectors([0, 0, 1], random_axis)

        for mol in molecular:
            x_array = np.array(mol[1:])
            x_array = rotation @ x_array
            mol[1] = x_array[0]
            mol[2] = x_array[1]
            mol[3] = x_array[2]
        if verbose:
            print(f"after test rotation {molecular}")
        return rotation
    return np.eye(3)
