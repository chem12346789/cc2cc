import numpy as np

MASS = {
    "h": 1.00782503207,
    "he": 4.00260325415,
    "li": 6.938,
    "be": 9.012183065,
    "b": 10.806,
    "c": 12.0096,
    "n": 14.006855,
    "o": 15.9994,
    "f": 18.998403163,
    "ne": 20.1797,
    "na": 22.989769282,
    "mg": 24.304,
    "al": 26.9815385,
    "si": 28.085,
    "p": 30.973761998,
    "s": 32.0675,
    "cl": 35.4515,
    "ar": 39.948,
    "k": 39.0983,
    "ca": 40.078,
    "sc": 44.9559085,
    "ti": 47.867,
    "v": 50.9415,
    "cr": 51.9961,
    "mn": 54.938044,
    "fe": 55.845,
    "co": 58.933194,
    "ni": 58.6934,
    "cu": 63.546,
    "zn": 65.38,
    "ga": 69.723,
    "ge": 72.630,
    "as": 74.921595,
    "se": 78.971,
    "br": 79.901,
    "kr": 83.798,
    "rb": 85.4678,
    "sr": 87.621,
    "y": 88.90584,
    "zr": 91.224,
    "nb": 92.90637,
    "mo": 95.95,
    "tc": 98.0,
    "ru": 101.07,
    "rh": 102.90550,
    "pd": 106.42,
    "ag": 107.8682,
    "cd": 112.414,
    "in": 114.818,
    "sn": 118.710,
    "sb": 121.760,
    "te": 127.603,
    "i": 126.90447,
    "xe": 131.293,
    "cs": 132.90545196,
    "ba": 137.327,
    "la": 138.904,
    "ce": 140.116,
    "pr": 140.90766,
    "nd": 144.242,
    "pm": 145.0,
    "sm": 150.362,
    "eu": 151.964,
    "gd": 157.253,
    "tb": 158.92535,
    "dy": 162.500,
    "ho": 164.93033,
    "er": 167.259,
    "tm": 168.93422,
    "yb": 173.054,
    "lu": 174.9668,
    "hf": 178.486,
    "ta": 180.94788,
    "w": 183.84,
    "re": 186.207,
    "os": 190.233,
    "ir": 192.217,
    "pt": 195.084,
    "au": 196.966569,
    "hg": 200.592,
    "tl": 204.3834,
    "pb": 207.2,
    "bi": 208.9804,
    "po": 208.9824,
    "at": 209.9871,
    "rn": 222.0176,
    "fr": 223.0197,
    "ra": 226.0254,
    "ac": 227.0278,
    "th": 232.0377,
    "pa": 231.03588,
    "u": 238.02891,
    "np": 237.0482,
    "pu": 244.0642,
    "am": 243.0614,
    "cm": 247.0704,
    "bk": 247.0703,
    "cf": 251.0796,
    "es": 252.083,
    "fm": 257.0951,
    "md": 258.0984,
    "no": 259.101,
    "lr": 262.1096,
    "rf": 267.1218,
    "db": 268.1257,
    "sg": 271.1339,
    "bh": 272.1383,
    "hs": 270.1343,
    "mt": 276.1516,
}


def get_barycenter(molecule):
    """
    Get the barycenter
    """
    barycenter = np.array([0, 0, 0], dtype=np.float64)
    mass = 0.0
    for mol in molecule:
        mass += MASS[mol[0].lower()]
        barycenter[0] += mol[1] * MASS[mol[0].lower()]
        barycenter[1] += mol[2] * MASS[mol[0].lower()]
        barycenter[2] += mol[3] * MASS[mol[0].lower()]
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


def get_inertia_moment(molecule):
    """
    Get the moment of inertia
    """
    I = np.zeros((3, 3), dtype=np.float64)
    for mol in molecule:
        I[0, 0] += MASS[mol[0].lower()] * (mol[2] ** 2 + mol[3] ** 2)
        I[1, 1] += MASS[mol[0].lower()] * (mol[1] ** 2 + mol[3] ** 2)
        I[2, 2] += MASS[mol[0].lower()] * (mol[1] ** 2 + mol[2] ** 2)
        I[0, 1] -= MASS[mol[0].lower()] * mol[1] * mol[2]
        I[0, 2] -= MASS[mol[0].lower()] * mol[1] * mol[3]
        I[1, 2] -= MASS[mol[0].lower()] * mol[2] * mol[3]
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
    return I


def rotate(molecule, rotation=None, degree=None, verbose=False):
    """
    Rotate the molecule to certain direction, center of mass is at the origin, and the (three) principal axis of charge is along the x, y, z axis.
    """
    if rotation is not None:
        verbose = True

    if verbose:
        print("Rotate the molecule to certain direction")
        print(f"before rotation {molecule}")
    # Get the barycenter
    barycenter = get_barycenter(molecule)

    for mol in molecule:
        mol[1] -= barycenter[0]
        mol[2] -= barycenter[1]
        mol[3] -= barycenter[2]

    I = get_inertia_moment(molecule)
    eig_val, eig_vec = np.linalg.eig(I)
    index1 = np.argsort(eig_val)[2]
    index2 = np.argsort(eig_val)[1]
    if np.abs((eig_val[index1] - eig_val[index2])) > 1e-12:
        list_max_eig = eig_vec[:, index1]
        rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [0, 0, 1])
        # rotate the molecule
        for mol in molecule:
            x_array = np.array(mol[1:])
            x_array = rotation_matrix @ x_array
            mol[1] = x_array[0]
            mol[2] = x_array[1]
            mol[3] = x_array[2]

        I = get_inertia_moment(molecule)
        eig_val, eig_vec = np.linalg.eig(I)
        index2 = np.argsort(eig_val)[1]
        list_max_eig = eig_vec[:, index2]
        rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [0, 1, 0])
        for mol in molecule:
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
            for mol in molecule:
                x_array = np.array(mol[1:])
                x_array = rotation_matrix @ x_array
                mol[1] = x_array[0]
                mol[2] = x_array[1]
                mol[3] = x_array[2]

        for mol in molecule:
            x_array = np.array(mol[1:])
            index_ = np.argsort(np.abs(x_array))[-1]
            mol[1] = x_array[index_]

    I = get_inertia_moment(molecule)
    eig_val, eig_vec = np.linalg.eig(I)
    index2 = np.argsort(eig_val)[0]
    list_max_eig = eig_vec[:, index2]
    rotation_matrix = rotation_matrix_from_vectors(list_max_eig, [1, 0, 0])
    for mol in molecule:
        x_array = np.array(mol[1:])
        x_array = rotation_matrix @ x_array
        mol[1] = x_array[0]
        mol[2] = x_array[1]
        mol[3] = x_array[2]
    if verbose:
        print(f"after rotation {molecule}")

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

        for mol in molecule:
            x_array = np.array(mol[1:])
            x_array = rotation @ x_array
            mol[1] = x_array[0]
            mol[2] = x_array[1]
            mol[3] = x_array[2]
        if verbose:
            print(f"after test rotation {molecule}")
        return rotation
    return np.eye(3)
