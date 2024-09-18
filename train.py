from dft2cc import train_model

TRAIN_STR_DICT = [
    "methane",
    "ethane",
    "ethylene",
    "acetylene",
    "cyclopropene",
    "cyclopropane",
    "propyne",
    "allene",
    "propane",
    "propylene",
    # "methyl-openshell",
]
EVAL_STR_DICT = [
    "butane",
    "butyne",
    "isobutane",
    "bicyclobutane",
    "cyclobutane",
    "butadiene",
    "cyclopropylmethyl",
    "cyclopentane",
    "spiropentane",
    "neopentane",
    "isopentane",
    "pentane",
    "benzene",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_DICT, EVAL_STR_DICT)
