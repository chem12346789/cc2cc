from dft2cc import train_model

TRAIN_STR_DICT = [
    "methane",
    # "ethylene",
    # "acetylene",
    # "cyclopropene",
    # "cyclopropane",
    # "allene",
    # "propyne",
    # "propane",
    # "propylene",
    # "isobutane",
    # "cyclobutane",
    # "butane",
    # "butyne",
    # "butadiene",
    # "bicyclobutane",
    # "cyclopropylmethyl",
    # "cyclopentane",
    # "spiropentane",
    # "benzene",
    # "neopentane",
    # "isopentane",
    # "methyl-openshell",
]
EVAL_STR_DICT = [
    "ethane",
    # "pentane",
    # "hexane",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_DICT, EVAL_STR_DICT)
