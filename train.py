from cc2cc import train_model

TRAIN_STR_DICT = [
    "methane",
    "ethane",
    "ethylene",
    "acetylene",
    # "cyclopropene",
    # "cyclopropane",
    # "allene",
    # "propyne",
    # "methyl-openshell",
]
EVAL_STR_DICT = [
    "propane",
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
    # "pentane",
    # "hexane",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_DICT, EVAL_STR_DICT)
