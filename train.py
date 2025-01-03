from cc2cc import train_model

TRAIN_STR_DICT = [
    "H",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Al",
    "P",
    "S",
    "Cl",
    "Si",
    "H2",
    "CO",
    "NO",
    "NH3",
    "CH4",
]
EVAL_STR_DICT = [
    "SiH4",
    "NaCl",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_DICT, EVAL_STR_DICT)
