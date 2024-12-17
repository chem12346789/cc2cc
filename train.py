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
    "C2H6",
    "NaCl",
    "CH3CN",
    "C2H3",
    "C3H9C",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_DICT, EVAL_STR_DICT)
