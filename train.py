from cc2cc import train_model

TRAIN_STR_LIST = [
    "c",
    "c2",
    "c2h2",
    "c2h4",
    "c2h6",
    "cch",
    "ch",
    "ch2-sing",
    "ch2-trip",
    "ch2c",
    "ch2ch",
    "ch3",
    "ch4",
    "h",
    "h2",
]
EVAL_STR_LIST = [
    "allene",
    "propane",
    "propene",
    "propyne",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_LIST, EVAL_STR_LIST)
