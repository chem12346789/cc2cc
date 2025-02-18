from cc2cc import train_model

TRAIN_STR_LIST = [
    "H",
    "H2",
    "C",
    "CH",
    "CH2_s3B1d",
    "CH2_s1A1d",
    "CH3",
    "CH4",
    "CCH",
    "C2H2",
    "C2H3",
    "C2H4",
    "C2H5",
    "C2H6",
    "C3H6_D3h",
    "C3H7",
    "C3H4_D2d",
    "C3H8",
    "C3H6_Cs",
    "C3H4_C3v",
    "C3H4_C2v",
]
EVAL_STR_LIST = [
    "isobutene",
    "trans-butane",
    "butadiene",
]

if __name__ == "__main__":
    train_model(TRAIN_STR_LIST, EVAL_STR_LIST)
