import argparse

from cc2cc.utils import add_args, print_computer_info
from cc2cc.utils.parser import gen_name_args
from cc2cc.train_model import train_model

train_str_list = [
    "molecule0-W4_11",
    # "ADDON_As",
    # "ADDON_Ge",
    # "ADDON_Se",
    # "ADDON_Te",
    # "ADDON_I",
    # "ADDON_Bi",
    # "ADDON_Pb",
    # "ADDON_Sb",
    "AHB21-1A",
    "AHB21-4A",
    "ALK8-li+",
    "ALK8-na+",
    "ALKBDE10-ca",
    "ALKBDE10-k",
    "ALKBDE10-li",
    "ALKBDE10-mg",
    "ALKBDE10-na",
    "CHB6-24A",
    "DIPCS10-be_2+",
    "DIPCS10-mg_2+",
    "G21EA-EA_c-",
    "G21EA-EA_o-",
    "G21EA-EA_p-",
    "G21EA-EA_s-",
    "G21EA-EA_si-",
    "G21IP-al+",
    "G21IP-b+",
    "G21IP-be+",
    "G21IP-c+",
    "G21IP-cl+",
    "G21IP-f+",
    "G21IP-mg+",
    "G21IP-n+",
    "G21IP-o+",
    "G21IP-p+",
    "G21IP-s+",
    "G21IP-si+",
    "HEAVYSB11-br",
    "RG18-ar",
    "RG18-kr",
    "RG18-ne",
    "SIE4x4-he",
    "SIE4x4-he+",
    # #######################
    # #########   1  ########
    # #######################
    "molecule1-W4_11",
    # "AHB21-1B",
    # "AHB21-10A",
    # "AHB21-15A",
    # "ALK8-li_h",
    # "G21EA-EA_11",
    # "G21EA-EA_14",
    # "G21EA-EA_17",
    # "G21EA-EA_17n",
    # "G21EA-EA_8",
    # "G21IP-IP_61",
    # "G21IP-IP_63",
    # "G21IP-IP_65",
    # "G21IP-IP_68",
    # "G21IP-IP_70",
    # "HEAVY28-hbr",
    # "HEAVY28-hi",
    # "HEAVYSB11-seh",
    # "MB16_43-NaH",
    # "SIE4x4-h2+_1.0",
    # "SIE4x4-h2+_1.25",
    # "SIE4x4-h2+_1.5",
    # "SIE4x4-h2+_1.75",
    # # ######## 2 H ########
    # "BH76-PH2",
    # "BH76-RKT01",
    # "BH76-RKT13",
    # "BH76-RKT06",
    # "BH76-RKT10",
    # "BH76-hfhts",
    # "BH76-RKT14",
    # "CARBHB12-3CL_B",
    # "DIPCS10-h2s_2+",
    # "G21EA-EA_12",
    # "G21EA-EA_15",
    # "G21EA-EA_18",
    # "G21EA-EA_9",
    # "G21IP-IP_62",
    # "G21IP-IP_66",
    # "HEAVY28-teh2",
    # "MB16_43-BeH2",
    # "MB16_43-MgH2",
    # "PA26-h2p",
    # "PA26-hclp",
    # # #####################
    # # ########  2  ########
    # # #####################
    # "molecule2-W4_11",
    # "ALK8-li2",
    # "ALK8-na2",
    # "ALKBDE10-bef",
    # "ALKBDE10-beo",
    # "ALKBDE10-cao",
    # "ALKBDE10-kf",
    # "ALKBDE10-lif",
    # "ALKBDE10-lio",
    # "ALKBDE10-mgo",
    # "ALKBDE10-mgs",
    # "ALKBDE10-nao",
    # "G21EA-EA_20",
    # "G21EA-EA_21",
    # "G21EA-EA_22",
    # "G21EA-EA_23",
    # "G21EA-EA_23n",
    # "G21EA-EA_24",
    # "G21EA-EA_25",
    # "G21IP-IP_73",
    # "G21IP-IP_74",
    # "G21IP-IP_75",
    # "G21IP-IP_76",
    # "G21IP-IP_77",
    # "G21IP-IP_78",
    # "G21IP-IP_79",
    # "G21IP-IP_80",
    # "HAL59-BrBr",
    # "HAL59-FBr",
    # "HAL59-FI",
    # "RG18-ar2",
    # "RG18-kr2",
    # "RG18-ne2",
    # "SIE4x4-he2+_1.0",
    # "SIE4x4-he2+_1.25",
    # "SIE4x4-he2+_1.5",
    # "SIE4x4-he2+_1.75",
]

eval_str_list = [
    # #####################
    # ######  4(<9)  ######
    # #####################
    # "molecule4-W4_11",
    # #####################
    # ######  5(<9)  ######
    # #####################
    "molecule5-W4_11",
    # ######## 0 H ########
    "ALK8-li4_c",
    "G2RC-62",
    "G2RC-67",
    "HAL59-29_CF3Br-benB",
    "IL16-152B",
    "IL16-214B",
    "IL16-229B",
    # ######## 1 H ########
    "HAL59-BrBr_FCCH",
    "PNICO23-22b",
    # ######## 2 H ########
    "DC13-o3_c2h2_add",
    "RSE43-P5",
    "RSE43-P7",
    "TAUT15-7a",
    "TAUT15-7b",
    "YBDE18-nf3-ch2",
    "YBDE18-pf3-ch2",
    # ######## 3 H ########
    "IL16-230B",
    "RSE43-E5",
    "RSE43-E7",
    # #####################
    # ######  6(<9)  ######
    # #####################
    # ######## 0 H ########
    "RG18-ne6",
    # ######## 1 H ########
    "ALK8-li5_ch",
    "IL16-144B",
    "PArel-c2cl41",
    "PArel-c2cl42",
    "PArel-c2cl43",
    # ######## 2 H ########
    "RSE43-P28",
    # ######## 3 H ########
    "PArel-c2h2f41",
    "PArel-c2h2f42",
    "RSE43-E28",
]

train_str_exclude_list = []
eval_str_exclude_list = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    print_computer_info(args.device)

    train_str_list = gen_name_args(train_str_list, args.dataset, args.name_mol_reverse)
    train_str_exclude_list = gen_name_args(
        train_str_exclude_list, args.dataset, args.name_mol_reverse, if_exclude=True
    )
    eval_str_list = gen_name_args(eval_str_list, args.dataset, args.name_mol_reverse)
    eval_str_exclude_list = gen_name_args(
        eval_str_exclude_list, args.dataset, args.name_mol_reverse, if_exclude=True
    )

    # remove the same name in train and train_str_exclude_list
    train_str_list = [
        mol for mol in train_str_list if mol not in train_str_exclude_list
    ]

    # remove the same name in eval and eval_str_exclude_list
    eval_str_list = [mol for mol in eval_str_list if mol not in eval_str_exclude_list]

    print(f"Train set size: {len(train_str_list)}")
    print(f"Train set: {train_str_list}")
    print(f"Eval set size: {len(eval_str_list)}")
    print(f"Eval set: {eval_str_list}")
    train_model(train_str_list, eval_str_list, args)
