""" """

import argparse
from itertools import product

import numpy as np

from cc2cc.utils import gen_mole, print_computer_info, add_args
from cc2cc.utils import Grid, DATA_PATH
from cc2cc.utils.parser import gen_name_args
from cc2cc.gen_cc import cc
from cc2cc.gen_ucc import ucc


train_str_list = [
    # "molecule0-W4_11",
    # "ADDON_As",
    # "ADDON_Ge",
    # "ADDON_Se",
    # "ADDON_Te",
    # "ADDON_I",
    # "ADDON_Bi",
    # "ADDON_Pb",
    # "ADDON_Sb",
    # "AHB21-1A",
    # "AHB21-4A",
    # "ALK8-li+",
    # "ALK8-na+",
    # "ALKBDE10-ca",
    # "ALKBDE10-k",
    # "ALKBDE10-li",
    # "ALKBDE10-mg",
    # "ALKBDE10-na",
    # "CHB6-24A",
    # "DIPCS10-be_2+",
    # "DIPCS10-mg_2+",
    # "G21EA-EA_c-",
    # "G21EA-EA_o-",
    # "G21EA-EA_p-",
    # "G21EA-EA_s-",
    # "G21EA-EA_si-",
    # "G21IP-al+",
    # "G21IP-b+",
    # "G21IP-be+",
    # "G21IP-c+",
    # "G21IP-cl+",
    # "G21IP-f+",
    # "G21IP-mg+",
    # "G21IP-n+",
    # "G21IP-o+",
    # "G21IP-p+",
    # "G21IP-s+",
    # "G21IP-si+",
    # "HEAVYSB11-br",
    # "RG18-ar",
    # "RG18-kr",
    # "RG18-ne",
    # "SIE4x4-he",
    # "SIE4x4-he+",
    # # # #####################
    # # # ######  add 1  ######
    # # # #####################
    # "molecule1-W4_11",
    "AHB21-1B",
    "AHB21-10A",
    "AHB21-15A",
    "ALK8-li_h",
    "G21EA-EA_11",
    "G21EA-EA_14",
    "G21EA-EA_17",
    "G21EA-EA_17n",
    "G21EA-EA_8",
    "G21IP-IP_61",
    "G21IP-IP_63",
    "G21IP-IP_65",
    "G21IP-IP_68",
    "G21IP-IP_70",
    "HEAVY28-hbr",
    "HEAVY28-hi",
    "HEAVYSB11-seh",
    "MB16_43-NaH",
    "SIE4x4-h2+_1.0",
    "SIE4x4-h2+_1.25",
    "SIE4x4-h2+_1.5",
    "SIE4x4-h2+_1.75",
    # # # ######## 2 H ########
    "BH76-PH2",
    "BH76-RKT01",
    "BH76-RKT13",
    "BH76-RKT06",
    "BH76-RKT10",
    "BH76-hfhts",
    "BH76-RKT14",
    "CARBHB12-3CL_B",
    "DIPCS10-h2s_2+",
    "G21EA-EA_12",
    "G21EA-EA_15",
    "G21EA-EA_18",
    "G21EA-EA_9",
    "G21IP-IP_62",
    "G21IP-IP_66",
    "HEAVY28-teh2",
    "MB16_43-BeH2",
    "MB16_43-MgH2",
    "PA26-h2p",
    "PA26-hclp",
    # #####################
    # ########  2  ########
    # #####################
    # "W4_11-alcl",
    # "W4_11-alf",
    # "W4_11-b2",
    # "W4_11-be2",
    # "W4_11-bf",
    # "W4_11-bn",
    # "W4_11-bn3pi",
    # "W4_11-c2",
    # "W4_11-cf",
    # "W4_11-cl2",
    # "W4_11-clf",
    # "W4_11-clo",
    # "W4_11-cn",
    # "W4_11-co",
    # "W4_11-cs",
    # "W4_11-f2",
    # "W4_11-n2",
    # "W4_11-no",
    # "W4_11-o2",
    # "W4_11-of",
    # "W4_11-p2",
    # "W4_11-s2",
    # "W4_11-sif",
    # "W4_11-sio",
    # "W4_11-so",
    # "W4_11-cch",
    # "W4_11-hcn",
    # "W4_11-hnc",
    # "W4_11-hco",
    # "W4_11-hno",
    # "W4_11-hocl",
    # "W4_11-hof",
    # "W4_11-hoo",
    # "W4_11-n2h",
    # "W4_11-ssh",
    # "W4_11-c-hcoh",
    # "W4_11-h2co",
    # "W4_11-t-hcoh",
    # "W4_11-c-n2h2",
    # "W4_11-t-n2h2",
    # "W4_11-c2h2",
    # "W4_11-ch2c",
    # "W4_11-h2cn",
    # "W4_11-hcnh",
    # "W4_11-hooh",
    # "W4_11-nh2cl",
    # "W4_11-ch2ch",
    # "W4_11-ch2nh",
    # "W4_11-ch3f",
    # "W4_11-sih3f",
    # "W4_11-c2h4",
    # "W4_11-ch2nh2",
    # "W4_11-ch3nh",
    # "W4_11-methanol",
    # "W4_11-n2h4",
    # "W4_11-ch3nh2",
    # "W4_11-b2h6",
    # "W4_11-c2h6",
    # "W4_11-si2h6",
    # # ######## 0 H ########
    "ALK8-li2",
    "ALK8-na2",
    "ALKBDE10-bef",
    "ALKBDE10-beo",
    "ALKBDE10-cao",
    "ALKBDE10-kf",
    "ALKBDE10-lif",
    "ALKBDE10-lio",
    "ALKBDE10-mgo",
    "ALKBDE10-mgs",
    "ALKBDE10-nao",
    "G21EA-EA_20",
    "G21EA-EA_21",
    "G21EA-EA_22",
    "G21EA-EA_23",
    "G21EA-EA_23n",
    "G21EA-EA_24",
    "G21EA-EA_25",
    "G21IP-IP_73",
    "G21IP-IP_74",
    "G21IP-IP_75",
    "G21IP-IP_76",
    "G21IP-IP_77",
    "G21IP-IP_78",
    "G21IP-IP_79",
    "G21IP-IP_80",
    "HAL59-BrBr",
    "HAL59-FBr",
    "HAL59-FI",
    "RG18-ar2",
    "RG18-kr2",
    "RG18-ne2",
    "SIE4x4-he2+_1.0",
    "SIE4x4-he2+_1.25",
    "SIE4x4-he2+_1.5",
    "SIE4x4-he2+_1.75",
    # # ######## 1 H ########
    # "AHB21-3",
    # "AHB21-6",
    # "AHB21-8",
    # "BH76-RKT17",
    # "BH76-hf2ts",
    # "HAL59-PCH",
    # "RG18-hfAr",
    # "RG18-hfKr",
    # "RG18-hfNe",
    # # ######## 2 H ########
    # "AHB21-17",
    # "AHB21-2",
    # "AHB21-5",
    # "AHB21-7",
    # "ALK8-li_na_h2",
    # "ALK8-na2_h2",
    # "BHROT27-h2s2",
    # "BHROT27-h2s2_cis",
    # "BHROT27-h2s2_trans",
    # "HEAVYSB11-h2s2",
    # "CHB6-22",
    # "CHB6-23",
    # "CHB6-24",
    # "DIPCS10-ch2o_2+",
    # "DIPCS10-n2h2_2+",
    # "G21IP-IP_71",
    # "HEAVYSB11-h2se2",
    # "NBPRC-nh-bh",
    # "PA26-ch2s",
    # "PNICO23-13a",
    # "PNICO23-14a",
    # "PNICO23-4a",
    # "PX13-hf_2",
    # "PX13-hf_2_ts",
    # "RSE43-P20",
    # # ######## 3 H ########
    # "AHB21-1",
    # "AHB21-10",
    # "AHB21-16",
    # "AHB21-4",
    # "ALK8-li_me",
    # "BH76-CH2OH",
    # "RSE43-P32",
    # "BH76-RKT15",
    # "BH76-ch3cl",
    # "CARBHB12-1CL",
    # "BHROT27-nh2oh_ecl",
    # "BHROT27-nh2oh_st1",
    # "BHROT27-nh2oh_st2",
    # "CARBHB12-3CL",
    # "HAL59-27_CH3Br-benB",
    # "HAL59-28_CH3I-benB",
    # "HAL59-OPH3",
    # "PNICO23-7a",
    # "HEAVY28-teh2_hbr",
    # "HEAVY28-teh2_hcl",
    # "HEAVY28-teh2_hi",
    # "HEAVYSB11-teme",
    # "PA26-c2h2p",
    # "PA26-ch2sp",
    # "RSE43-P37",
    # # ######## 4 H ########
    # "AHB21-15",
    # "AHB21-9",
    # "BH76-RKT07",
    # "BH76-RKT08",
    # "BH76-hch3clts",
    # "BH76-RKT11",
    # "BH76-hfch3ts",
    # "CARBHB12-3O",
    # "DIPCS10-c2h4_2+",
    # "G21IP-IP_72",
    # "HAL59-25_benBr-mSHB",
    # "YBDE18-h2s-ch2",
    # "HEAVY28-bih3_hbr",
    # "HEAVY28-bih3_hcl",
    # "HEAVY28-bih3_hi",
    # "HEAVY28-sbh3_hbr",
    # "HEAVY28-sbh3_hcl",
    # "HEAVY28-sbh3_hi",
    # "HEAVY28-teh2_2",
    # "HEAVY28-teh2_h2o",
    # "HEAVY28-teh2_h2s",
    # "NBPRC-nh2-bh2",
    # "PNICO23-6a",
    # "PNICO23-8a",
    # "PX13-h2o_2",
    # "PX13-h2o_2_ts",
    # "S66-01",
    # "RSE43-P36",
    # "SIE4x4-h2o2+_1.0",
    # "SIE4x4-h2o2+_1.25",
    # "SIE4x4-h2o2+_1.5",
    # "SIE4x4-h2o2+_1.75",
    # # #####################
    # # ########  3  ########
    # # #####################
    # "molecule3-W4_11",
    # "G2RC-68",
    # "INV24-SO2_TS",
    # "HAL59-FCCH",
    # "BHPERI-13r_2",
    # "G2RC-104",
    # "WCPT18-ts1",
    # "G2RC-121",
    # "G2RC-113",
    # "S66-59",
    # "WCPT18-reac3",
    # "WCPT18-ts3",
    # "ISO34-E1",
    # "ISO34-P2",
    # "RSE43-E9",
    # "INV24-Ether",
    # "INV24-Ether_TS",
    # "ISO34-E24",
    # "PA26-ethanol",
    # "RSE43-E10",
    # "G2RC-82",
    # "ISO34-E3",
    # "WCPT18-reac6",
    # "WCPT18-ts6",
    # "ADIM6-AM3",
    # "BHDIV10-ed4",
    # "BHDIV10-ts4",
    # # ######## 0 H ########
    # "AHB21-11A",
    # "AHB21-12A",
    # "RG18-ar3",
    # "RG18-kr3",
    # "RG18-ne3",
    # "YBDE18-f2s",
    # # ######## 1 H ########
    # "AHB21-18A",
    # "BH76-n2ohts",
    # "RC21-7p4",
    # # ######## 2 H ########
    # "BHPERI-13r_3",
    # "BHPERI-13r_5",
    # "G2RC-97",
    # "PNICO23-11a",
    # "PNICO23-12a",
    # "RG18-c2h2Ar",
    # "RG18-c2h2Ne",
    # "RSE43-P13",
    # "RSE43-P4",
    # "TAUT15-9a",
    # "TAUT15-9b",
    # "WCPT18-ts7",
    # # ######## 3 H ########
    # "BH76-ch3fclts",
    # "BH76-clch3clcomp",
    # "BH76-clch3clts",
    # "BH76-fch3clcomp1",
    # "BH76-fch3clcomp2",
    # "BH76-fch3clts",
    # "BH76-fch3fcomp",
    # "BH76-fch3fts",
    # "BHDIV10-ed10",
    # "BHDIV10-ts10",
    # "BHPERI-13r_6",
    # "ISO34-P14",
    # "BHPERI-13r_7",
    # "S22-04a",
    # "WCPT18-reac2",
    # "WCPT18-ts2",
    # "CARBHB12-2CL_B",
    # "HAL59-BrBr_NH3",
    # "HAL59-FI_NH3",
    # "HAL59-NH3_FBr",
    # "HAL59-NH3_FCl",
    # "PNICO23-10a",
    # "PX13-hf_3",
    # "PX13-hf_3_ts",
    # "RC21-1p2",
    # "RSE43-P12",
    # "RSE43-P43",
    # "WCPT18-reac8",
    # "WCPT18-ts8",
]

eval_str_list = [
    # # #####################
    # # ######    3    ######
    # # #####################
    "molecule3-W4_11",
    # # #####################
    # # ######  4(<9)  ######
    # # #####################
    "molecule4-W4_11",
    # # #####################
    # # ######  5(<9)  ######
    # # #####################
    "molecule5-W4_11",
    # ######## 0 H ########
    "ALK8-li4_c",
    "G2RC-62",
    "G2RC-67",
    "HAL59-29_CF3Br-benB",
    # "HAL59-30_CF3I-benB",
    "IL16-152B",
    "IL16-214B",
    "IL16-229B",
    # ######## 1 H ########
    "HAL59-BrBr_FCCH",
    # "HAL59-FI_FCCH",
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
    # "HAL59-NH3_F3CI",
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
    error_molecule = []

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

    if args.if_eval:
        name_mol_list, evaluate = eval_str_list[args.training_cycle :: 3], True
    else:
        name_mol_list, evaluate = train_str_list[args.training_cycle :: 3], False

    # name_mol_list = [
    #     "AHB21-1A",
    #     "AHB21-4A",
    #     "G21EA-EA_c-",
    #     "G21EA-EA_o-",
    #     "G21EA-EA_p-",
    #     "G21EA-EA_s-",
    #     "G21EA-EA_si-",
    # ]
    # evaluate = False

    error_molecule = []
    print(f"Name Molecule List: {name_mol_list}")

    for (
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
    ) in product(
        name_mol_list,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
    ):
        name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"

        try:
            mol = gen_mole(
                name_mol,
                extend_atom,
                extend_xyz,
                distance,
                args.basis,
                ma_basis=False,
                dataset_name=args.dataset,
            )

            if mol is None:
                print(f"SKIP: {name_mol} {extend_atom} {extend_xyz} {distance}")
                continue

            grids = Grid(mol, args.grid_level)

            if args.if_continue:
                if (DATA_PATH / f"data_{name}.npz").exists():
                    continue
                    # if mol.charge >= 0:
                    #     print(f"SKIP: {name_mol} {extend_atom} {extend_xyz} {distance}")
                    #     continue

            if mol.spin == 0:
                cc(mol, grids, name, args, evaluate=evaluate)
            else:
                ucc(mol, grids, name, args, evaluate=evaluate)
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {name_mol} {extend_atom} {extend_xyz} {distance}")
            print(e)
            error_molecule.append(name)
            print(f"Error molecule: {error_molecule}")
        finally:
            print(f"Processed: {name_mol} {extend_atom} {extend_xyz} {distance}")
        print()

    print(f"Error molecule: {error_molecule}")
