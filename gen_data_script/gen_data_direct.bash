#!/bin/bash

# Parameters for train.py
export dl_args="-0.5 0.5 3"
# export basis_args="Def2-SVP"
export basis_args="cc-pVDZ"
export n_rad_args=""
export n_ang_args=""

## user's own commands below
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

export PID_THIS_RUN=$$

export PYSCF_MAX_MEMORY=80000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
export DFT2CC_CUBE_USE=3
export DFT2CC_GENERATE_DATA=1
export PYSCF_TMPDIR=~/raid/tmp

export NVIDIA_VISIBLE_DEVICES=1
# use less power GPU
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# use most free memory GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -n | tail -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

mkdir -p log
mkdir -p validate
mkdir -p data/grids_dft

if [ -z "$n_rad_args" ]; then
	export mol_args="--distance_list ${dl_args} --basis ${basis_args} --extend_atom 0-1 --extend_xyz 0"
else
	export mol_args="--distance_list ${dl_args} --basis ${basis_args} --n_rad ${n_rad_args} --n_ang ${n_ang_args} --extend_atom 0-1 --extend_xyz 0"
fi

nohup bash <<'EOF' >log/gen_data-${PID_THIS_RUN}.log 2>&1 &
set -e  # Exit on any error
~/anaconda3/envs/pyscf/bin/python gen_data.py ${mol_args} --name_mol BHDIV10-ts6 PA26-glyp CDIE20-R21 BHPERI-03r FH51-C2H5CO2H CDIE20-R20 GW100-542-92-7 PA26-phosphapyrrolp RSE43-E6 CDIE20-P20 BHDIV10-ed6 S66-46A ISO34-E30 S66-16A RSE43-E16 FH51-C3H7CN S66-04B S66-15B FH51-C2H5CONH2 S66-64A S66-15A S66-11B S66-13A S66-57B ISO34-P30 S66-14A S66-07B FH51-2-pentyne ISO34-E8 FH51-dimethyloxirane ISO34-E9 FH51-pentadiene FH51-1-pentyne ISO34-P9 ISO34-E29 ISO34-P29 ISO34-P8 BH9-06_31R1 WATER27-OHmH2O4cs ISO34-P18 ISO34-E18 WATER27-OHmH2O4c4 S66-38B FH51-1-pentene FH51-cis-2-pentene S66-37A S66-38A FH51-trans-2-pentene FH51-S_C2H5_2 IL16-212 WATER27-H2O5 GW100-60-29-7 S66-39B S66-42B FH51-diethylamine FH51-C4H9NH2 RSE43-P45 S66-62A ICONF-SI5H12_2 S30L-27B S66-34A GW100-14868-53-2 S66-34B S66-44B ACONF-P_TT S66-46B S66-41B S66-45B ACONF-P_TG ICONF-SI5H12_3 S66-37B ADIM6-AM5 ICONF-SI5H12_1 IL16-202A S66-43B RSE43-E45 ACONF-P_GG S66-35B ICONF-SI5H12_4 S66-40B S66-35A ACONF-P_GX S66-36B ISO34-E10 S66-36A ISO34-P10 S66-61A --dataset gmtkn || exit 1
echo DONE
EOF

echo $! >>log/save_pid.txt 2>&1
