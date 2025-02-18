#!/bin/bash

## user's own commands below
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

export PYSCF_MAX_MEMORY=80000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

export NVIDIA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

# basis_args="Def2-TZVPD"
# nohup bash -c "~/anaconda3/envs/pyscf/bin/python test.py -dl 0 0 1 --basis ${basis_args} --extend_atom 0 --extend_xyz 0 --precision float64 --load cycle1 --load_epoch 15000 --dataset g2" >log/test-${basis_args}.log 2>&1 &

mkdir -p log
mkdir -p validate
mkdir -p data/grids_dft

export load_pid_args="2520597"
export dl_args="0 0 1"
# export basis_args="Def2-TZVPD"
export basis_args="cc-pVDZ"
export n_rad_args="302"
export n_ang_args="302"
if [ -z "$n_rad_args" ]; then
    export mol_args="--distance_list ${dl_args} --basis ${basis_args} --extend_atom 0 --extend_xyz 0"
else
    export mol_args="--distance_list ${dl_args} --basis ${basis_args} --n_rad ${n_rad_args} --n_ang ${n_ang_args} --extend_atom 0 --extend_xyz 0"
fi

cat <<EOF | nohup bash >log/test-${basis_args}-${load_pid_args}.log 2>&1 &
echo Starting test.py at $(date)
echo "${mol_args}"
~/anaconda3/envs/pyscf/bin/python test.py ${mol_args} --precision float64 --load cycle1-${load_pid_args} --load_epoch 10000 --dataset g2
echo DONE
EOF

echo $! >>log/save_pid.txt 2>&1
