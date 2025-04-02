#!/bin/bash
export MODEL="densenet"
export load_args_list="atom-1-2272051"

# export MODEL="transformer_4_ang"
# export load_args_list="atom-1-3928342"

# export DATASET="g2"
export DATASET="gmtkn"
# export DATASET="gmtkn-cc-pVDZ"

export if_continue_args="0"

export dl_args="0 0 1"
# export basis_args="Def2-TZVPD"
export basis_args="cc-pVDZ"
export n_rad_args=""
export n_ang_args=""

## user's own commands below
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

export PYSCF_MAX_MEMORY=80000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH

export NVIDIA_VISIBLE_DEVICES=1
# use less power GPU
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=power.draw,index --format=csv,nounits,noheader | sort -n | head -1 | awk '{ print $NF }')
# use most free memory GPU
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -n | tail -1 | awk '{ print $NF }')
# export CUDA_VISIBLE_DEVICES=NUMBER_OF_GPU

mkdir -p log
mkdir -p validate
mkdir -p data/grids_dft
mkdir -p data/test

if [ -z "$n_rad_args" ]; then
    export mol_args="--distance_list ${dl_args} --basis ${basis_args} --extend_atom 0 --extend_xyz 0"
else
    export mol_args="--distance_list ${dl_args} --basis ${basis_args} --n_rad ${n_rad_args} --n_ang ${n_ang_args} --extend_atom 0 --extend_xyz 0"
fi

for load_args in ${load_args_list}; do
    cat <<EOF | nohup bash >log/test-${basis_args}-${load_args}.log 2>&1 &
echo Starting test.py at $(date)
echo "Testing mol: ${mol_args}"
echo "Load model: ${MODEL} ${load_args}"
~/anaconda3/envs/pyscf/bin/python test.py ${mol_args} --precision float64 ${load_model_args} --load_epoch -3000 --load ${load_args} --model ${MODEL} --dataset ${DATASET} --name_mol molecule_W4_11 molecule_DC13 molecule_G21EA molecule_BH76 --if_continue ${if_continue_args}
echo "Test completed successfully."
echo DONE
EOF
    echo $! >>log/save_pid.txt 2>&1
done
