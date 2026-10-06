#!/bin/bash

# Parameters for train.py
export DATASET="gmtkn-def2"

export basis_args="cc-pVDZ"
# export basis_args="cc-pVTZ"

export MP_TOTAL=1

export DATASET="gmtkn-def2"
export IF_CONTINUE=0

export NUMBER_OF_GPU=1
export NUMBER_OF_THREADS=48
export OMP_NUM_THREADS=${NUMBER_OF_THREADS}
export MKL_NUM_THREADS=${NUMBER_OF_THREADS}
export NUMEXPR_NUM_THREADS=${NUMBER_OF_THREADS}
export OPENBLAS_NUM_THREADS=${NUMBER_OF_THREADS}
export LD_PRELOAD=~/.local/lib/libjemalloc.so:$LD_PRELOAD

export PID_THIS_RUN=$$

export PYSCF_MAX_MEMORY=64000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/envs/pyscf/lib:$LD_LIBRARY_PATH
export DFT2CC_CUBE_USE=3
export PYSCF_TMPDIR=~/raid/tmp


mkdir -p log
mkdir -p data/grids_dft

# Clear previous PID file
> log/save_pid.txt

for MP_NUMBER in $(seq 0 $((MP_TOTAL - 1)));
do
    nohup bash -c "
        export OMP_NUM_THREADS=${NUMBER_OF_THREADS}
        export MKL_NUM_THREADS=${NUMBER_OF_THREADS}
        export NUMEXPR_NUM_THREADS=${NUMBER_OF_THREADS}
        export OPENBLAS_NUM_THREADS=${NUMBER_OF_THREADS}
        export VECLIB_MAXIMUM_THREADS=${NUMBER_OF_THREADS}
        export LD_PRELOAD=~/.local/lib/libjemalloc.so:$LD_PRELOAD
        ~/anaconda3/envs/pyscf/bin/python gen_data.py --basis ${basis_args} --dataset ${DATASET} --if_continue ${IF_CONTINUE} --if_eval 0 --name_mol_reverse 0 --md_number 0 --mp_number ${MP_NUMBER} --mp_total ${MP_TOTAL} --grid_level 0
    " >log/gen_data-${PID_THIS_RUN}-${MP_NUMBER}.log 2>log/gen_data-${PID_THIS_RUN}-${MP_NUMBER}.err &
    echo $! >>log/save_pid.txt
done

