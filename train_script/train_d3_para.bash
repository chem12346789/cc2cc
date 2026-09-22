#!/bin/bash
#slurm options
#SBATCH -p gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=20:00:00
#SBATCH -J train-d3-para
#SBATCH -o log/train-d3-para-%A.log -e log/train-d3-para-%A.err

export basis_args="def2-QZVP(D)"
# export load_args=""
export load_args="atom-2647768"
# export load_args="atom-98386"
# export load_args=""

export NVIDIA_VISIBLE_DEVICES=1
# use less power GPU with available free memory larger than 2000 MiB
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,power.draw,index --format=csv,nounits,noheader | awk -F, '$1 > 2000 {print $2 $3}' | sort -n | head -1 | awk '{print $NF}')
# # use most free memory GPU
# export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -n | tail -1 | awk '{ print $NF }')

export VALIDATE_DIR="validate_hkqai_done"

export NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${NUMBER_OF_THREADS}
export MKL_NUM_THREADS=${NUMBER_OF_THREADS}
export NUMEXPR_NUM_THREADS=${NUMBER_OF_THREADS}
export OPENBLAS_NUM_THREADS=${NUMBER_OF_THREADS}
export LD_PRELOAD=~/.local/lib/libjemalloc.so:$LD_PRELOAD

export PYSCF_MAX_MEMORY=8000
export PYTHONPATH=~/python:$PYTHONPATH
export PYTHON_BIN=~/anaconda3/envs/pyscf/bin/python
export LOSS_TYPE="mse"

mkdir -p log
mkdir -p validate

for rs18 in $(seq 0.15 0.05 0.15); do
    for rs6 in $(seq 0.8 0.1 0.8); do
        echo "Training with rs6=${rs6}, rs18=${rs18}..."

        # ${PYTHON_BIN} d3_para.py \
        #     --mode train \
        #     --basis "${basis_args}" \
        #     --load "${load_args}" \
        #     --epochs 1000 \
        #     --print_step 1 \
        #     --lr 1e-2 \
        #     --dataset gmtkn-def2 \
        #     --damping bj \
        #     --loss-type "${LOSS_TYPE}" \
        #     --s6 1.0 \
        #     --alp 14 \
        #     --rs6 "${rs6}" \
        #     --rs18 "${rs18}" \
        #     --optimizer lbfgs

        ${PYTHON_BIN} d3_para.py \
            --mode test \
            --basis "${basis_args}" \
            --load "${load_args}" \
            --dataset gmtkn-def2 \
            --damping bj
    done
done

# ~/anaconda3/envs/pyscf/bin/python d3_para.py --mode train --basis ${basis_args} --load ${load_args} --epochs 150000 --lr 1e-1 --dataset gmtkn-def2 --damping bj --s6 1.0 --s18 0.1 --alp 14
# ~/anaconda3/envs/pyscf/bin/python d3_para.py --mode train --basis ${basis_args} --load ${load_args} --epochs 1000 --print_step 1 --lr 1e-2 --dataset gmtkn-def2 --damping bj --s18 -0.6 --alp 14 --optimizer levenberg-marquardt
# # "adagrad", "levenberg-marquardt", "nelder-mead"
# echo "Training completed successfully."
# echo "DONE"
# ~/anaconda3/envs/pyscf/bin/python d3_para.py --mode test --basis ${basis_args} --load ${load_args} --dataset gmtkn-def2 --damping bj
# echo "DONE"

# echo "Starting data collection for model ${load_args} with basis ${basis_args}..."
# ~/anaconda3/envs/pyscf/bin/python collect_info.py --model_load ${load_args} --basis ${basis_args} --verbose 0 --data_set dft-fitset-def2 --max_checks 0 --sota
# ~/anaconda3/envs/pyscf/bin/python collect_info.py --model_load ${load_args} --basis ${basis_args} --verbose 0 --max_checks 0 --sota
# echo "Data collection started in the background. Check log/intermol_${load_args}_${basis_args}.log for progress."
