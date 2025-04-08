#!/bin/bash
export MODEL="densenet"
export load_args_list="atom-1-1055975"

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

export NUMBER_OF_GPU=1
export NUMBER_OF_THREADS=12

export OMP_NUM_THREADS=${NUMBER_OF_THREADS}
export MKL_NUM_THREADS=${NUMBER_OF_THREADS}
export NUMEXPR_NUM_THREADS=${NUMBER_OF_THREADS}
export OPENBLAS_NUM_THREADS=${NUMBER_OF_THREADS}
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export LD_PRELOAD=~/.local/lib/libjemalloc.so:$LD_PRELOAD

export PYSCF_MAX_MEMORY=80000
export PYTHONPATH=~/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
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
~/anaconda3/envs/pyscf/bin/python test.py ${mol_args} --precision float64 ${load_model_args} --load_epoch -41000 --load ${load_args} --model ${MODEL} --dataset ${DATASET} --device cpu --name_mol molecule_W4_11 molecule_G21EA molecule_G21IP molecule_DIPCS10 molecule_PA26 molecule_SIE4x4 molecule_ALKBDE10 molecule_YBDE18 molecule_AL2X6 molecule_HEAVYSB11 molecule_NBPRC molecule_ALK8 molecule_RC21 molecule_G2RC molecule_BH76 molecule_FH51 molecule_TAUT15 molecule_DC13 molecule_MB16_43 molecule_DARC molecule_RSE43 molecule_BSR36 molecule_CDIE20 molecule_ISO34 molecule_ISOL24 molecule_C60ISO molecule_PArel molecule_BHPERI molecule_BHDIV10 molecule_INV24 molecule_BHROT27 molecule_PX13 molecule_WCPT18 molecule_RG18 molecule_ADIM6 molecule_S22 molecule_S66 molecule_HEAVY28 molecule_WATER27 molecule_CARBHB12 molecule_PNICO23 molecule_HAL59 molecule_AHB21 molecule_CHB6 molecule_IL16 molecule_IDISP molecule_ICONF molecule_ACONF molecule_Amino20x4 molecule_PCONF21 molecule_MCONF molecule_SCONF molecule_UPU23 molecule_BUT14DIOL --if_continue ${if_continue_args}
echo "Test completed successfully."
echo DONE
EOF
    echo $! >>log/save_pid.txt 2>&1
done

# # Basic properties and reaction energies for small systems
# molecule_W4_11 molecule_G21EA molecule_G21IP molecule_DIPCS10 molecule_PA26 molecule_SIE4x4 molecule_ALKBDE10 molecule_YBDE18 molecule_AL2X6 molecule_HEAVYSB11 molecule_NBPRC molecule_ALK8 molecule_RC21 molecule_G2RC molecule_BH76 molecule_FH51 molecule_TAUT15 molecule_DC13
# # Reaction energies for large systems and isomerisation reactions
# molecule_MB16_43 molecule_DARC molecule_RSE43 molecule_BSR36 molecule_CDIE20 molecule_ISO34 molecule_ISOL24 molecule_C60ISO molecule_PArel
# # Reaction barrier heights
# molecule_BHPERI molecule_BHDIV10 molecule_INV24 molecule_BHROT27 molecule_PX13 molecule_WCPT18
# # Intermolecular noncovalent interactions
# molecule_RG18 molecule_ADIM6 molecule_S22 molecule_S66 molecule_HEAVY28 molecule_WATER27 molecule_CARBHB12 molecule_PNICO23 molecule_HAL59 molecule_AHB21 molecule_CHB6 molecule_IL16
# # Intramolecular noncovalent interactions
# molecule_IDISP molecule_ICONF molecule_ACONF molecule_Amino20x4 molecule_PCONF21 molecule_MCONF molecule_SCONF molecule_UPU23 molecule_BUT14DIOL
