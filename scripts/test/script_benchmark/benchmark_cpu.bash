#!/bin/bash
#slurm options
#SBATCH -p cpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=2400:00:00
#SBATCH -a [0-54]%8
#SBATCH -J validate-data
#SBATCH -o log/test-atom-%A-%a.log -e log/test-atom-%A-%a.err
###SBATCH --exclude=cpu[18]

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="${ROOT_DIR}"
source "${SCRIPT_DIR}/lib/runtime.sh"
source "${SCRIPT_DIR}/lib/test_job.sh"

export if_continue_args=0
export name_mol_reverse=0
export load_epoch="-1"

# export basis_args="def2-QZVPPD"
export basis_args="def2-QZVP(D)"
# export basis_args="cc-pVQZ"

export DATASET="gmtkn-def2"
select_molecule_profile gmtkn55
setup_cpu_test_job 8000
begin_test_job benchmark_dft
run_cpu_benchmark_job
finish_test_job

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
# # addon molecules
# molecule_ACC24 molecule_GAPS molecule_GW100 molecule_MRADC molecule_S30L
