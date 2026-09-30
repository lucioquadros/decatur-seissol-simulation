#!/bin/bash -l
#SBATCH --job-name=decatur-run
#SBATCH --partition=cpu_amd
#SBATCH --account=sismo_co2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# Run SeisSol simulation in DIR (make_scenario inputs, material.nc, mesh.puml.hdf5), output in DIR/output.
# Usage, from the repository root (nodes, ranks and threads as sbatch flags):
#   sbatch sdumont/run.sbatch <dir>
#   sbatch -p cpu_amd_dev -t 00:20:00 --mem=32G sdumont/run.sbatch <dir>
#   sbatch --exclusive --mem=0 --nodes=4 --ntasks-per-node=8 --cpus-per-task=24 sdumont/run.sbatch <dir>

source "${SLURM_SUBMIT_DIR}/sdumont/env.sh" || exit 1
spack env activate seissol-env
set -euo pipefail

cd "$(realpath "$1")"
mkdir -p output

# one core per rank stays free for SeisSol's communication thread
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK - 1))
export OMP_PLACES="cores(${OMP_NUM_THREADS})" OMP_PROC_BIND=close OMP_STACKSIZE=16M
ulimit -Ss 2097152
export SEISSOL_ASAGI_MPI_MODE=OFF
export ASYNC_MODE=THREAD ASYNC_BUFFER_ALIGNMENT=8388608

seissol=$(compgen -c SeisSol_Release_ | head -n 1)
echo "${seissol} in ${PWD}, $(date '+%Y-%m-%d %H:%M')"
echo "${SLURM_JOB_NUM_NODES} nodes x ${SLURM_NTASKS_PER_NODE} ranks x ${SLURM_CPUS_PER_TASK} CPUs"
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA node" || true
# block:block keeps each rank's cores contiguous on one socket
srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --distribution=block:block \
    "${seissol}" parameters.par
ls -la output
