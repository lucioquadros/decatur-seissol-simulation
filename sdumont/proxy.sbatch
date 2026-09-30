#!/bin/bash -l
#SBATCH --job-name=decatur-proxy
#SBATCH --partition=cpu_amd
#SBATCH --account=sismo_co2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# SeisSol proxy (order 5, 100,000 elements, 20 steps) at several thread counts, for
# estimate_cost.py --gflops-per-core (hardware GFLOPS / threads).
# Usage, from the repository root: sbatch sdumont/proxy.sbatch [threads ...]

source "${SLURM_SUBMIT_DIR}/sdumont/env.sh" || exit 1
spack env activate seissol-env
set -eo pipefail

export OMP_PLACES=cores OMP_PROC_BIND=close
proxy=$(compgen -c SeisSol_proxy_ | head -n 1)
threads=${*:-48 96 192}
echo "${proxy} on $(hostname), $(date '+%Y-%m-%d %H:%M')"
echo "threads,repeat,gflops_hardware,gflops_nonzero,gflops_hardware_per_core"
for t in ${threads}; do
    for r in 1 2 3; do
        out=$(OMP_NUM_THREADS=${t} "${proxy}" 100000 20 all)
        hw=$(awk -F: 'index($0, "GFLOPS (hardware)") { gsub(/ /, "", $2); print $2 }' <<< "${out}")
        nz=$(awk -F: 'index($0, "GFLOPS (non-zero)") { gsub(/ /, "", $2); print $2 }' <<< "${out}")
        echo "${t},${r},${hw},${nz},$(awk -v g="${hw}" -v t="${t}" 'BEGIN { printf "%.2f", g / t }')"
    done
done
