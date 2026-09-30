#!/bin/bash -l
#SBATCH --job-name=decatur-tools
#SBATCH --partition=cpu_amd
#SBATCH --account=sismo_co2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Phase 2, offline: installs decatur-tools from the mirror and builds evaluate_easi
# against the easi/ASAGI of seissol-env. Submit from the repository root after
# sdumont/tools_fetch.sh.

source "${SLURM_SUBMIT_DIR}/sdumont/env.sh" || exit 1
set -eo pipefail

# a short timeout makes an unexpected download fail quickly instead of hanging
spack -c config:connect_timeout:5 -e decatur-tools install --jobs "${SLURM_CPUS_PER_TASK}" \
    --fail-fast --yes-to-all
CMAKE=$(spack -e decatur-tools location -i cmake)/bin/cmake

# patch to stop MPI aborting the script at exit
patch=${SLURM_SUBMIT_DIR}/sdumont/evaluate_easi.patch
if git -C "${MESHING_DIR}" apply --check "${patch}" 2>/dev/null; then
    git -C "${MESHING_DIR}" apply "${patch}"
fi

spack env activate seissol-env
src=${MESHING_DIR}/evaluate_material
"${CMAKE}" -S "${src}" -B "${src}/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
"${CMAKE}" --build "${src}/build" -j "${SLURM_CPUS_PER_TASK}"

spack env activate decatur-tools
echo "pumgen: $(command -v pumgen)"
echo "evaluate_easi: ${EVALUATE_EASI}"
