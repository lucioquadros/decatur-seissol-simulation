#!/bin/bash -l
#SBATCH --job-name=decatur-mesh
#SBATCH --partition=cpu_amd
#SBATCH --account=sismo_co2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# SeisSol material grid and Gmsh mesh of a scenario in DIR.
# Usage, from the repository root:
#   sbatch sdumont/mesh.sbatch <scenario> <dir> [build_mesh.py flags, e.g. --lc-fault 100]

source "${SLURM_SUBMIT_DIR}/sdumont/env.sh" || exit 1
source "${DECATUR_REPO}/.venv/bin/activate"
set -euo pipefail

scenario=$1
dir=$(realpath -m "$2")
shift 2
cd "${DECATUR_REPO}"
python scripts/make_scenario.py "${scenario}" --outdir "${dir}"
python scripts/build_material.py "${scenario}" --outdir "${dir}"
python scripts/build_mesh.py "${scenario}" --outdir "${dir}" \
    --threads "${SLURM_CPUS_PER_TASK}" --algorithm3d hxt "$@"
