#!/bin/bash -l
#SBATCH --job-name=decatur-pumgen
#SBATCH --partition=cpu_amd
#SBATCH --account=sismo_co2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# mesh.msh -> mesh.puml.hdf5 (+ .xdmf).
# Usage, from the repository root: sbatch sdumont/pumgen.sbatch <dir>

source "${SLURM_SUBMIT_DIR}/sdumont/env.sh" || exit 1
spack env activate decatur-tools
source "${DECATUR_REPO}/.venv/bin/activate"
set -euo pipefail

cd "$(realpath "$1")"
# PUMGen reads msh2 on rank 0 and distributes the mesh to the other ranks
srun pumgen -s msh2 mesh.msh mesh.puml.hdf5
python "${DECATUR_REPO}/scripts/check_mesh.py" mesh.puml.hdf5
