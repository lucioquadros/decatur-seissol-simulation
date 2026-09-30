#!/bin/bash -l
#SBATCH --job-name=decatur-easi
#SBATCH --partition=cpu_amd
#SBATCH --account=sismo_co2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# material.yaml and fault.yaml evaluated by easi at every element, compared with
# the Python model. Output: easi_material.{h5,xdmf}, easi_fault.{h5,xdmf} for ParaView.
# Usage, from the repository root: sbatch sdumont/evaluate_easi.sbatch <scenario> <dir>

source "${SLURM_SUBMIT_DIR}/sdumont/env.sh" || exit 1
spack env activate seissol-env
source "${DECATUR_REPO}/.venv/bin/activate"
set -euo pipefail

scenario=$1
dir=$(realpath "$2")
# the ASAGI file names in the yaml files are relative
cd "${dir}"
# the writer is correct for one rank only
srun -n 1 "${EVALUATE_EASI}" -m mesh.puml.hdf5 -e material.yaml -o easi_material
srun -n 1 "${EVALUATE_EASI}" -m mesh.puml.hdf5 -e fault.yaml -o easi_fault
python "${DECATUR_REPO}/scripts/check_easi.py" "${scenario}" --dir "${dir}"
