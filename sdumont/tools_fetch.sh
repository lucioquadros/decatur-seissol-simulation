#!/bin/bash
# Phase 1 on a login node (internet): the decatur-tools Spack environment (pumgen, cmake)
# resolved and its sources mirrored, plus SeisSol's Meshing repository for evaluate_easi.
# Phase 2 is tools_build.sbatch.

source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/env.sh"
set -euo pipefail

ENV=decatur-tools
SITE_YAML=${DECATUR_BASE}/seissol-spack-installer/sites/sdumont2nd/packages.yaml
MESHING_COMMIT=47bfd783746f67ade76e795b66e149e6206f4258

# a SEPARETE ENVIRONMENT for tools
if ! spack env list | grep -qE "^[*[:space:]]*${ENV}$"; then
    spack env create "${ENV}"
    cp "${SITE_YAML}" "$(spack location -e "${ENV}")/site-packages.yaml"
    spack -e "${ENV}" config add "include:[site-packages.yaml]"
    spack -e "${ENV}" config add "packages:all:require:'target=zen3'"
    spack -e "${ENV}" config add "concretizer:targets:host_compatible:false"
    spack -e "${ENV}" add pumgen@1.1.1
    spack -e "${ENV}" add cmake
    spack -e "${ENV}" mirror add --type source decatur-mirror "file://${SPACK_MIRROR}"
fi
spack -e "${ENV}" concretize
spack -e "${ENV}" mirror create -d "${SPACK_MIRROR}" --all --private -j 4

if [[ ! -d "${MESHING_DIR}/.git" ]]; then
    git clone https://github.com/SeisSol/Meshing.git "${MESHING_DIR}"
fi
git -C "${MESHING_DIR}" checkout -q "${MESHING_COMMIT}"
git -C "${MESHING_DIR}" submodule update --init --recursive submodules/PUML submodules/glm submodules/utils

echo "Phase 1 done. Next, from ${DECATUR_REPO}: sbatch sdumont/tools_build.sbatch"
