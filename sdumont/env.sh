# Sourced by the jobs and by interactive shells on SDumont: paths, MPI and Spack.

export DECATUR_BASE=${DECATUR_BASE:-/petrobr/parceirosbr/sismo_co2/${USER}}
export DECATUR_REPO=${DECATUR_BASE}/decatur
export DECATUR_WORK=${DECATUR_BASE}/decatur-work
export SPACK_MIRROR=${DECATUR_BASE}/spack-mirror
export MESHING_DIR=${DECATUR_BASE}/Meshing
export EVALUATE_EASI=${MESHING_DIR}/evaluate_material/build/evaluate_easi

module load openmpi/gnu/5.0.5.1.0
source "${DECATUR_BASE}/spack/share/spack/setup-env.sh"
export PMIX_MCA_psec=^munge
# MPI-IO on /petrobr (Lustre): OpenMPI's ompio fails in MPI_File_open (MPI_ERR_INTERN), even
# on one rank and with fs=ufs, ROMIO works
export OMPI_MCA_io=romio341
