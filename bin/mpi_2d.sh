#!/usr/bin/zsh

### Setup in this script:
### - 2 nodes (c18g, default)
### - 2 ranks per node
### - 1 GPU per rank (= both GPUs from the node)

#SBATCH -J 2-2-1
#SBATCH -o 2-2-1.%J.log
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2



#print some debug informations...
echo; export; echo; nvidia-smi; echo

module unload intel
module unload intelmpi
module load CUDA
module load GCC/11
module load intel-compilers/2021.2.0
module load OpenMPI/4.1.1

module load Automake/1.16

export PSM2_CUDA=1

$MPIEXEC $FLAGS_MPI_BATCH ../bin/test_vp_ion_acoustic_1d_dirichlet

