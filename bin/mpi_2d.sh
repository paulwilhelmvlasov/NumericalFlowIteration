#!/usr/bin/zsh

### Setup in this script:
### - 4 nodes (c18g)
### - 1 rank per node
### - 2 GPUs per rank (= both GPUs from the node)
#SBATCH -J 4-1-2
#SBATCH -o 4-1-2.%J.log
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --output=output.%J.txt


# Print some debug information
echo; export; echo; nvidia-smi; echo


#SBATCH --time=0-00:55:00
module unload intel
module unload intelmpi
module load CUDA
module load GCC/11
module load intel-compilers/2021.2.0
module load OpenMPI/4.1.1

module load Automake/1.16

export PSM2_CUDA=1

$MPIEXEC $FLAGS_MPI_BATCH ../bin/test_vp_ion_acoustic_1d_dirichlet



