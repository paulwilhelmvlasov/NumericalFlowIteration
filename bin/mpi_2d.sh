#!/usr/local_rwth/bin/zsh

# ask for eight tasks
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:2

# ask for less than 4 GB memory per task=MPI rank
#SBATCH --mem-per-cpu=3900M   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)

# name the job
#SBATCH --job-name=dergeraet_2d

# declare the merged STDOUT/STDERR file
#SBATCH --output=output.%J.txt

### beginning of executable commands
module unload intel
module unload intelmpi
module load CUDA
module load GCC/11
module load intel-compilers/2021.2.0
module load OpenMPI/4.1.1

module load Automake/1.16

export PSM2_CUDA=1

$MPIEXEC $FLAGS_MPI_BATCH ../bin/test_dergeraet_2d

