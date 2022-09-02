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
module load gcc/11
module load cuda/11.6
module load openmpi/4.1.1

export PSM2_CUDA=1

$MPIEXEC $FLAGS_MPI_BATCH ./test_dergeraet_2d

