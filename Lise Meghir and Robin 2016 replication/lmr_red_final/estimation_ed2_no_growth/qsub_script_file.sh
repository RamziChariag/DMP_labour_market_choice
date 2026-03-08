# HELP: just run qsub qsub_script_file.sh 
# at the command line
#---------------------------Start programname.job------------------------
#!/bin/bash

# The job should be placed into the queue 'batch.q'.
#$ -q batch.q

# The name of the job, can be whatever makes sense to you
#$ -N ed2_v3_LMR

# Redirect output stream to this file.
#$ -o lmr.out.dat

# Redirect error stream to this file.
#$ -e lmr.err.dat

# The batchsystem should use the current directory as working directory.
# Both files (output.dat and error.dat) will be placed in the current
# directory. The batchsystem assumes to find the executable in this directory.
#$ -cwd

# This is my email address for notifications. I want to have all notifications
# at the master node of this cluster. This is optional.
#$ -M jeremy.lise@gmail.com

# Send me an email when the job is finished.
#$ -m e

# Set the number of processors to 1 + chain_count (from main_mpi)
#$ -pe openmpi 192

# This is the file to be executed.
. /etc/profile.d/modules.sh

module load  intel/fortran openmpi/intel/64 nag/f90

echo "Loaded modules"
module list

echo "Path: "
echo $PATH

mpirun -np $NSLOTS ./dbg/LMR_mpi
#---------------------------End programname.job------------------------


