# HELP: just run qsub qsub_script_file.sh 
# at the command line
#---------------------------Start programname.job------------------------
#!/bin/bash

# The job should be placed into the queue 'all.q'.
#$ -q batch.q

# The name of the job, can be whatever makes sense to you
#$ -N MPI_MCMC_Example

# Redirect output stream to this file.
#$ -o mpimcmcex.out.dat

# Redirect error stream to this file.
#$ -e mpimcmcex.err.dat

# The batchsystem should use the current directory as working directory.
# Both files (output.dat and error.dat) will be placed in the current
# directory. The batchsystem assumes to find the executable in this directory.
#$ -cwd

# This is my email address for notifications. I want to have all notifications
# at the master node of this cluster. This is optional.
#$ -M jeremy.lise@gmail.com

# Send me an email when the job is finished.
#$ -m e

# Use the parallel environment "lam", which assigns two processes
# to one host. In this example, if there are not enough machines to run the
# mpi job on 120 processors the batchsystem can also use fewer than 120 but
# the job should not run on fewer than 30 processors.
#$ -pe openmpi 24

# This is the file to be executed.
echo $PATH

. /etc/profile.d/modules.sh
module load intel/fortran openmpi/intel/64

mpirun -np $NSLOTS ./example
#---------------------------End programname.job------------------------


