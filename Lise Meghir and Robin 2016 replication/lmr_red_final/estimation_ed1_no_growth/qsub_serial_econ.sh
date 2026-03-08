#!/bin/bash
#
# Script to submit a batch job
#
# ----------------------------
# Replace these with the name of the executable 
# and the parameters it needs
# ---------------------------
#$ -S /bin/bash

export MYAPP=./dbg/LMR_serial
export MYAPP_FLAGS='' 

. /etc/profile.d/modules.sh

module load  intel/fortran openmpi/intel/64 nag/f90


# ---------------------------
# set the name of the job
#$ -N ed1_v3
                                                                                # Redirect output stream to this file.
#$ -o lmr_serial.out.dat

# Redirect error stream to this file.
#$ -e lmr_serial.err.dat
                              
#################################################################
#################################################################
# there shouldn't be a need to change anything below this line



#----------------------------
# set up the parameters for qsub
# ---------------------------

#  Mail to user at beginning/end/abort/on suspension
#$ -m beas
#  By default, mail is sent to the submitting user 
#  Use  $ -M username    to direct mail to another userid 
#$ -M jeremy.lise@gmail.com

# Execute the job from the current working directory
# Job output will appear in this directory
#$ -cwd
#   can use -o dirname to redirect stdout 
#   can use -e dirname to redirect stderr

#to request resources at job submission time 
# use #-l resource=value
# For instance, the commented out 
# lines below request a resource of 'express'
# and a hard CPU time of 10 minutes 
####$ -l express
####$ =l h_cpu=10:00

#  Export these environment variables
#$ -v PATH 



export PATH=$TMPDIR:$PATH


# ---------------------------
# run the job
# ---------------------------

$MYAPP $MYAPP_FLAGS
