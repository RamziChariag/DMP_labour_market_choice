For replication.

This repository contains the code for 
Lise, Jeremy; Costas Meghir and Jean-Marc Robin "Matching, Sorting and Wages" RED

*** ESTIMATION CODE***
The code to estimate the parameters is contained in estimation_ed#_no_growth
- The file data/starting_val.raw contain the starting values for the
MCMC estimation (and now contain the final values)
- The estimation routine can be started in parllel using the script
qsub_script_file.sh, which executes the code in src/main_mpi.f90

*** POST ESTIMATION SIMULATION CODE ***
- At the current parameters, the model can be solved and simulated
using the code in src/main_serial.f90

*** PLANNER SOLUTION ***
The code to compute the planner's solution is contained in the
directories planner_ed#
- The file data/planner_starting_val.raw contains the set of planner's
parameters to start at.  
- The estimated structural parameters are hardcoded into the file
src/model/param.f90
- The estimation routine can be started in parllel using the script
qsub_script_file.sh, which executes the code in src/main_mpi.f90
- At the current parameters, the model can be solved and simulated
using the code in src/main_serial.f90
- this code produces the output used in 
  - Table 3, column 2

*** POLICY UI ***
The code to compute the optimal UI plicy is contained in policy_UI_ed#
- This is a one parameter policy and the solution is found by
looping over policies in the range {A,B] 
- this code produces the output used in 

*** Figures ***
Figures 1 and 2 are produced with the code in 
	estimation_ed#_no_growth/Moment_fit_graphs.do
Figures 3, 4 and 5 are porduced with the code in 
	estimation_ed#_no_growth/plot_model.m
Figure 6 is produced with the code in 
       policy_UI_ed#_RED/policy_effects.m

*** Tables ***
Table 1, Table 2, Table 3 (columns 1, 3 & 4), Table 4 and Table 5 
      are produced from the output of  
      estimation_ed#_no_growth/src/main_serial.f90 
Table 3, column 2 is produced from the output of 
      planner_ed#_RED/src/main_serial.f90
Table 3, column 6 is produced from the output of
      policy_UI_ed#_RED/src/main_serial.f90 
Tables 6 and 7 are produced from running 
       identification_ed#_original/local_identification.m on the outupt from
       identification_ed#_original/src/main_serial.f90

*** NOTES ON COMPILING the Fortran 90 CODE ***
1) set the path to the fortran compiler in makedef 
makedef.ifort, makedef.osx, and makedef.ubuntu)
2) compile the serial version of the code with
make fast
3) compile the MPI version of the code with
make mpi
4) compile with full debugging with
make dbg
5) The code for the parallel MCMC estimation routine is in FMPIOpt,
and can be compiled in the directory FMPIOpt/src with make all.  
Note: this code was written
for and tested on a linux cluster with openmpi, It has not been tested on
any other archetecture or with any other mpi libraries.
6) The code uses some math libraries that need to be compiled within each 
directory ./lib/cdflib90/source with make all

*** DATA Used in Estimation ***
The final moments, along with their standard errors, used in estimation are: 
estimation_ed#_no_growth/data/mom_ed#.raw
estimation_ed#_no_growth/data/se_ed#.raw

These moments were created from the public use NLSY79 micro data.  
The files to construct th moments form the micro data are in the folder:
Create_NLSY79_moments/create_moments/
	1_lmr_trim_data_create_moments_remove_growth.do
	2_bootstrap_se_for_moments.do
	3_create_bs_se_moments.m

The files to extract the sample from the NLSY79 website are
Create_NLSY79_moments/read_data/
	lpv_cross_section/lpv_cross_section.NLSY79
	lpv_weekly_panel/lpv_weekly_panel.NLSY79

The files to create the data form which we create the moments are
Create_NLSY79_moments/read_data/
	lpv_cross_section.do
	lpv_weekly_panel.do
	reshape.do


