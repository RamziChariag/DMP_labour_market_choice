#######################################################################
#			Lise, Meghir, Robin						  				  #
#																	  #
#	make mpi  - builds the mpi program, then use 					  #
#	qsub qsbu_init_file.sh											  #
#																	  #
#	make all- build the serial program that evaluates for given params#
#######################################################################

#-------------------------------
#      SELECTING COMPILER
#---------------------------------

# LOAD LOCAL DEFINITIONS
# copy the suitable makedef
include makedef

#-------------------------------
#        DEFINING FMPIOPT paths
#---------------------------------
INCS      =  -I../FMPIOpt/inc -I./dbg/
LIB_PATHS =  -L../FMPIOpt/lib -L./lib/
LIBS      =  -lfmpiopt -lcdflib90

#-------------------------------
#        DEFINING FLAGS
#---------------------------------
CP_ARGS_DBG  = -g
LK_ARGS_DBG  = -g

CP_ARGS_FAST  = -O2
LK_ARGS_FAST  = -O2

## uncomment the next 2 lines for optimaized code as default
LK_ARGS = -O2
CP_ARGS = -O2 -fast

## uncomment the next 2 lines for debugging as default
#LK_ARGS = -g -check all -traceback -debug extended
#CP_ARGS = -g -check all -traceback -debug extended

## uncomment the next 2 lines for debugging as default
#LK_ARGS = -g 
#CP_ARGS = -g 

ifeq ($(F90),ifort)
#	@echo 'Setting intel fortran compiler flags'
	CP_ARGS_DBG  = -check all -traceback -g -O0 -debug extended
	LK_ARGS_DBG  = -check all -traceback -g -O0 -debug extended
	CP_ARGS_FAST = -O2 -fast
	LK_ARGS_FAST = -O2
endif

ifeq ($(F90),/usr/local/bin/gfortran)
	CP_ARGS_DBG  = -g
	LK_ARGS_DBG  = -g
	CP_ARGS_FAST = 
	LK_ARGS_DBG  = 
endif

#-------------------------------------------------
#      DEFINING THE COMPLETE LIST OF OBJECTS
#-------------------------------------------------

COMMON_OBJS =	\
	dbg/glob.o \
	dbg/randgen.o \
	dbg/output.o \
	dbg/prob.o \
	dbg/test/test_helper.o \
	dbg/stat_helper.o \
	dbg/array_helper.o \
	dbg/copula.o \
	dbg/sort_mod.o \
	dbg/model/params.o \
	dbg/model/modeldef.o \
	dbg/model/solver_helper.o \
	dbg/model/solver.o \
	dbg/test/test.o \
	dbg/model/compute_surplus_helper_mod.o \
	dbg/model/compute_surplus_mod.o \
	dbg/model/moments_mod.o \
	dbg/model/compute_theoritical_moments_mod.o \
	dbg/simulation/simulate_wages_mod.o \
	dbg/simulation/compute_moments_mod.o \
	dbg/optimization/data_statistics_mod.o \
	dbg/optimization/objective_function_mod.o \
	dbg/optimization/optimization_mod.o 

SERIAL_OBJS = \
	dbg/main_serial.o

HOPSPACK_OBJS = \
	dbg/main_hopspack.o

MPI_OBJS = \
	dbg/main_mpi.o

#---------------------------------
#      TARGET: ALL
#---------------------------------
# all: mpi 
 all : serial-build


mpi: F90 = /share/apps/openmpi-1.8.1-intel-13.0.1/bin/mpif90 
mpi: mpi-build 

serial-build: createdir external dbg/main_serial.o
	@echo 'Linking target: $@'
	@echo 'Invoking: Intel(R) Fortran Linker'
	$(F90) -o "dbg/LMR_serial" $(LK_ARGS) dbg/main_serial.o $(COMMON_OBJS) $(LIB_PATHS) $(LIBS)
	@echo 'Finished building target: $@'	

mpi-build: createdir external dbg/main_mpi.o
	@echo 'Linking target: $@'
	@echo 'Invoking: Intel(R) Fortran Linker'
	$(F90) -o "dbg/LMR_mpi" $(LK_ARGS) $(MPI_OBJS) $(COMMON_OBJS) $(LIB_PATHS) $(LIBS)
	@echo 'Finished building target: $@'	


# ARGUMENTS FOR IFORT
fast: CP_ARGS = $(CP_ARGS_FAST)
fast: LK_ARGS = $(LK_ARGS_FAST)
fast: all

dbg: CP_ARGS = $(CP_ARGS_DBG)
dbg: LK_ARGS = $(LK_ARGS_DBG)
dbg: all

createdir:
	-mkdir -p dbg
	-mkdir -p dbg/model
	-mkdir -p dbg/test
	-mkdir -p dbg/simulation
	-mkdir -p dbg/optimization	
	-mkdir -p dbg/mpi	

#---------------------------------
#      EXTERNAL LIB
#---------------------------------
external: lib/libcdflib90.a

lib/libcdflib90.a: 
	$(MAKE) -C lib/cdflib90/source/

#---------------------------------
#      DEPENDENCIES
#---------------------------------
dbg/main_mpi.o: \
  dbg/glob.o \
  dbg/stat_helper.o \
  dbg/model/modeldef.o \
  dbg/model/params.o \
  dbg/model/solver.o \
  dbg/copula.o \
  dbg/output.o \
  dbg/test/test.o \
  dbg/simulation/simulate_wages_mod.o \
  dbg/simulation/compute_moments_mod.o \
  dbg/optimization/objective_function_mod.o \
  dbg/model/compute_theoritical_moments_mod.o \
  dbg/sort_mod.o 

dbg/main_serial.o: \
  dbg/glob.o \
  dbg/stat_helper.o \
  dbg/model/modeldef.o \
  dbg/model/params.o \
  dbg/model/solver.o \
  dbg/copula.o \
  dbg/output.o \
  dbg/test/test.o \
  dbg/simulation/simulate_wages_mod.o \
  dbg/simulation/compute_moments_mod.o \
  dbg/optimization/objective_function_mod.o \
  dbg/model/compute_theoritical_moments_mod.o \
  dbg/sort_mod.o 

dbg/simulation/compute_moments_mod.o: \
  dbg/model/moments_mod.o \
  dbg/sort_mod.o

dbg/simulation/simulate_wages_mod.o: \
  dbg/stat_helper.o 	

dbg/model/modeldef.o: \
  dbg/model/params.o \
  dbg/copula.o \
  dbg/output.o \
  dbg/stat_helper.o

dbg/stat_helper.o: \
	dbg/glob.o \
	dbg/randgen.o \
	dbg/array_helper.o \
	dbg/prob.o

dbg/model/params.o: \
  dbg/glob.o \
  dbg/array_helper.o

dbg/model/solver.o: \
  dbg/model/compute_surplus_mod.o \
  dbg/model/solver_helper.o \
  dbg/output.o

dbg/model/compute_surplus_mod.o: \
  dbg/model/compute_surplus_helper_mod.o \
  dbg/output.o

dbg/test/test.o: \
  dbg/model/modeldef.o \
  dbg/model/params.o \
  dbg/stat_helper.o \
  dbg/model/solver_helper.o \
  dbg/glob.o \
  dbg/output.o \
  dbg/test/test_helper.o

dbg/optimization/objective_function_mod.o: \
  dbg/glob.o \
  dbg/optimization/data_statistics_mod.o \
  dbg/model/compute_theoritical_moments_mod.o \
  dbg/optimization/optimization_mod.o

dbg/optimization/data_statistics_mod.o: \
  dbg/glob.o \
  dbg/optimization/optimization_mod.o


#---------------------------------
#      module compilation rule
#---------------------------------		
dbg/%.o: src/%.f90
	@echo 'Building: $<'
	$(F90) $(INCS) $(CP_ARGS) -I "./dbg" -c -o "$@" "$<" 

dbg/%.o: src/%.f
	@echo 'Building: $<'
	$(F90) $(INCS) $(CP_ARGS) -I "./dbg" -c -o "$@" "$<" 

#-check all
#---------------------------------
#      cleaning rule
#---------------------------------		
clean:
	rm -f $(COMMON_OBJS) $(SERIAL_OBJS) $(MPI_OBJS)  $(HOPSPACK_OBJS)
	rm -f *.o *.mod 

clean-lib:
	rm -rf *.o *.mod *.a
	rm lib/cdflib90/source/libcdflib90.a

