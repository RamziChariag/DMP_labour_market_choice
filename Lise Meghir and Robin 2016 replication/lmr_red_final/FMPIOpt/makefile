#######################################################################
#								MPI OPTIMIZER
#
#
#######################################################################

#-------------------------------
#      SELECTING COMPILER
#---------------------------------
include makedef.inc

#-------------------------------
#        DEFINING FLAGS
#---------------------------------
## uncomment the next 2 lines for optimaized code as default
LK_ARGS = -O3
CP_ARGS = -O3

INCS     = \
	-I./dbg/ \
	-I./inc/

LIB_PATHS= \
	-L./lib/

#-------------------------------------------------
#      DEFINING THE COMPLETE LIST OF OBJECTS
#-------------------------------------------------

#OBJS =	biomath_constants_mod.o biomath_interface_mod.o biomath_mathlib_mod.o \
#	biomath_sort_mod.o biomath_strings_mod.o cdf_aux_mod.o cdf_beta_mod.o \
#	cdf_binomial_mod.o cdf_chisq_mod.o cdf_f_mod.o cdf_gamma_mod.o \
#	cdf_nc_chisq_mod.o cdf_nc_f_mod.o cdf_nc_t_mod.o \
#	cdf_neg_binomial_mod.o cdf_normal_mod.o cdf_poisson_mod.o cdf_t_mod.o \
#	zero_finder.o \

OBJS = \
	dbg/fmpioptglob.o \
	dbg/util/array_helper.o \
	dbg/util/output.o \
	dbg/util/prob.o \
	dbg/util/randgen.o \
	dbg/util/sort_mod.o \
	dbg/util/rand_helper.o \
	dbg/mpi/mpi_mcmc_mod.o 

#---------------------------------
#      TARGET: ALL
#---------------------------------

all: createdir mpi-lib-build example/example

createdir:
	-mkdir -p lib
	-mkdir -p inc			
	-mkdir -p dbg/mpi
	-mkdir -p dbg/util

mpi-lib-build: lib/libfmpiopt.a 
	@echo 'Done'
	
lib/libfmpiopt.a: $(OBJS)
	ar -r libfmpiopt.a $(OBJS)
	mv libfmpiopt.a  lib
	mv *.mod inc

example/example: lib/libfmpiopt.a ./example/main_mpi.f90
	make -C ./example
#---------------------------------
#      DEPENDENCIES
#---------------------------------


#---------------------------------
#      module compilation rule
#---------------------------------		
dbg/%.o: src/%.f90
	@echo 'Building: $<'
	$(F90) $(INCS) $(CP_ARGS) -c -o "$@" "$<" 

dbg/%.o: src/%.f
	@echo 'Building: $<'
	$(F90) $(INCS) $(CP_ARGS)  -c -o "$@" "$<" 

#-check all
#---------------------------------
#      cleaning rule
#---------------------------------		
clean:
	rm -f $(OBJS)
	rm -f *.o *.mod
	rm inc/*.mod
	rm lib/libfmpiopt.a
