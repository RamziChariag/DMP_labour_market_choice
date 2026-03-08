include ../../../makedef

PROG =	dummy

SRCS =	biomath_constants_mod.f90 biomath_interface_mod.f90 \
	biomath_mathlib_mod.f90 biomath_sort_mod.f90 biomath_strings_mod.f90 \
	cdf_aux_mod.f90 cdf_beta_mod.f90 cdf_binomial_mod.f90 \
	cdf_chisq_mod.f90 cdf_f_mod.f90 cdf_gamma_mod.f90 \
	cdf_nc_chisq_mod.f90 cdf_nc_f_mod.f90 cdf_nc_t_mod.f90 \
	cdf_neg_binomial_mod.f90 cdf_normal_mod.f90 cdf_poisson_mod.f90 \
	cdf_t_mod.f90 zero_finder.f90

OBJS =	biomath_constants_mod.o biomath_interface_mod.o biomath_mathlib_mod.o \
	biomath_sort_mod.o biomath_strings_mod.o cdf_aux_mod.o cdf_beta_mod.o \
	cdf_binomial_mod.o cdf_chisq_mod.o cdf_f_mod.o cdf_gamma_mod.o \
	cdf_nc_chisq_mod.o cdf_nc_f_mod.o cdf_nc_t_mod.o \
	cdf_neg_binomial_mod.o cdf_normal_mod.o cdf_poisson_mod.o cdf_t_mod.o \
	zero_finder.o

LIBS =	

CC = cc
CFLAGS = 
FC = f77
FFLAGS = 
F90FLAGS = -O3
# F90FLAGS = -g

LDFLAGS = 

all: $(PROG)

$(PROG): $(OBJS)
	ar -r libcdflib90.a $(OBJS)
	cp *.mod  ../../../dbg
	mv libcdflib90.a  ../../
	rm -f $(PROG) $(OBJS) *.mod

clean:
	rm -f $(PROG) $(OBJS) *.mod

.SUFFIXES: $(SUFFIXES) .f90

.f90.o:
	$(F90) $(F90FLAGS) -c $<

biomath_interface_mod.o: biomath_constants_mod.o biomath_sort_mod.o \
	biomath_strings_mod.o
biomath_mathlib_mod.o: biomath_constants_mod.o
biomath_sort_mod.o: biomath_constants_mod.o
biomath_strings_mod.o: biomath_constants_mod.o
cdf_aux_mod.o: biomath_constants_mod.o biomath_interface_mod.o zero_finder.o
cdf_beta_mod.o: biomath_constants_mod.o biomath_mathlib_mod.o cdf_aux_mod.o \
	zero_finder.o
cdf_binomial_mod.o: biomath_constants_mod.o cdf_aux_mod.o cdf_beta_mod.o \
	zero_finder.o
cdf_chisq_mod.o: biomath_constants_mod.o cdf_aux_mod.o cdf_gamma_mod.o \
	zero_finder.o
cdf_f_mod.o: biomath_constants_mod.o cdf_aux_mod.o cdf_beta_mod.o \
	zero_finder.o
cdf_gamma_mod.o: biomath_constants_mod.o biomath_mathlib_mod.o cdf_aux_mod.o \
	zero_finder.o
cdf_nc_chisq_mod.o: biomath_constants_mod.o biomath_mathlib_mod.o \
	cdf_aux_mod.o cdf_chisq_mod.o zero_finder.o
cdf_nc_f_mod.o: biomath_constants_mod.o biomath_mathlib_mod.o cdf_aux_mod.o \
	cdf_beta_mod.o cdf_f_mod.o cdf_gamma_mod.o zero_finder.o
cdf_nc_t_mod.o: biomath_constants_mod.o biomath_mathlib_mod.o cdf_aux_mod.o \
	cdf_beta_mod.o cdf_normal_mod.o cdf_t_mod.o zero_finder.o
cdf_neg_binomial_mod.o: biomath_constants_mod.o cdf_aux_mod.o cdf_beta_mod.o \
	zero_finder.o
cdf_normal_mod.o: biomath_constants_mod.o biomath_mathlib_mod.o cdf_aux_mod.o
cdf_poisson_mod.o: biomath_constants_mod.o cdf_aux_mod.o cdf_gamma_mod.o
cdf_t_mod.o: biomath_constants_mod.o biomath_mathlib_mod.o cdf_aux_mod.o \
	cdf_beta_mod.o cdf_normal_mod.o zero_finder.o
zero_finder.o: biomath_constants_mod.o
