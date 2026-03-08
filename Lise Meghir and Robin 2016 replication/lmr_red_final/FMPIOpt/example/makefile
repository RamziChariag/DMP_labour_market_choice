include ../makedef.inc

#compile main_mpi
all: 
	$(F90) -o example main_mpi.f90 -L../lib -I../inc -lfmpiopt
	
clean:
	rm example
	rm *.o
	rm *.mod
