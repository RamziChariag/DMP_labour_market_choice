program MPIexample
  use mpi_mcmc_mod
  implicit none

  integer, parameter :: wp = selected_real_kind(8) !Statement to control the precision of real variables in the program.
  integer, parameter :: N = 5

  type(MCMCParams) :: optimizer_params
  type(MCMCChains) :: ch

  real(wp) :: theta(N)
  real(wp) :: best_theta(N)
  real(wpfmpi) :: theta2(N) , val


  theta =1.0_wp
  optimizer_params%CR = 0.0_wpfmpi
  optimizer_params%parameter_size = N
  optimizer_params%chain_count    = 23    ! number of parrallel chains
  optimizer_params%chain_length   = 1000    ! length of chain
  optimizer_params%delta		  = 4	     ! number of elements in the difference to compute candidates
  optimizer_params%max_iteration  = 100000	 ! number of elements in the difference to compute candidates
  optimizer_params%stat_display_rate = 1000	 ! number of elements in the difference to compute candidates
  optimizer_params%shock_mult_std = 0.01_wpfmpi
  optimizer_params%shock_add_std = 0.02_wpfmpi
  optimizer_params%initial_posterior_value = -10000.0_wpfmpi
  optimizer_params%dump_chain = .true.



  allocate (optimizer_params%prior_lower_bound(optimizer_params%parameter_size))
  allocate (optimizer_params%prior_upper_bound(optimizer_params%parameter_size))

  optimizer_params%prior_lower_bound = -5.0_wpfmpi ! bounds to use for the prior uniform distribution
  optimizer_params%prior_upper_bound = 5.0_wpfmpi ! bounds to use for the prior uniform distribution

  write(*,*) "starting the program"

  call MpiMCMCInit()

  if(procId .ne. 0) then
     ! ONLY ON SLAVES
	 call Slaves_ComputeValues_Loop(optimizer_params,objective_function2)
	 write(*,*) "salve is done"
  else
     !ONLY ON MASTER
     call MCMCChainsInit(ch,optimizer_params)
	 call StartOptimizerMaster(theta ,best_theta,ch, optimizer_params)
	 call MasterReleaseSlaves(optimizer_params)

 	 call writetocsv('chain_vals_1q2.dat',real(transpose(ch%thetas_knl(1,:,:)) ,wpfmpi))
 	 call writetocsv('chain_vals_1.dat',real(transpose(ch%thetas_knl(1,:,:)) ,wpfmpi))
  	 call writetocsv('chain_vals_2.dat',real(transpose(ch%thetas_knl(2,:,:)) ,wpfmpi) )
  	 call writetocsv('chain_vals_3.dat',real(transpose(ch%thetas_knl(3,:,:)) ,wpfmpi))
  	 call writetocsv('chain_vals_4.dat',real(transpose(ch%thetas_knl(4,:,:)) ,wpfmpi))
  	 call writetocsv('chain_vals_5.dat',real(transpose(ch%thetas_knl(5,:,:)) ,wpfmpi))
     call writetocsv('chain_posts.dat',real(transpose(ch%posts_nl(:,:)),wp) )

  end if

  call MpiMCMFinalize()

contains

	subroutine objective_function(theta,val)
		implicit none
		real(wpfmpi), intent(IN):: theta(:)
		real(wpfmpi),intent(OUT) :: val

		val = 10.0_wpfmpi * (1 - 40.0_wpfmpi * ((theta(1) - 0.1_wpfmpi)**2 + (theta(2) - 0.2_wpfmpi)**2)  )

	end subroutine objective_function


	subroutine objective_function2(theta,val)
		implicit none
		real(wpfmpi), intent(IN):: theta(:)
		real(wpfmpi),intent(OUT) :: val
	  val = 2.0_wpfmpi*max(0.0_wpfmpi, 7.0_wpfmpi -sum((theta - 0.4_wpfmpi)**2))
	  val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.1_wpfmpi)**2)))
	  val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.2_wpfmpi)**2)))
	  val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.6_wpfmpi)**2)))
	  val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.7_wpfmpi)**2)))
	  val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.9_wpfmpi)**2)))
	  val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.8_wpfmpi)**2)))

	  val =  4.0_wpfmpi*val
	end subroutine objective_function2


end program MPIexample




