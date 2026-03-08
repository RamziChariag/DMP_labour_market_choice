
! This code is copied from Search / Save and we know it works
program MPI_MSW
  use mpi_mcmc_mod
  use params
  use output
  use glob
  use objective_function_mod
  use optimization_mod
  
  
  implicit none
  integer, parameter :: N = 16
  
  type(MCMCParams) :: optimizer_params
  type(MCMCChains) :: ch
  real(wp) :: param_start(N), theta(N)
  real(wp) :: best_theta(N)
  real(wpfmpi) :: theta2(N) , val
  integer i, j
  
  
  ! The structural parameters of the model
  type (ExogenousParameters) :: p
  INTEGER :: converged
  REAL(wp) :: objFun
  
  ! read in starting values for parameters from disc
  call readFromCSVvec('data/starting_val.raw', param_start )
  write(*,*) "Starting Values: "
  call printVec(param_start)
  
  theta = param_start
  ! INIT PARAMETERS
  call Init(p);
  
  ! turn off printing and saving of simulated moments etc
  GLOBAL_DISPLAY_SWITCH = .FALSE.
  ! load true moments oince
  ! load true moments oince
  
  ! load true moments oince
  allocate(GLOBAL_TRUE_MOMENT_MATRIX(nmoments))
  allocate(GLOBAL_TRUE_SD_MATRIX(nmoments))
  ! use this to zero out some moments to get things going
  allocate(GLOBAL_TRUE_N_MATRIX(nmoments))  
  allocate(GLOBAL_TRUE_CHOLESKY_MATRIX(nmoments,nmoments))  
  
  open(13,file='data/mom_ed2.raw')
  read(13,*) (GLOBAL_TRUE_MOMENT_MATRIX(i),i=1,nmoments)
  close(13)
  
  open(13,file='data/se_ed2.raw',form='formatted')
  read(13,*) (GLOBAL_TRUE_SD_MATRIX(i),i=1,nmoments)
  close(13)
  
  !  open(13,file='data/mom_ed3.raw')
  !  read(13,*) (GLOBAL_TRUE_SD_MATRIX(i),i=1,nmoments)
  !  close(13)
    ! To start, scale by the level of data moment.  Note, this will give a very wide cofidence interval for the parameters that is clearly wrong.  We only do this for the exploration phase of the estimation.
  

  ! INITIALIZE THE VECTOR OF RANDOM NUMBERS 
  ! (need to be the same for each call across all processors)
  allocate( global_rand_x_r(p%nWorkerSimulation) )
  allocate( global_rand_delta_rt(p%nWorkerSimulation,p%nPeriodSimulation) )
  allocate( global_rand_zeta_rt(p%nWorkerSimulation,p%nPeriodSimulation) )
  allocate( global_rand_lambda_rt(p%nWorkerSimulation,p%nPeriodSimulation) )
  allocate( global_rand_y_rt(p%nWorkerSimulation,p%nPeriodSimulation) )
  allocate( global_merror(p%nWorkerSimulation,p%momentsMonths) )
  
  ! read random draws so they are gurenteed to be common on all nodes of MPI process
  call readfromcsvvec("./data/random_x.raw", global_rand_x_r )
  call readfromcsv("./data/random_delta.raw", global_rand_delta_rt )
  call readfromcsv("./data/random_zeta.raw", global_rand_zeta_rt )
  call readfromcsv("./data/random_lambda.raw", global_rand_lambda_rt )
  call readfromcsv("./data/random_y.raw", global_rand_y_rt )
  call readfromcsv("./data/random_error.raw", global_merror )
  
!!$  !--------------------------------------
!!$  !           GENERATE DRAWS
!!$  !--------------------------------------
!!$  call initSeed_lmr(1234567)
!!$  
!!$  call randVec_lmr( global_rand_x_r(1:p%nWorkerSimulation/2) )
!!$  global_rand_x_r(1+p%nWorkerSimulation/2:p%nWorkerSimulation) = &
!!$       1.0_wp - global_rand_x_r(1:p%nWorkerSimulation/2)
!!$  
!!$  call randMat_lmr( global_rand_delta_rt )
!!$  call randMat_lmr( global_rand_zeta_rt )
!!$  call randMat_lmr( global_rand_lambda_rt )
!!$  call randMat_lmr( global_rand_y_rt )
!!$  
!!$  call randMat_lmr( global_merror )
!!$  ! transform uniform draw to normal mean 0, standard deviation sigma (iid measurement error)
!!$  
!!$  do i=1,p%nWorkerSimulation
!!$     do j=1,p%momentsMonths
!!$        global_merror(i,j) = p%sigma * norminv( global_merror(i,j) )
!!$     end do
!!$  end do
!!$  
!!$  ! Now, add auto-correlation to measurement error
!!$  do i=1,p%nWorkerSimulation
!!$     do j=2,p%momentsMonths
!!$        ! auto correlated measurment error within year, iid across years
!!$        !! 2013 revision: we only use annual data now so just make measurement error iid across years and constnat within year
!!$        if (mod(j-1,12) .ne. 0) then
!!$           !! global_merror(i,j) = p%corr * global_merror(i,j-1) + global_merror(i,j)
!!$           global_merror(i,j) = global_merror(i,j-1)
!!$        end if
!!$     end do
!!$  end do
!!$  
!!$  !! we will use it multiplicatively with wage level, so take exponential
!!$  global_merror = exp( global_merror ) 
  

  ! we no longer use optimal weights so this is unnecessary
  !! Read in the cholesky decomposition of covariacne matrix from data mometns (estimate based on shrinkage)
  !  call readFromCSV('data/Choleski_shrink_ed3.raw', GLOBAL_TRUE_CHOLESKY_MATRIX )
  write(*,*) 'Finished reading in data moments and covariance matrix ... '
  
  ! INIT MPI
  call MpiMCMCInit()
  
  optimizer_params%CR = 0.33_wpfmpi      ! 25% prob of 2 coeff changing.
  optimizer_params%parameter_size = N
  optimizer_params%chain_count    = 191   ! number of parrallel chains
  optimizer_params%chain_length   = 200  ! length of chain
  optimizer_params%delta    = 2      ! number of elements in the difference to compute candidates
!  optimizer_params%max_iteration  = 1000     ! number of elements in the differ  
 optimizer_params%max_iteration  = 10000     ! number of elements in the differ

  optimizer_params%stat_display_rate = 200   ! frequency of display
!!$  optimizer_params%shock_mult_std = 0.01_wpfmpi   ! use for estimation
!!$  optimizer_params%shock_add_std  = 0.0001_wpfmpi
  optimizer_params%shock_mult_std = 0.001_wpfmpi      ! use for starting values
  optimizer_params%shock_add_std  = 0.00001_wpfmpi
  optimizer_params%initial_posterior_value = -10000000000.0_wpfmpi
  optimizer_params%dump_chain     = .TRUE.
  
  allocate (optimizer_params%prior_lower_bound(optimizer_params%parameter_size))
  allocate (optimizer_params%prior_upper_bound(optimizer_params%parameter_size))
  
  optimizer_params%prior_lower_bound = -20.0_wpfmpi
  optimizer_params%prior_upper_bound =  20.0_wpfmpi

  write(*,*) "starting the program"
  write(*,*) "guess: ", theta
  
  if(procId .ne. 0) then
     ! ONLY ON SLAVES
     call Slaves_ComputeValues_Loop(optimizer_params,objective_function_MSW)
     write(*,*) "slave is done"
  else
     !ONLY ON MASTER
     call MCMCChainsInit(ch,optimizer_params)
     call StartOptimizerMaster(real(theta,wpfmpi) ,best_theta,ch, optimizer_params)
     call MasterReleaseSlaves(optimizer_params)
     
!!$     call writetocsv('chain_vals_s0.dat',real(transpose(ch%thetas_knl(1,:,:)) ,wp))
!!$     call writetocsv('chain_vals_s1.dat',real(transpose(ch%thetas_knl(2,:,:)) ,wp))
!!$     call writetocsv('chain_vals_zeta.dat',real(transpose(ch%thetas_knl(3,:,:)) ,wp))
     
!!$     call writetocsv('chain_vals_rho.dat',real(transpose(ch%thetas_knl(3,:,:)) ,wp))
!!$     call writetocsv('chain_vals_nu.dat',real(transpose(ch%thetas_knl(4,:,:)) ,wp))
!!$     call writetocsv('chain_vals_beta.dat',real(transpose(ch%thetas_knl(5,:,:)) ,wp))
!!$     call writetocsv('chain_vals_b.dat',real(transpose(ch%thetas_knl(6,:,:)) ,wp))
!!$     call writetocsv('chain_vals_c.dat',real(transpose(ch%thetas_knl(7,:,:)) ,wp))
!!$     call writetocsv('chain_vals_f1.dat',real(transpose(ch%thetas_knl(8,:,:)) ,wp))
!!$     call writetocsv('chain_vals_f2.dat',real(transpose(ch%thetas_knl(9,:,:)) ,wp))
!!$     call writetocsv('chain_vals_f3.dat',real(transpose(ch%thetas_knl(10,:,:)) ,wp))
!!$     call writetocsv('chain_vals_f4.dat',real(transpose(ch%thetas_knl(11,:,:)) ,wp))
!!$     call writetocsv('chain_vals_f5.dat',real(transpose(ch%thetas_knl(12,:,:)) ,wp))

     call writetocsv('chain_posts.dat',real(transpose(ch%posts_nl(:,:)),wp) )
  end if
  
  call MpiMCMFinalize()
  
  !saving parameters to files


contains

    subroutine objective_function_MSW(theta,val)
        use params
        use output
        use glob
        use objective_function_mod
        use optimization_mod
        implicit none

        real(wpfmpi), intent(IN):: theta(:)
        real(wpfmpi),intent(OUT) :: val
        real(wpfmpi) :: theta_rescaled(nparams)
        REAL(wp) :: objFun

        ! Rescale from real line to required interval
        theta_rescaled(1)  = exp( theta(1) )                                  ! alpha
        theta_rescaled(2)  = exp( theta(2) )                                  ! s1
        theta_rescaled(3)  = 0.1*exp(theta(3))/(exp(theta(3))+exp(-theta(3))) ! zeta
        theta_rescaled(4)  = exp( theta(4) )                                  ! f1
        theta_rescaled(5)  = exp(theta(5))/(exp(theta(5))+exp(-theta(5)))     ! f2
        theta_rescaled(6)  = 1.0 -  exp( theta(6) )                           ! f3
        theta_rescaled(7)  = exp( theta(7) )                                  ! f4
        theta_rescaled(8)  = exp( theta(8) )                                  ! f5
        theta_rescaled(9)  = 0.5*exp(theta(9))/(exp(theta(9))+exp(-theta(9))) ! delta
        theta_rescaled(10) = exp( theta(10) )                                 ! sigma
        theta_rescaled(11) = exp(theta(11))/(exp(theta(11))+exp(-theta(11)))  ! beta
        theta_rescaled(12) = exp(theta(12))/(exp(theta(12))+exp(-theta(12)))  ! b
        theta_rescaled(13) = exp( theta(13) )                                 ! c
        theta_rescaled(14) = exp( theta(14) )             ! f6
        theta_rescaled(15) = exp( theta(15) )             ! f7
        theta_rescaled(16)  = 0.5*exp(theta(16))/(exp(theta(16))+exp(-theta(16))) ! chi
        
        objFun = objective_function( real(theta_rescaled, wp), p )
        
        val = real(objFun, wpfmpi)
        ! write(*,*) val
      end subroutine objective_function_MSW
      
!! Examples that are useful for debugging

!    subroutine objective_function3(theta,val)
!        implicit none
!        real(wpfmpi), intent(IN):: theta(:)
!        real(wpfmpi),intent(OUT) :: val
!
!        val = 10.0_wpfmpi * (1 - 40.0_wpfmpi * ((theta(1) - 0.1_wpfmpi)**2 + (theta(2) - 0.2_wpfmpi)**2)  )
!
!    end subroutine objective_function3
!
!
!    subroutine objective_function2(theta,val)
!        implicit none
!        real(wpfmpi), intent(IN):: theta(:)
!        real(wpfmpi),intent(OUT) :: val
!      val = 2.0_wpfmpi*max(0.0_wpfmpi, 7.0_wpfmpi -sum((theta - 0.4_wpfmpi)**2))
!      val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.1_wpfmpi)**2)))
!      val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.2_wpfmpi)**2)))
!      val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.6_wpfmpi)**2)))
!      val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.7_wpfmpi)**2)))
!      val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.9_wpfmpi)**2)))
!      val = max(val , max(0.0_wpfmpi, 7.0_wpfmpi -sum( (theta - 0.8_wpfmpi)**2)))
!
!      val =  4.0_wpfmpi*val
!    end subroutine objective_function2


end program MPI_MSW
