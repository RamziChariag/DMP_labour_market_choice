program Msw
  use params
  use output
  use glob
  use objective_function_mod
  use optimization_mod
  use data_statistics_mod

  ! use planner_objective
  ! use tax_objective
  ! use minw_objective
  ! use sev_objective

  implicit none
  real(wp) :: theta(nparams) , param_start(nparams), best_theta(nparams) ,value
  real(wp) :: theta_rescaled(nparams)
  type (ExogenousParameters) :: p

  real(wp), allocatable :: theta_planner(:)
  logical :: planner_switch, tax_switch, MIN_WAGE_SWITCH, base_switch, SEVERENCE_SWITCH
  real(wp), allocatable :: theta_tax(:)

  real(wp), allocatable :: minw(:) ! minimum wage
  real(wp), allocatable :: theta_sev(:) ! severence pay  and minimum wage
  integer i, j

  ! read in starting values for parameters from disc
  call readFromCSVvec('data/starting_UI.raw', theta )
  write(*,*) "Starting Values: "
  call printVec(theta)

  




  ! INIT PARAMETERS
  call Init(p)
  ! turn ON printing and saving of simulated moments etc
  GLOBAL_DISPLAY_SWITCH = .TRUE.

  ! INITIALIZE THE VECTOR OF RANDOM NUMBERS 
  ! (need to be the same for each call across all processors)
  allocate( global_rand_x_r(p%nWorkerSimulation) )
  allocate( global_rand_delta_rt(p%nWorkerSimulation,p%nPeriodSimulation) )
  allocate( global_rand_zeta_rt(p%nWorkerSimulation,p%nPeriodSimulation) )
  allocate( global_rand_lambda_rt(p%nWorkerSimulation,p%nPeriodSimulation) )
  allocate( global_rand_y_rt(p%nWorkerSimulation,p%nPeriodSimulation) )
  
  allocate( global_merror(p%nWorkerSimulation,p%momentsMonths) )

  !--------------------------------------
  !           GENERATE DRAWS
  !--------------------------------------
  call initSeed_lmr(1234567)
  
  call randVec_lmr( global_rand_x_r(1:p%nWorkerSimulation/2) )
  global_rand_x_r(1+p%nWorkerSimulation/2:p%nWorkerSimulation) = &
       1.0_wp - global_rand_x_r(1:p%nWorkerSimulation/2)
  
  call randMat_lmr( global_rand_delta_rt )
  call randMat_lmr( global_rand_zeta_rt )
  call randMat_lmr( global_rand_lambda_rt )
  call randMat_lmr( global_rand_y_rt )
  
  call randMat_lmr( global_merror )
  ! transform uniform draw to normal mean 0, standard deviation 1 (iid measurement error)

  do i=1,p%nWorkerSimulation
     do j=1,p%momentsMonths
        global_merror(i,j) = norminv( global_merror(i,j) )
     end do
  end do
  
  ! Now, add auto-correlation to measurement error
  do i=1,p%nWorkerSimulation
     do j=2,p%momentsMonths
        !! 2013 revision: we only use annual data now so just make measurement error iid across years and constant within year
        if (mod(j-1,12) .ne. 0) then
           !! global_merror(i,j) = p%corr * global_merror(i,j-1) + global_merror(i,j)
           global_merror(i,j) = global_merror(i,j-1)
        end if
     end do
  end do


  !! finished setting up random draws
  
  base_switch = .TRUE.
  
  if (base_switch) then
     
     write(*,*) "Baseline Version"
     write(*,*) "starting the program"
     write(*,*) "loading the moments into common block"
     
     ! load true moments once
     allocate(GLOBAL_TRUE_MOMENT_MATRIX(nmoments))
     allocate(GLOBAL_TRUE_SD_MATRIX(nmoments))
     ! use this to zero out some moments to get things going
     allocate(GLOBAL_TRUE_COV_MATRIX(nmoments,nmoments))  
     allocate(GLOBAL_TRUE_CHOLESKY_MATRIX(nmoments,nmoments))  
     
     open(13,file='data/mom_ed2.raw')
     read(13,*) (GLOBAL_TRUE_MOMENT_MATRIX(i),i=1,nmoments)
     close(13)
     
     open(13,file='data/se_ed2.raw',form='formatted')
     read(13,*) (GLOBAL_TRUE_SD_MATRIX(i),i=1,nmoments)
     close(13)
     
     ! We no longer use the optimal weiting matrix
     !call readFromCSV('data/COV_shrink_ed3.raw', GLOBAL_TRUE_COV_MATRIX)
     !! Read in the cholesky decomposition of covariacne matrix from data mometns (estimate based on shrinkage)
     !call readFromCSV('data/Choleski_shrink_ed3.raw', GLOBAL_TRUE_CHOLESKY_MATRIX )
     
     ! WRITE(*,*) "GLOBAL_TRUE_CHOLESKY_MATRIX: "
     ! CALL printMat( GLOBAL_TRUE_CHOLESKY_MATRIX )
     
     
     !call readFromCSVvec('data/mom_ed3.raw'  , GLOBAL_TRUE_MOMENT_MATRIX)
     !call readFromCSVvec('data/se_ed3.raw'   , GLOBAL_TRUE_SD_MATRIX)
     ! call readFromCSV('data/n_ed3_v1.raw' , GLOBAL_TRUE_N_MATRIX)
     
     !  call printMat(GLOBAL_TRUE_MOMENT_MATRIX);
     
  ! CALLING THE OBJECTIVE FUNCTION
     ! GLOBAL_DISPLAY_SWITCH = .FALSE.
  GLOBAL_DISPLAY_SWITCH = .TRUE.
  theta = 0.00017_wp
  value = real(objective_function(real(theta,wp),p))

!!$
!!$  write(*,*) "Looping over tax rates.  This is a one parameter policy."
!!$  do i=0,500
!!$     theta = 0.00017_wp + real(i,wp) / 25000000.0_wp
!!$     write(*,*) i, " ", theta(1)
!!$     value = real(objective_function(real(theta,wp),p))
!!$  end do
!!$   
     
  end if
  
  write(*,*) "Finished"
  
  deallocate( global_rand_x_r )
  deallocate( global_rand_delta_rt )
  deallocate( global_rand_zeta_rt )
  deallocate( global_rand_lambda_rt )
  deallocate( global_rand_y_rt )
  deallocate( global_merror )
    
end program Msw




