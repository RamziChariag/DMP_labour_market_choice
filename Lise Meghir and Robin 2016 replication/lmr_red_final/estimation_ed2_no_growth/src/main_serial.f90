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
  call readFromCSVvec('data/starting_val.raw', theta )
  write(*,*) "Starting Values: "
  call printVec(theta)

  
!!$  ! Rescale theta from [0,1] to sensible range for this problem
!!$  theta_rescaled(1)  = 0.1 + ( 2.0  - 0.1 ) * theta(1)                 ! alpha
!!$  theta_rescaled(2)  = 0.1 + ( 2.0  - 0.1 ) * theta(2)                 ! s1
!!$  theta_rescaled(3)  = 0.0 + ( 0.05 - 0.0 ) * theta(3)                 ! zeta
!!$  theta_rescaled(4)  = 0.01 + ( 5.0 - 0.01 ) * theta(4)                ! f1
!!$  theta_rescaled(5)  = 0.01 + ( 0.99 - 0.01 ) * theta(5)               ! f2
!!$  theta_rescaled(6)  =-25.0 + ( 1.0 + 25.0 ) * theta(6)                ! f3
!!$  theta_rescaled(7)  = 0.01 + ( 10.0 - 0.01 ) * theta(7)               ! f4
!!$  theta_rescaled(8)  = 0.01 + ( 10.0 - 0.01 ) * theta(8)               ! f5
!!$  theta_rescaled(9)  = 0.0 + ( 0.5 - 0.0 ) * theta(9)                  ! delta
!!$  theta_rescaled(10) = 0.0 + ( 0.5 - 0.0 ) * theta(10)                 ! sigma
!!$  theta_rescaled(11) = 0.0 + ( 1.0 - 0.0 ) * theta(11)                 ! beta
!!$  theta_rescaled(12) = 0.0 + ( 1.0 - 0.0 ) * theta(12)                 ! b
!!$  theta_rescaled(13) = 0.0 + ( 10.0 - 0.0 ) * theta(13)                ! c
!!$  theta_rescaled(14) = theta_rescaled(7) + ( 10.0 - 0.0 ) * theta(14)  ! f6
!!$  theta_rescaled(15) = theta_rescaled(8) + ( 10.0 - 0.0 ) * theta(15)  ! f7

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

!  write(*,*) "Rescaled Starting Values: "
!  call printVec(theta_rescaled)
  write(*,*) "Model Scaled Values"
  write(*,*) "------------------------------"
  write(*,'("alpha    ",F15.10," ")'),       theta_rescaled(1)
  write(*,'("s1       ",F15.10," ")'),       theta_rescaled(2)
  write(*,'("zeta     ",F15.10," ")'),       theta_rescaled(3)
  write(*,'("chi      ",F15.10," ")'),       theta_rescaled(16)
  write(*,'("f1       ",F15.10," ")'),       theta_rescaled(4)
  write(*,'("f2       ",F15.10," ")'),       theta_rescaled(5)
  write(*,'("f3       ",F15.10," ")'),       theta_rescaled(6) 
  write(*,'("f4       ",F15.10," ")'),       theta_rescaled(7)
  write(*,'("f5       ",F15.10," ")'),       theta_rescaled(8)
  write(*,'("f6       ",F15.10," ")'),       theta_rescaled(14)
  write(*,'("f7       ",F15.10," ")'),       theta_rescaled(15)
  write(*,'("delta    ",F15.10," ")'),       theta_rescaled(9)
  write(*,'("sigma    ",F15.10," ")'),       theta_rescaled(10)
  write(*,'("beta     ",F15.10," ")'),       theta_rescaled(11) 
  write(*,'("b        ",F15.10," ")'),       theta_rescaled(12)
  write(*,'("c        ",F15.10," ")'),       theta_rescaled(13)
  write(*,*) "------------------------------"
  write(*,*) "------------------------------"



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

  ! write these to disk to be read in the mpi version
  ! this only needs to be done once, unless we change the length of the simulation
  call writetocsvvec("./data/random_x.raw", global_rand_x_r )
  call writetocsv("./data/random_delta.raw", global_rand_delta_rt )
  call writetocsv("./data/random_zeta.raw", global_rand_zeta_rt )
  call writetocsv("./data/random_lambda.raw", global_rand_lambda_rt )
  call writetocsv("./data/random_y.raw", global_rand_y_rt )
  call writetocsv("./data/random_error.raw", global_merror )

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
     write(*,*) "calling objective function"
     value = real(objective_function(real(theta_rescaled,wp),p))
     
     write(*,*) "Value= " , value
     
     
  end if
  
  write(*,*) "Finished"
  
  deallocate( global_rand_x_r )
  deallocate( global_rand_delta_rt )
  deallocate( global_rand_zeta_rt )
  deallocate( global_rand_lambda_rt )
  deallocate( global_rand_y_rt )
  deallocate( global_merror )
    
end program Msw




