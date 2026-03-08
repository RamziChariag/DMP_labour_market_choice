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
  real(wp) :: simmoments(nmoments)
  real(wp) :: theta_rescaled(nparams)
  type (ExogenousParameters) :: p

  real(wp), allocatable :: theta_planner(:)
  logical :: planner_switch, tax_switch, MIN_WAGE_SWITCH, base_switch, SEVERENCE_SWITCH
  real(wp), allocatable :: theta_tax(:)

  real(wp), allocatable :: minw(:) ! minimum wage
  real(wp), allocatable :: theta_sev(:) ! severence pay  and minimum wage
  integer i, j

  real(wp) :: loop(5)
  ! loop(1) contains parameter index
  ! loop(2) indicator for (1) absolute or (2) relative parameter bounds
  ! loop(3) lower bound
  ! loop(4) upper bound
  ! loop(5) number of grid points

  ! for reading in the parameter file for the identificaion loop
  CHARACTER(len=80) :: filename ! filename to read loop parameters from
  character(len=40) :: Lfilename
  ! to hold obj_fun and moments from loop
  real(wp), allocatable :: identification(:,:)

  call get_command_argument(1, filename)
  WRITE(*,'(A,A)') 'Filename: ', TRIM(filename)
  call readFromCSVvec(TRIM(filename), loop)
  call printVec(loop)

  if (int(loop(2)).eq.1) then
    write(*,*) 'absolute bounds ',loop(3), ' ', loop(4)
  else
    write(*,*) 'relative bounds',loop(3), ' ', loop(4)
  end if 

  allocate(identification(2+nmoments,int(loop(5))))

  ! read in starting values for parameters from disc
  call readFromCSVvec('data/starting_val.raw', theta )
  write(*,*) "Starting Values: "
  call printVec(theta)

  ! Rescale from real line to required interval
  theta_rescaled = theta 

!  write(*,*) "Rescaled Starting Values: "
!  call printVec(theta_rescaled)
  write(*,*) "Model Scaled Values"
  write(*,*) "------------------------------"
  write(*,'("alpha    ",F25.20," ")'),       theta_rescaled(1)
  write(*,'("s1       ",F25.20," ")'),       theta_rescaled(2)
  write(*,'("zeta     ",F25.20," ")'),       theta_rescaled(3)
  write(*,'("f1       ",F25.20," ")'),       theta_rescaled(4)
  write(*,'("f2       ",F25.20," ")'),       theta_rescaled(5)
  write(*,'("f3       ",F25.20," ")'),       theta_rescaled(6) 
  write(*,'("f4       ",F25.20," ")'),       theta_rescaled(7)
  write(*,'("f5       ",F25.20," ")'),       theta_rescaled(8)
  write(*,'("delta    ",F25.20," ")'),       theta_rescaled(9)
  write(*,'("sigma    ",F25.20," ")'),       theta_rescaled(10)
  write(*,'("beta     ",F25.20," ")'),       theta_rescaled(11) 
  write(*,'("b        ",F25.20," ")'),       theta_rescaled(12)
  write(*,'("c        ",F25.20," ")'),       theta_rescaled(13)
  write(*,'("f6       ",F25.20," ")'),       theta_rescaled(14)
  write(*,'("f7       ",F25.20," ")'),       theta_rescaled(15)
  write(*,'("chi      ",F25.20," ")'),       theta_rescaled(16)
  write(*,*) "------------------------------"
  write(*,*) "------------------------------"


  ! INIT PARAMETERS
  call Init(p)
  ! turn ON printing and saving of simulated moments etc
  ! GLOBAL_DISPLAY_SWITCH = .TRUE.
  GLOBAL_DISPLAY_SWITCH = .FALSE.

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
     
    
     !  call printMat(GLOBAL_TRUE_MOMENT_MATRIX);
     
     ! CALLING THE OBJECTIVE FUNCTION
     write(*,*) "calling objective function"
     call objective_function(real(theta_rescaled,wp),p,value,simmoments)
     
     write(*,*) "Value= " , value
     
     write(*,'(A,I2)') "Loop over parameter ", int(loop(1))

     do i=1,int(loop(5))
      !increment parameter of interest
       if (loop(2).eq.1) then
         theta_rescaled(int(loop(1))) &
          = loop(3)*(1.0-(real(i,wp)-1.0)/(loop(5)-1.0)) &
          + loop(4)*((real(i,wp)-1.0)/(loop(5)-1.0))
       else
         theta_rescaled(int(loop(1))) &
          = theta(int(loop(1))) * &
          ( loop(3)*(1.0-(real(i,wp)-1.0)/(loop(5)-1.0)) &
            +  loop(4)*((real(i,wp)-1.0)/(loop(5)-1.0)))
       end if
       write(*,'(A,i5)') 'i: ', i
       write(*,'(A,F15.10)'), 'parameter: ', theta_rescaled(int(loop(1)))
       ! call objective function at current parameter vector
       call objective_function(real(theta_rescaled,wp),p,value,simmoments)
       write(*,*), 'value: ', value
       ! populate identification matrix
       identification(1,i) = theta_rescaled(int(loop(1)))
       identification(2,i) = value
       identification(3:(2+nmoments),i) = simmoments
      end do

      ! save to file
      write(Lfilename, '("data/ident_theta_",i5,".dat")'),10000+int(loop(1))
      write(*,*) 'Saving to ', trim(Lfilename)
      call writetocsv(Lfilename, identification)

  end if
  
  write(*,*) "Finished"
  
  deallocate( global_rand_x_r )
  deallocate( global_rand_delta_rt )
  deallocate( global_rand_zeta_rt )
  deallocate( global_rand_lambda_rt )
  deallocate( global_rand_y_rt )
  deallocate( global_merror )
  deallocate(identification)

end program Msw




