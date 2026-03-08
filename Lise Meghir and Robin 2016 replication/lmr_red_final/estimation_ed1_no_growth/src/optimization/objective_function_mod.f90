MODULE objective_function_mod
  USE stat_helper
  USE modeldef
  USE params
  USE solver
  USE output
  USE glob
  USE Compute_Surplus_Mod
  USE Simulate_Wages_Mod
  USE Compute_Moments_Mod
  USE Data_Statistics_Mod
  USE moments_mod
  ! USE nag_lin_lsq, ONLY : nag_lin_lsq_sol
  ! use compute_theoritical_moments_mod

CONTAINS
  
  FUNCTION objective_function(theta,p)
    IMPLICIT NONE
    REAL(wp) :: objective_function
    REAL(wp), INTENT(IN) :: theta(nparams)
    REAL(wp) :: param_guess(nparams)
    TYPE (ExogenousParameters), INTENT(in) :: p
    TYPE (ExogenousParameters) :: local_p
    TYPE (model) :: mymodel
    TYPE (SimulationWorkerPanel) :: s, sb
    TYPE (Moments) :: model_moments
    INTEGER e
    INTEGER i, j, k, ii, jj, kk
    INTEGER :: converged
    
    REAL(wp) :: simMoments(nmoments)
    REAL(wp) :: tmp_array(nmoments)
    REAL(wp) :: DMoments(nmoments)
    REAL(wp) :: sim_Moments_matrix(nmoments,1)
    REAL(wp) :: sim_Moments_matrix_T(1,nmoments)
    REAL(wp) :: W_Matrix(nmoments,nmoments)
    REAL(wp) :: W_tmp(nmoments,nmoments)
    REAL(wp) :: tmp_matrix(nmoments,nmoments)
    REAL(wp) :: obj_tmp(1,1)
    real(wp) :: tmp, tmp2, tmp3, tmp4
    REAL(wp) :: H, E_x, E_y, Var_x, Var_y, Cov_xy
    INTEGER  :: lowest

    REAL(wp) :: sim_COV_matrix(nmoments,nmoments)
    logical  :: update
    REAL(wp) :: MomentFit(nmoments,4)
    
    REAL(wp) :: steady_state_output
    REAL(wp) :: benefit_xy( p%gridSizeWorkers, p%gridSizeFirms )
    REAL(wp) :: vacancy_cost_xy( p%gridSizeWorkers, p%gridSizeFirms )
    
    ! for use in calculating Frictionless benchmarks
    integer,  allocatable :: worker_x(:)
    integer,  allocatable :: firm_y(:)
    real(wp), allocatable :: temp_vector(:)
    integer,  allocatable :: h_x(:)
    integer,  allocatable :: h_y(:)
    
    !INIT
    CALL Init(local_p);
    local_p = p
    
    ! Place the new theta guess into the local parameter structure
    local_p%alpha          = theta(1)
    local_p%s1             = theta(2)
    local_p%zeta           = theta(3)
    local_p%f1             = theta(4)
    local_p%f2             = theta(5)
    local_p%f3             = theta(6) 
    local_p%f4             = theta(7)
    local_p%f5             = theta(8)
    local_p%delta          = theta(9)
    local_p%sigma          = theta(10)
    local_p%beta           = theta(11) 
    local_p%b              = theta(12)
    local_p%c              = theta(13)
    local_p%f6             = theta(14)
    local_p%f7             = theta(15)
    local_p%chi            = theta(16)
       
    !STARTING THE PROCEDURE
    
    CALL Init(mymodel, local_p)
    CALL InitSimulationWorkerPanel(s,local_p)
    CALL scaleWages(local_p,MINVAL(mymodel%f_xy), MAXVAL(mymodel%f_xy))
    CALL InitMoments(model_moments) ! this is a structure to facilitate calculations
    
    ! write(*,*) "make sure the data moments are in scope"
    ! call printMat(GLOBAL_TRUE_MOMENT_MATRIX)
    
    !SOLVE THE MODEL
    CALL solveModel(mymodel, local_p, converged)
    
    IF (converged .EQ. 1) THEN
       
       !if ( 1 ) then ! temporary to see what is goin on
       
       ! computes W1(x,y,w)
       CALL ComputeSurpluses(mymodel, local_p)
                     
       !GENERATE A SIMULATED POPULATION
       CALL SimulateWages(s, mymodel, local_p)
       
       ! COMPUTE THE wage and employment transiton MOMENTS based on simulation

       sim_COV_matrix = 0.0_wp
       update = .FALSE.  ! don't update the covariance for simulated moments.
       ! need to check the simulated wages at this point
       CALL computeMoments(s, local_p, model_moments, sim_COV_matrix, update)
       ! compute the V/U RATIO MOMENT BASED ON THE STATIONARY EQUILIBRIUM OF MODEL
       model_moments%V_U = ( mymodel%V / mymodel%U )

       ! Use the steady-state employment rate as the momed moment
       ! we will match this to the employment rate in years 18-20 in labour force
       !model_moments%E   = mymodel%E_rate

       ! convert the moment structure into a vactor 
       simmoments = moments2vector(model_moments)

       sim_Moments_matrix(:,1) = simMoments
       sim_Moments_matrix_T = transpose(sim_Moments_Matrix)
       

       IF (GLOBAL_DISPLAY_SWITCH) THEN
          
          MomentFit(:,1) = simmoments
          MomentFit(:,2) = GLOBAL_TRUE_MOMENT_MATRIX
          MomentFit(:,3) = GLOBAL_TRUE_SD_MATRIX
          MomentFit(:,4) = (GLOBAL_TRUE_MOMENT_MATRIX - simmoments) &
               / GLOBAL_TRUE_SD_MATRIX
          
          WRITE(*,*) "Easy to read fit: "
          WRITE(*,*) "E: "
          CALL printMat( MomentFit(1:20,:))
          WRITE(*,*) "U2E: "
          CALL printMat( MomentFit(21:40,:))
          WRITE(*,*) "E2U: "
          CALL printMat( MomentFit(41:60,:))
          WRITE(*,*) "J2J: "
          CALL printMat( MomentFit(61:80,:))
          WRITE(*,*) "w: "
          CALL printMat( MomentFit(81:100,:))
          WRITE(*,*) "Dw: "
          CALL printMat( MomentFit(101:120,:))
          WRITE(*,*) "Dw_EE: "
          CALL printMat( MomentFit(121:140,:))
          WRITE(*,*) "Dw_DJ: "
          CALL printMat( MomentFit(141:160,:))
          WRITE(*,*) "w2: "
          CALL printMat( MomentFit(161:180,:))
          WRITE(*,*) "Dw2: "
          CALL printMat( MomentFit(181:200,:))
          WRITE(*,*) "Dw2_EE: "
          CALL printMat( MomentFit(201:220,:))
          WRITE(*,*) "Dw2_DJ: "
          CALL printMat( MomentFit(221:240,:))
          WRITE(*,*) "V_U: "
          CALL printVec( MomentFit(241,:))

          WRITE(*,*) "Objective function using diagonal of COV:   = ", -0.5_wp* SUM( MomentFit(:,4)**2 )

          CALL writeToCSV("Moment_Fit.dat", MomentFit)

          ! Save model solution to disk
          CALL writeToCSV("S_xy.dat",mymodel%S_xy)
          CALL writeToCSV("h_xy.dat",mymodel%h_xy)
          CALL writeToCSV("f_xy.dat",mymodel%f_xy)
          CALL writeToCSVVec("W0_x.dat",mymodel%W0_x)
          CALL writeToCSVVec("Pi0_y.dat",mymodel%Pi0_y)
          CALL writeToCSVVec("u_x.dat",mymodel%u_x )
          CALL writeToCSVVec("v_y.dat",mymodel%v_y )

           ! print some key aspect of model equilibrium
          PRINT '( a )', "Decentralized Labour Market Tightness: "
          PRINT '( a , (ES9.2E2) )', "kappa = " ,mymodel%kappa
          PRINT '( a , (ES9.2E2) )', "U     = " ,mymodel%U
          PRINT '( a , (ES9.2E2) )', "V     = " ,mymodel%V
          
          vacancy_cost_xy = local_p%c * sum(mymodel%f_xy) &
               / ( local_p%gridSizeWorkers*local_p%gridSizeFirms )

          CALL CreateBenefit( benefit_xy, local_p )
          
          steady_state_output = SUM( mymodel%f_xy*mymodel%h_xy ) + SUM( benefit_xy(:,1)*mymodel%u_x ) - vacancy_cost_xy(1,1)*mymodel%V
          
          WRITE (*,*) "Decentralized steady-state output= " &
               , steady_state_output
          WRITE (*,*) "Market Production                = " &
               , SUM( mymodel%f_xy*mymodel%h_xy )
          WRITE (*,*) "Home Production                  = " &
               , SUM( benefit_xy(:,1)*mymodel%u_x )
          WRITE (*,*) "Recruiting costs                 = " &
               , - vacancy_cost_xy(1,1)*mymodel%V
          WRITE (*,*) " "
          WRITE (*,*) "Employment Rate                  = " &
               , 100_wp*(local_p%nWorkers - mymodel%U) / local_p%nWorkers
          WRITE (*,*) "Unemployment Rate                = " &
               , 100_wp*(mymodel%U) / local_p%nWorkers
          WRITE (*,*) " "
          
          ! write (*,*) "THETA: "
          ! call printVec(theta)
          
          WRITE (*,*), "Some additonal Checks:"
          WRITE (*,*), "Measure of employed workers:   = ",  SUM(mymodel%h_xy )
          WRITE (*,*), "Measure of unemployed workers: = ",  SUM(mymodel%u_x )
          WRITE (*,*), "Measure of workers:            = ",  SUM(mymodel%u_x ) &
               + SUM(mymodel%h_xy )
          WRITE (*,*), "Measure of filled jobs:        = ",  SUM(mymodel%h_xy )
          WRITE (*,*), "Measure of vacancies:          = ",  SUM(mymodel%v_y )                
          WRITE (*,*), "Measure of idle posts:         = ",  SUM(mymodel%i_y )
          WRITE (*,*), "Measure of potential jobs:     = ",  SUM(mymodel%h_xy ) &
               + SUM(mymodel%v_y ) + SUM(mymodel%i_y )

          WRITE (*,*), " "
          WRITE (*,*), "E[Pi0_y] :                   = ", SUM(mymodel%PI0_y) / real(local_p%gridSizeFirms,wp) 
          WRITE (*,*), "Pi0_0/E[Pi0_y]    :                   = ", mymodel%PI0_y(1) / (SUM(mymodel%PI0_y) / real(local_p%gridSizeFirms,wp))
          WRITE (*,*), " "
          WRITE (*,*), " "
          
          WRITE (*,*), "Prob of shock to y             = ", local_p%delta

          !!!
          tmp = 0.0_wp
          do i = 1, local_p%gridSizeWorkers
             do j = 1, local_p%gridSizeFirms
                do k = 1, local_p%gridSizeFirms

                   tmp = tmp + mymodel%h_xy(i,j) &
                        * merge(1.0_wp, 0.0_wp, mymodel%S_xy(i,k) .le. 0.0_wp )
                   
                end do
             end do
          end do
          tmp = tmp / (SUM(mymodel%h_xy)*local_p%gridSizeFirms) 

          WRITE (*,*), "Prob of separation | y-shock   = ", tmp

          WRITE (*,*), "Prob of job contact when unempl= ", &
               local_p%s0*mymodel%kappa*mymodel%V       

          !!!
          tmp = 0.0_wp
          do i = 1, local_p%gridSizeWorkers
             do j = 1, local_p%gridSizeFirms
                
                tmp = tmp + mymodel%u_x(i)*mymodel%v_y(j)&
                     * merge(1.0_wp, 0.0_wp, mymodel%S_xy(i,j) .ge. 0.0_wp )
                
             end do
          end do
          tmp = tmp / (mymodel%U * mymodel%V)
   
          WRITE (*,*), "Prob of match | contact        =", tmp
          
          WRITE (*,*), "Prob of job contact when empl  =", &
               local_p%s1*mymodel%kappa*mymodel%V       
          
          !!!
          tmp = 0.0_wp
          do i = 1, local_p%gridSizeWorkers
             do j = 1, local_p%gridSizeFirms
                do k = 1, local_p%gridSizeFirms
                   
                   tmp = tmp + mymodel%h_xy(i,j)*mymodel%v_y(k)&
                        * merge(1.0_wp, 0.0_wp, mymodel%S_xy(i,k) &
                        .gt. mymodel%S_xy(i,j) )

                end do
             end do
          end do
          tmp = tmp / (SUM(mymodel%h_xy) * mymodel%V) 
          WRITE (*,*), "Prob of match | contact        =", tmp

          WRITE (*,*), "--- Worker-Firm Correlation ---"
          H  = 1.0_wp / sum(mymodel%h_xy)  ! normalizing constant
          E_x = 0.0_wp 
          do i=1, local_p%gridSizeWorkers
             do j=1, local_p%gridSizeFirms
                E_x = E_x + i * mymodel%h_xy(i,j)
             end do
          end do
          E_x = E_x * H
          
          E_y = 0.0_wp 
          do i=1, local_p%gridSizeWorkers
             do j=1, local_p%gridSizeFirms
                E_y = E_y + j * mymodel%h_xy(i,j)
             end do
          end do
          E_y = E_y * H

          Var_x = 0.0_wp
          do i=1, local_p%gridSizeWorkers
             do j=1, local_p%gridSizeFirms
                Var_x = Var_x + mymodel%h_xy(i,j)*(i - E_x)**2
             end do
          end do
          Var_x = Var_x * H

          Var_y = 0.0_wp
          do i=1, local_p%gridSizeWorkers
             do j=1, local_p%gridSizeFirms
                Var_y = Var_y + mymodel%h_xy(i,j)*(j - E_y)**2
             end do
          end do
          Var_y = Var_y * H

          Cov_xy = 0.0_wp
          do i=1, local_p%gridSizeWorkers
             do j=1, local_p%gridSizeFirms
                Cov_xy = Cov_xy + mymodel%h_xy(i,j)*(i - E_x)*(j - E_y)
             end do
          end do
          Cov_xy = Cov_xy * H

          WRITE (*,*) "Corr(x,y): ", Cov_xy / sqrt(Var_x * Var_y)
         

          WRITE (*,*), "--- FRICTIONLESS ALLOCATION ---"

          ! allocate and create the vectors of workers and firms in the economy
          allocate( worker_x( int(max(local_p%nWorkers, local_p%nFirms )) ) )
          allocate(   firm_y( int(max(local_p%nWorkers, local_p%nFirms )) ) )  
          allocate( temp_vector(int(max(local_p%nWorkers, local_p%nFirms ))))
          allocate( h_x(local_p%gridSizeWorkers) )
          allocate( h_y(local_p%gridSizeFirms) )


          ! fill with 0, note this is an index and will fail if used.
          worker_x = 0
          firm_y   = 0
          temp_vector = 0.0

          call linspace(temp_vector, local_p%nWorkers, 1.0_wp)
          worker_x = ceiling( real(temp_vector,wp) & 
               / real(local_p%gridSizeWorkers,wp) )

          temp_vector = 0.0
          call linspace(temp_vector(1:int(local_p%nFirms)), &
               local_p%nWorkers, 1.0_wp )
          firm_y = ceiling( real(temp_vector,wp) &
               / real(local_p%gridSizeFirms,wp) )
                    
          tmp2 = 0
          tmp3 = 0
          tmp4 = 0
          
          do i=1, int((local_p%nWorkers ) )
             
             if ( (worker_x(i) .ne. 0) .and. (firm_y(i) .ne. 0) ) then
                
                tmp2 = tmp2 + merge(mymodel%f_xy(worker_x(i),firm_y(i)), 0.0_wp, &
                     mymodel%f_xy(worker_x(i),firm_y(i)) .gt. &
                     benefit_xy(worker_x(i),1) - vacancy_cost_xy(1,1) )
                
                tmp3 = tmp3 + merge( benefit_xy(worker_x(i),1), 0.0_wp, &
                     mymodel%f_xy(worker_x(i),firm_y(i)) .le. &
                     benefit_xy(worker_x(i),1) - vacancy_cost_xy(1,1) )
                
                tmp4 = tmp4 + merge(1.0_wp, 0.0_wp, &
                     mymodel%f_xy(worker_x(i),firm_y(i)) .gt. &
                     benefit_xy(worker_x(i),1) - vacancy_cost_xy(1,1) )
                
             else
                if ( firm_y(i) .eq. 0 .and. worker_x(i) .ne. 0 ) then
                   tmp3 = tmp3 + benefit_xy(worker_x(i),1)
                end if
             end if
             
          end do
          

          WRITE (*,*) "Frictionless steady-state output= " &
               , tmp2 + tmp3 &
               - vacancy_cost_xy(1,1)*max(0.0, real(local_p%nFirms, wp) - tmp4) 

          WRITE (*,*) "Frictionless Market Production                = " &
               , tmp2
          WRITE (*,*) "Frictionless Home Production                  = " &
               , tmp3
          WRITE (*,*) "Frictionless Vacancy Loss                     = " &
               , - vacancy_cost_xy(1,1)*max(0.0, real(local_p%nFirms, wp) - tmp4)
          WRITE (*,*) " "
          WRITE (*,*) "Frictionless Employment Rate                  = " &
               , 100.0_wp * tmp4 / local_p%nWorkers
           WRITE (*,*) " "
 

          WRITE (*,*), "--- FRICTIONLESS ALLOCATION WITH UNEMPLOYMENT---"

          ! re initialize to zero
          worker_x = 0
          firm_y   = 0
          
          do i=1,local_p%gridSizeFirms
             h_y(i) = nint( sum( mymodel%h_xy(:,i) ) )
          end do

          do i=1,local_p%gridSizeWorkers
             h_x(i) = nint(sum(mymodel%h_xy(i,:) ) )
          end do

          jj = 0
          kk = 0
          do i=1,local_p%gridSizeWorkers
             
             jj = kk+1
             kk = jj+ h_x(local_p%gridSizeWorkers+1-i)
             
             worker_x(jj:kk) = local_p%gridSizeWorkers+1-i
             
          end do

          jj = 0
          kk = 0
          do i=1,local_p%gridSizeFirms
             
             jj = kk+1
             kk = jj+ h_x(local_p%gridSizeFirms+1-i)
             
             firm_y(jj:kk) = local_p%gridSizeFirms+1-i
             
          end do

          tmp2 = 0.0
          do i=1, int(max(local_p%nWorkers, local_p%nFirms ) )
             
             if ( (worker_x(i) .ne. 0) .and. (firm_y(i) .ne. 0) ) then
                tmp2 = tmp2 + mymodel%f_xy( worker_x(i), firm_y(i) )
             end if
             
          end do
          

          WRITE (*,*) "Frictionless steady-state market production = " &
               ,  tmp2


          ! allocate and create the vectors of workers and firms in the economy
          deallocate( worker_x )
          deallocate(   firm_y )
          deallocate( h_x )
          deallocate( h_y )

!          ! Also compute wage bounds for (x,y) matches
!!$          call initWorkerBounds(sb, local_p)
!!$          call WagesBounds(sb, mymodel, local_p)
!!$          
!!$          call freeWorkerBounds(sb)
       END IF
       
       
!!$       !COMPUTE THE DISTANCE
!!$       objective_function = -0.5_wp &
!!$            * SUM( ((GLOBAL_TRUE_MOMENT_MATRIX - simmoments) &
!!$            / GLOBAL_TRUE_SD_MATRIX)**2 )


       ! COMPUTE THE DISTANCE USING OPTIMAL WEIGHTING MATRIX
       ! W_tmp = GLOBAL_TRUE_COV_MATRIX + sim_COV_Matrix
       ! W_Matrix = inv(W_tmp)  ! inv(A) is a user written function that inverts the matrix A.  It needs to call LAPACK (here the  MKL version)

       ! vWRITE(*,*) "GLOBAL_TRUE_CHOLESKY_MATRIX: "
       ! CALL printMat( GLOBAL_TRUE_CHOLESKY_MATRIX )
       
       tmp_matrix = GLOBAL_TRUE_CHOLESKY_MATRIX
       ! W_Matrix = GLOBAL_TRUE_COV_MATRIX  ! here we assume we read in the inverse of the COV matrix

       ! WRITE(*,*) "tmp_matrix: "
       ! CALL printMat( tmp_matrix )

       DMoments = simMoments - GLOBAL_TRUE_MOMENT_MATRIX
       ! Only match long run employment, and transision rates
       
       DMoments(1:15)  = 0.0_wp
       DMoments(16:20) = DMoments(16:20) * 4.0_wp * 1000.0_wp ! only employment after 15 years
       DMoments(21:35)  = 0.0_wp
       DMoments(36:40) = DMoments(36:40) * 4.0_wp * 1000.0_wp ! only U2E after 15 years
       DMoments(41:55)  = 0.0_wp
       DMoments(56:60) = DMoments(56:60) * 4.0_wp * 100.0_wp ! only E2U after 15 years
       DMoments(61:75)  = 0.0_wp
       DMoments(76:80) = DMoments(76:80) * 4.0_wp * 100.0_wp ! only J2J after 15 years

       DMoments(81:100) =  DMoments(81:100) * 100.0_wp  ! age profile for wages
       DMoments(161:200) = DMoments(161:200) * 1000.0_wp * 20.0_wp/15.0_wp ! age profile for variacne of wages
       DMoments(161:165) = 0.0_wp                       ! drop the first 5 years of variance     

       ! increase weight on V/U ratio so it has same weight as otehr mometns
       DMoments(201) = DMoments(201) * 20.0_wp ! for 20 years

!!$       tmp_array = DMoments
!!$
!!$       ! sim_Moments_matrix_T = MATMUL(sim_Moments_matrix_T, W_Matrix)
!!$       
!!$       CALL nag_lin_lsq_sol(tmp_matrix, DMoments, tmp_array)
!!$       
!!$       objective_function = -0.5_wp * sum( (tmp_array)**2 )

       ! Use diagonal covaraince weights

       tmp_array = DMoments / GLOBAL_TRUE_SD_MATRIX
       objective_function = -0.5_wp * sum( (tmp_array)**2 )
       
    ELSE
       ! return huge negative number if no convergence at current parameter values
       objective_function = -999999999.0_wp
    END IF
    
    ! check if trying to return either nan or infinite objective function
    IF ( isnan(objective_function) ) THEN
       objective_function = -888888888.0_wp
    ELSE IF ( ABS(objective_function) .GT. HUGE(objective_function) ) THEN
       objective_function = -888888888.0_wp
    END IF
    
    
    !		!! NEED TO DEALLOCATE ALL THESE ARRAYS
    CALL FreeExogenousParameters(local_p);
    CALL FreeModel(mymodel)
    CALL freeSimulationWorkerPanel(s)
    CALL freeMoments(model_moments)
    
    ! WRITE(*,*) "objective function at current values= " , objective_function
    
  END FUNCTION objective_function
  
  FUNCTION moments2vector(moms)
    IMPLICIT NONE
    TYPE (Moments),INTENT(IN) :: moms
    REAL(wp) :: moments2vector(nmoments)
    INTEGER  :: i

    ! NOTE: this code is not very general, it requires that we are using 20 years for each moment and we use the correct order
     
    moments2vector(1:20)    = moms%E 
    moments2vector(21:40)   = moms%U2E 
    moments2vector(41:60)   = moms%E2U
    moments2vector(61:80)   = moms%J2J
    moments2vector(81:100)  = moms%w
    moments2vector(101:120) = moms%Dw
    moments2vector(121:140) = moms%Dw_EE
    moments2vector(141:160) = moms%Dw_DJ
    moments2vector(161:180) = moms%w2
    moments2vector(181:200) = moms%Dw2
    moments2vector(201:220) = moms%Dw2_EE
    moments2vector(221:240) = moms%Dw2_DJ
    moments2vector(241)     = moms%V_U 
    
  END FUNCTION moments2vector


END MODULE objective_function_mod

