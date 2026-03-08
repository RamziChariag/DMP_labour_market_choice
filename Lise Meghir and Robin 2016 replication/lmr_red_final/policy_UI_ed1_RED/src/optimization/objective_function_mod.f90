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
  USE nag_lin_lsq, ONLY : nag_lin_lsq_sol
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
    
    REAL(wp) :: RandD_costs

    
    !INIT
    CALL Init(local_p);
    local_p = p
    
    ! Place the new theta guess into the local parameter structure
    local_p%tau            = theta(1)
    
    !STARTING THE PROCEDURE
    
    CALL Init(mymodel, local_p)
    CALL InitSimulationWorkerPanel(s,local_p)
    CALL scaleWages(local_p,MINVAL(mymodel%f_xy), MAXVAL(mymodel%f_xy))
    CALL InitMoments(model_moments) ! this is a structure to facilitate calculations
    
    ! write(*,*) "make sure the data moments are in scope"
    ! call printMat(GLOBAL_TRUE_MOMENT_MATRIX)
    
    !SOLVE THE MODEL
    CALL solveModel(mymodel, local_p, converged)
 
    ! calculate steady state output
    vacancy_cost_xy = local_p%c * sum(mymodel%f_xy) &
         / ( local_p%gridSizeWorkers*local_p%gridSizeFirms )
    
    CALL CreateBenefit( benefit_xy, local_p )
    ! amortized cost of creating new jobs,
    RandD_costs = max(0.0_wp, &
         (local_p%nFirms - local_p%nFirms_decentralized) * local_p%rKost )
    
    
    steady_state_output = SUM( mymodel%f_xy*mymodel%h_xy ) &
         + SUM( benefit_xy(:,1)*mymodel%u_x ) &
         - vacancy_cost_xy(1,1)*mymodel%V - RandD_costs
    
    IF (converged .EQ. 1) THEN
       objective_function =  steady_state_output
    
       
    ELSE
       ! return huge negative number if no convergence at current parameter values
       objective_function = -99999999999.0_wp
    END IF
    
    ! check if trying to return either nan or infinite objective function
    IF ( isnan(objective_function) ) THEN
       objective_function = -88888888888.0_wp
    ELSE IF ( ABS(objective_function) .GT. HUGE(objective_function) ) THEN
       objective_function = -88888888888.0_wp
    END IF

    IF (GLOBAL_DISPLAY_SWITCH) THEN
       ! Save model solution to disk
       CALL writeToCSV("S_xy.dat",mymodel%S_xy)
       CALL writeToCSV("h_xy.dat",mymodel%h_xy)
       CALL writeToCSV("f_xy.dat",mymodel%f_xy)
       CALL writeToCSVVec("W0_x.dat",mymodel%W0_x)
       CALL writeToCSVVec("u_x.dat",mymodel%u_x )
       CALL writeToCSVVec("v_y.dat",mymodel%v_y )
       
       ! print some key aspect of model equilibrium
       PRINT '( a )', "Planner's Labour Market Tightness: "
       PRINT '( a , (ES9.2E2) )', "kappa = " ,mymodel%kappa
       PRINT '( a , (ES9.2E2) )', "U     = " ,mymodel%U
       PRINT '( a , (ES9.2E2) )', "V     = " ,mymodel%V
       

       WRITE(*,*) "NOTE, THESE NUBERS DO NOT INCLUDE THE TRANSFERS"
       WRITE (*,*) "UI policy steady-state output= " &
            , steady_state_output
       WRITE (*,*) "Market Production                = " &
            , SUM( mymodel%f_xy*mymodel%h_xy )
       WRITE (*,*) "Home Production                  = " &
            , SUM( benefit_xy(:,1)*mymodel%u_x )
       WRITE (*,*) "Recruiting costs                 = " &
            , - vacancy_cost_xy(1,1)*mymodel%V
       WRITE (*,*) "R & D costs                      = " &
            , - RandD_costs 
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

       ! calculate expected profit at y=0
       WRITE (*,*), "Expected Profit at y(0)        = ",&
            mymodel%PI0_y(1) + (local_p%delta/local_p%r) &
            * (sum(mymodel%PI0_y)/real(local_p%gridSizeFirms,wp))

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
         

          WRITE (*,*) "Tax Rate:      ", local_p%tau 
          WRITE (*,*) "Benefit Rate:  ", local_p%benefit_UI

    END IF
    
    
!!! NEED TO DEALLOCATE ALL THESE ARRAYS
    CALL FreeExogenousParameters(local_p);
    CALL FreeModel(mymodel)
    CALL freeSimulationWorkerPanel(s)
    CALL freeMoments(model_moments)
    
    WRITE(*,*) "objective function at current values= " , objective_function
    
  END FUNCTION objective_function
  
  FUNCTION moments2vector(moms)
    IMPLICIT NONE
    TYPE (Moments),INTENT(IN) :: moms
    REAL(wp) :: moments2vector(nmoments)
    INTEGER  :: i

    ! NOTE: this code is not very general, it requires that we are using 20 years for each moment
    
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

