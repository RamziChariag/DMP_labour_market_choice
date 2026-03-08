module solver
  use modeldef
  use params
  use stat_helper
  use solver_helper
  use glob
  use output
  use compute_surplus_mod
  implicit none


contains
  subroutine solveModel(m,p,e)
    use modeldef
    use params
    implicit none

    type (ExogenousParameters), intent(inout)  :: p
    integer, intent(out) :: e  ! 1 for convergence 0 no convergence
    integer i, j
    real(wp) kappa, Pi0_0, discounting
    real(wp) N_low, N_high

    !--- DEFINING TEMPORARY ARRAYS USED DURING COMPUTATION ----
    real(wp), allocatable  :: IntSmSpdH_y(:)
    real(wp), allocatable  :: IntSpdV_xy(:,:)
    real(wp), allocatable  :: VM1_xy(:,:)
    real(wp), allocatable  :: IntSpmSdV_xy(:,:)
    real(wp), allocatable  :: IbarM1dH_xy(:,:)
    real(wp), allocatable  :: r_FirmOppCost_star_xy(:,:)
    real(wp), allocatable  :: r_chi_WorkerOppCost_xy(:,:)
    real(wp), allocatable  :: Snew_xy(:,:)
    real(wp), allocatable  :: u_rep_xy(:,:) , v_rep_xy(:,:)
    real(wp), allocatable  :: RHS_xy(:,:)
    real(wp) tol1, tol2, tol3, tol4, tol5 
    real(wp), allocatable :: vacancy_cost_xy(:,:), benefit_xy(:,:)

    !-------- DEFINING POINTERS FOR THE MODELS ---------
    type (model), target 		:: m_tmp 
    type (model), target, intent(inout) :: m
    type (model), pointer 		:: m_old , m_new , m_swap

    !-------- ALLOCATING THE ARRAYS ---------
    call Init(m_tmp,p) 
    allocate(	IntSmSpdH_y			(p%gridSizeFirms))
    allocate(	r_FirmOppCost_star_xy	(p%gridSizeWorkers,p%gridSizeFirms))
    allocate(	r_chi_WorkerOppCost_xy	(p%gridSizeWorkers,p%gridSizeFirms))

    allocate(	IntSpdV_xy		(p%gridSizeWorkers,p%gridSizeFirms))
    allocate(	VM1_xy			(p%gridSizeWorkers,p%gridSizeFirms))
    allocate(	IntSpmSdV_xy	        (p%gridSizeWorkers,p%gridSizeFirms))
    allocate(	Snew_xy			(p%gridSizeWorkers,p%gridSizeFirms))
    allocate(	IbarM1dH_xy		(p%gridSizeWorkers,p%gridSizeFirms))
    allocate(	RHS_xy			(p%gridSizeWorkers,p%gridSizeFirms))
    allocate(	u_rep_xy		(p%gridSizeWorkers,p%gridSizeFirms))
    allocate(	v_rep_xy		(p%gridSizeWorkers,p%gridSizeFirms))

    allocate( vacancy_cost_xy(p%gridSizeWorkers,p%gridSizeFirms) )
    allocate( benefit_xy(p%gridSizeWorkers,p%gridSizeFirms) )

    m%id = 10 
    m_tmp%id = 11 

    !--------------- DEBUG SETTINGS -------------
    !set the first surpuls to random to test equations

    !call generateArray(0.6_wp,m_old%S_xy)
    !m_old%S_xy = 10.0_wp * m_old%S_xy + 5.0_wp 
    !call generateDistibutionVec(0.6_wp , m_old%v_y)
    !call generateDistributionArray(0.5_wp , m_old%h_xy)
    !call printMat(m_old%S_xy)
    !call printMat(m_old%h_xy)


    !-------- PREPARATION BEFORE THE LOOP ---------

    m_new=>m_tmp
    m_old=>m

    m_old%v = sum(m_old%v_y)
    m_old%M_xy =  merge(1.0_wp, 0.0_wp, m_old%S_xy >= 0.0_wp)
    m_new%M_xy =  merge(1.0_wp, 0.0_wp, m_new%S_xy >= 0.0_wp)

    !-------- CREATE MATRICES FOR COST OF CREATING A VACANCY AND WORKER UNEMPLOYMENT BENEFIT -

    ! call CreateVacancyCost( vacancy_cost_xy, p )

    ! Make vacancy cost proprtional to mean output of the production function
    vacancy_cost_xy = p%c * sum(m_old%f_xy) &
         / ( p%gridSizeWorkers*p%gridSizeFirms )


    call CreateBenefit( benefit_xy, p )

    ! dump to file to look at
    ! call writeToCSV('benefit_xy_test.dat', benefit_xy)
    ! call writeToCSV('vacancy_cost_xy_test.dat', vacancy_cost_xy)


	!    print *, "printing vacancy cost"
	!    call printMat(vacancy_cost_xy)
	!    print *, "printing benefit"
	!    call printMat(benefit_xy)

    !##################################################
    !-------- Firm size loop
    !##################################################

     N_low  = 0.5_wp * p%nWorkers 
     N_high = 2.0_wp * p%nWorkers 

     do j = 1, 2000
        
        !##################################################
        !-------- VALUE FUNCTION ITERATION LOOP ---------
        !##################################################
        e = 0 ! set no convergence as default result
        do i = 1, 1000  ! If change this it needs to be changed below when setting converged
           !    print *, 'Iteration: '
           !    print '(i5)', (i)
           
           ! m_new is updated version of m
           
           u_rep_xy = spread(m_old%u_x, 2 , p%gridSizeFirms)
           v_rep_xy = spread(m_old%v_y, 1 , p%gridSizeWorkers)
           
           !----------------------------------------------------
           ! kappa is an equilibrium parameter that incorporates the matching
           ! function AND ensures that expectations are taken using a probability
           ! measure (this is done through the denominator)
           !----------------------------------------------------
           kappa = matchingFunction(p%s0 * m_old%u &
                + p%s1 * (p%nWorkers-m_old%u), m_old%v, p) &
                / ((p%s0 * m_old%u + p%s1 * (p%nWorkers-m_old%u)) * m_old%v)
           m_new%kappa = kappa
           
           !       print *, "printing kappa"
           !       print '(f10.8)', (m_new.kappa)
           
           !----------------------------------------------------
           !               PRILIMINARY INTEGRALS
           !----------------------------------------------------
           ! res(y) = \int_{(x,y')|S(x,y)>S(x,y')} [S(x,y)-S(x,y')] h(x,y') dx dy'
           call IntSmSpdHxy(m_old%S_xy, m_old%h_xy , IntSmSpdH_y ) 
           ! res(x,y) = \int_{y'|S(x,y')>S(x,y)} S(x,y') v(y') dy'
           call IntSdVy(m_old%s_xy, m_old%v_y , IntSpdV_xy) 
           ! res(x,y) = \int_{y'|S(x,y')>S(x,y)} v(y') dy'
           call VM1(m_old%s_xy, m_old%v_y, VM1_xy) 
           ! 
           ! I do this in two parts because I will resule VM1 in the transition equation
           IntSpmSdV_xy = ( IntSpdV_xy - VM1_xy*m_old%s_xy ) 
           
           !       print *, "printing IntSmSpdH_y"
           !       call printVec(IntSmSpdH_y)
           !       print *, "printing IntSpmSdV"
           !       call printMat(IntSpmSdV_xy)
           
           !---------------------------------------------------
           !               FIRM OPPORTUNITY COST
           !---------------------------------------------------
           
           ! NOTE THIS IS r\Pi_0(y) - delta\int [ \Pi_0(y')-\Pi_0(y)]dy'
           r_FirmOppCost_star_xy =  - vacancy_cost_xy  &
                +  p%s0 * kappa * (1.0_wp - p%beta) &
                * matmul(transpose(u_rep_xy) , ( m_old%M_xy * m_old%S_xy)) &
                + p%s1 * kappa * (1.0_wp - p%beta) * spread( IntSmSpdH_y, 1 , p%gridSizeWorkers)
           
           !---------------------------------------------------
           !               WORKER OPPORTUNITY COST
           !---------------------------------------------------
           r_chi_WorkerOppCost_xy = benefit_xy &
                +  p%s0 * kappa * p%beta &
                *  matmul( m_old%M_xy * m_old%S_xy ,  transpose(v_rep_xy)) 
           
           r_chi_WorkerOppCost_xy = max(r_chi_WorkerOppCost_xy, 0.0_wp)
           m_new%W0_x = r_chi_WorkerOppCost_xy(:,1) / ( p%r + p%chi ) 
           
           !       print *, "printing FirmOppCost_xy"
           !       call printMat(FirmOppCost_xy)
           !       print *, "printing r_chi_WorkerOppCost_xy"
           !       call printMat(r_chi_WorkerOppCost_xy)
           
           !---------------------------------------------------
           !            NEW SURPLUS USING RELAXATION
           !---------------------------------------------------

           ! NOTE THE 1 in: (1+r+chi+delta+zeta) comes from adding S_xy to both 
           ! the LHS and the  RHS
           ! This puts the surplus function into a form that Satisfies 
           ! Balacwell's sufficient cond. and gurentees convergence.  
           ! The alternative version (in continusoud time) is not
           ! necessarily gurenteed to converge given the iteration scheme.
           ! This is soely to help the numerical stability of the iterative 
           ! algorithm. 
           discounting = 1.0_wp / (1.0_wp + p%r + p%chi + p%delta + p%zeta )

           Snew_xy = discounting * (m_old%f_xy &                ! flow output
                - r_chi_WorkerOppCost_xy & ! worker opportunity cost
                - r_FirmOppCost_star_xy &  ! firm opportunity cost
                + p%delta * matmul(m_old%M_xy * m_old%S_xy ,m_old%Q_yy) &  ! expected change due to prod. shock
                + p%beta*p%s1*kappa*IntSpmSdV_xy &   ! expected gain (to worker) from moving
                + m_old%S_xy )

           m_new%S_xy = (p%relax_s) * m_old%S_xy + (1.0_wp - p%relax_s) * Snew_xy 
           
           !---------------------------------------------------
           !              UPDATE THE MATCHING SET
           !---------------------------------------------------
           m_new%M_xy =  merge(1.0_wp, 0.0_wp,m_new%S_xy >= 0.0_wp)
           m_new%QbarM_xy = matmul( (1.0_wp-m_new%M_xy) , m_old%Q_yy ) 
           
           !       print *, "Matching set"
           !       call printMat(m_new%M_xy)
           
           
           !---------------------------------------------------
           !                UPDATE DISTRIBUTIONS USING THE NEW SURPLUSES
           !---------------------------------------------------
           call IntbarM1dHyp(m_new%S_xy , m_old%h_xy , IbarM1dH_xy)
           
           RHS_xy = p%delta * m_new%M_xy *matmul( m_old%h_xy , transpose(m_old%Q_yy) ) &  ! new {x,y} from old {x,y'}
                + (  p%s0 * kappa * u_rep_xy * m_new%M_xy * v_rep_xy )    &
                + (  p%s1 * kappa * IbarM1dH_xy * v_rep_xy ) 
           ! new {x,y} matches due to unemployed and employed search
           
           m_new%h_xy = (RHS_xy * m_new%M_xy / ( p%chi + p%delta + p%zeta  &
                + p%s1 * kappa * VM1_xy) ) 
           ! m_new%h_xy = p%relax_h * m_new%M_xy * m_old%h_xy + (1.0_wp-p%relax_h) * m_new%h_xy
           m_new%h_xy = m_new%M_xy * &
                ( p%relax_h * m_old%h_xy + (1.0_wp-p%relax_h) * m_new%h_xy )
           
           !       print *,"Snew_xy"
           !       call printMat(Snew_xy)
           !       print *,"h_xy"
           !       call printMat(m_new%h_xy)
           
           m_new%v_y = max(p%firmsDistribution - sum(m_new%h_xy,1), 0.0_wp) 
           m_new%u_x = max(p%workersDistribution - sum(m_new%h_xy,2) , 0.0_wp ) 
           
           m_new%v_y  = p%relax_v * m_old%v_y + (1.0_wp-p%relax_v) * m_new%v_y 
           m_new%u_x  = p%relax_v * m_old%u_x + (1.0_wp-p%relax_v) * m_new%u_x 
           
           !       print *, "u_x"
           !       call printVec(m_new%u_x)
           !       print *, "v_x"
           !       call printVec(m_new%v_y)
           !       print *, "i_x"
           !       call printVec(m_new%i_y)
           !       pause
           
           !---------------------------------------------------
           !                  		DONE
           !---------------------------------------------------
           
           m_new%v = sum(m_new%v_y) 
           m_new%u = sum(m_new%u_x) 

           m_new%E_rate = sum(m_new%h_xy) / sum(p%workersDistribution)

           m_new%Pi0_y = (1.0_wp/(p%r+p%delta)) * r_FirmOppCost_star_xy(1,:) &
                + (p%delta/(p%r*(p%r+p%delta))) &
                * (sum(r_FirmOppCost_star_xy(1,:))/real(p%gridSizeFirms,wp))
                      
           tol1 = maxval(  abs(m_new%S_xy - m_old%S_xy)/ maxval(abs(m_new%S_xy))   )**2
           tol2 = maxval(  abs(m_new%h_xy - m_old%h_xy)/ maxval(abs(m_new%h_xy))   )**2
           tol3 = maxval(  abs(m_new%u_x - m_old%u_x)  / maxval(abs(m_new%u_x))    )**2
           tol4 = ( abs( sum(p%workersDistribution) - sum(m_new%h_xy) & 
                - sum(m_new%u_x) )/(sum(p%workersDistribution)) )
           tol5 = ( abs( sum(p%firmsDistribution) - sum(m_new%h_xy) & 
                - sum(m_new%v_y) )/(sum(p%firmsDistribution)) )
           
           ! ------------- PRINTING -------------
           
           if (mod(i,100000) == 0) then
              print*, "---------------------------------  LOOPING ----------------------------"
              
              !print *,"Snew_xy"
              !call printMat(Snew_xy)
              !print *,"m_new%S_xy"
              !call printMat(Snew_xy)
              !print *,"h_xy"
              !call printMat(m_new%h_xy)
              !print *,"M_xy"
              !call printMat(m_new%m_xy)
              
              print '( a , (i9) )', "number of iterations " , i
              print '( a , (ES9.2E2) )', "kappa = " ,kappa
              print '( a , (ES9.2E2) )', "new unemployment " , 100.0_wp * m_new%u/real(p%nWorkers,wp)
              print '( a , (10ES9.2E2) )', "does it add up firm side  " , sum(abs(sum(m_new%h_xy,1) + m_new%v_y))
              print '( a , (10ES9.2E2) )', "does it add up worker side" , sum(abs(sum(m_new%h_xy,2) + m_new%u_x))
              print '( a , (ES9.2E2) , a , (ES9.2E2) )', "diff S_xy: " , tol1 , " tol: " ,p%tol_S
              print '( a , (ES9.2E2) , a , (ES9.2E2) )', "diff h_xy: " , tol2 , " tol: " ,p%tol_h
              print '( a , (ES9.2E2) , a , (ES9.2E2) )', "diff u_x : " , tol3 , " tol: " ,p%tol_u
              print '( a , (ES9.2E2) , a , (ES9.2E2) )', "diff v_x : " , tol4 , " tol: " ,p%tol_v
           end if
           
           if ( tol1 < p%tol_S)   then
              if ( tol2 < p%tol_h)   then
                 if ( tol3 < p%tol_u)   then
                    if ( tol4 < p%tol_u) then
                       if ( tol5 < p%tol_v) then
                          
                          !saving the lagged h
                          m_swap => m_new
                          m_new => m_old
                          m_old => m_swap
                          exit
                       end if
                    end if
                 end if
              end if
           end if


           !saving the lagged h
           m_swap => m_new
           m_new => m_old
           m_old => m_swap
        end do
        
        if (i .lt. 1000 ) then
           ! print '( a , (i9), a )', "Surplus solver converged after " , i , " iterations "
           e = 1 ! set convergence flag to 1
        else
           ! print '( a , (i9), a )', "NO CONVERGENCE after " , i , " iterations "
           e = 0 ! set convergence flat to 0
        end if
        
        !-- SAVE THE MOST UP TO DATE MODEL IN m --
        
        !print '( a , (i9) )', "Outer iteration " , j 
        !print '( a , (ES11.4E2) )', "Number of Firms = " , p%nFirms
        !print '( a , (ES11.4E2) )', "r_delta_Pi0 = " , r_delta_Pi0

        if ( abs(N_low - N_high) .lt. p%tol_0 ) then
           exit
        else

           if ( m_new%Pi0_y(1)/(sum(abs(m_new%Pi0_y))/real(p%gridSizeFirms,wp)) &
                .gt. p%tol_0 ) then
              N_low = p%nFirms
              p%nFirms = 0.5_wp * ( N_low + N_high )
              p%firmsDistribution = (real(p%nFirms,wp) / real(p%gridSizeFirms,wp)) 
              m_new%v_y = max(p%firmsDistribution - sum(m_new%h_xy,1), 0.0_wp) 
              m_swap => m_new
              m_new => m_old
              m_old => m_swap
              
              
           else if (  m_new%Pi0_y(1)/ &
                (sum(abs(m_new%Pi0_y))/real(p%gridSizeFirms,wp)) &
                .lt. - p%tol_0 ) then
              N_high = p%nFirms
              p%nFirms = 0.5_wp * ( N_low + N_high )
              p%firmsDistribution = (real(p%nFirms,wp) / real(p%gridSizeFirms,wp)) 
              m_new%v_y = max(p%firmsDistribution - sum(m_new%h_xy,1), 0.0_wp) 
              m_swap => m_new
              m_new => m_old
              m_old => m_swap
           else
              m_swap => m_new
              m_new => m_old
              m_old => m_swap
                            
              exit
           end if
        end if

     end do
     
     !! Save data to file.
     !    call writeToCSV('surplus.dat', m_new%s_xy)
     !    call writeToCSV('matches.dat', min(m_new%h_xy,3.0_wp))
     
     deallocate(	IntSmSpdH_y		)
     deallocate(	r_FirmOppCost_star_xy	)
     deallocate(	r_chi_WorkerOppCost_xy)
     
     deallocate(	IntSpdV_xy		)
     deallocate( VM1_xy )
     deallocate( IntSpmSdV_xy )
     deallocate( Snew_xy )
     deallocate(   IbarM1dH_xy )
     deallocate(   RHS_xy )
     deallocate(   u_rep_xy )
     deallocate(   v_rep_xy )
     deallocate( vacancy_cost_xy )
     deallocate( benefit_xy )
     
     call FreeModel(m_tmp)
     
   end subroutine solveModel
 end module solver
 
