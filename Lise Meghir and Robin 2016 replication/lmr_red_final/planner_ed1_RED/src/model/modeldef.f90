module modeldef
  use params
  use stat_helper
  use glob
  use output

  implicit none

  !This represents an equilibrium solution
  type Model
     real(wp),allocatable  :: W1_xyw(:,:,:)  !Equilibirum employed worker surplus
     real(wp), allocatable :: S_xy(:,:)   	 !Equilibrium surplus function
     real(wp), allocatable :: M_xy(:,:)   	 !Equilibrium Matching function
     real(wp), allocatable :: h_xy(:,:)    	 !Equilibrium distribution of matches
     real(wp), allocatable :: f_xy(:,:)    	 !Exogenous Production Function
     real(wp), allocatable :: Q_yy(:,:)      !Exogenous state transiton matrix for y
     real(wp), allocatable :: QbarM_xy(:,:)  !Exogenous state transiton matrix for y
     real(wp), allocatable :: W0_x(:)        !Equilibrium worker value function of being unemployed
     real(wp), allocatable :: PI0_y(:)       !Equilibrium firm value function of being available
     real(wp), allocatable :: u_x(:)   		 !Equilibrium distribution of unemployed types
     real(wp), allocatable :: v_y(:)   		 !Equilibrium distribution of vacant firm types
     real(wp), allocatable :: i_y(:)   		 !Equilibrium distribution of idle firm types
     real(wp), allocatable :: M_minw_xy(:,:) ! Matching set constrained by minimum wage (for policy simulation only)
     real(wp) 				v   			 !Equilibrium number of vacancies
     real(wp) 				u      			 !Equilibrium number of unemployed workers
     real(wp)                                  E_rate                   !steady state employment rate
     real(wp) 				kappa  			 !Equilibrium  market tightness
     integer :: id  !just an id for debug info
     real(wp), allocatable :: M_planner_xy(:,:) !Planner's choice for admissable matches

  end type Model

  interface Init
     module procedure InitModel
  end interface

contains

  subroutine InitModel(m,p)
    implicit none
    type (Model), intent(inout) :: m
    type (ExogenousParameters), intent(in) :: p

    allocate( m%W1_xyw ( p%gridSizeWorkers , p%gridSizeFirms , p%gridSizeWages) )
    allocate( m%S_xy ( p%gridSizeWorkers , p%gridSizeFirms) )
    allocate( m%h_xy ( p%gridSizeWorkers , p%gridSizeFirms) )
    allocate( m%f_xy ( p%gridSizeWorkers , p%gridSizeFirms) )
    allocate( m%M_xy ( p%gridSizeWorkers , p%gridSizeFirms) )
    allocate( m%M_minw_xy ( p%gridSizeWorkers , p%gridSizeFirms) )
    allocate( m%QbarM_xy ( p%gridSizeWorkers , p%gridSizeFirms) )
    allocate( m%Q_yy ( p%gridSizeFirms , p%gridSizeFirms) )
    allocate( m%u_x ( p%gridSizeWorkers) )
    allocate( m%i_y ( p%gridSizeFirms) )
    allocate( m%v_y ( p%gridSizeFirms) )
    allocate( m%W0_x ( p%gridSizeWorkers ) )
    allocate( m%PI0_y ( p%gridSizeWorkers ) )
    allocate( m%M_planner_xy ( p%gridSizeWorkers , p%gridSizeFirms) )

    !----------------------------------------------------
    ! 			  INITIALIZE DISTRIBUTIONS
    !----------------------------------------------------
!!$    m%h_xy =  0.9_wp * p%nFirms * spread( p%firmsDistribution / p%nFirms , 1 , p%gridSizeWorkers ) &
!!$         * transpose(spread( p%workersDistribution / p%nWorkers , 1 , p%gridSizeFirms))
!!$    m%u_x = p%workersDistribution - sum(m%h_xy,2)
!!$    m%v_y = p%firmsDistribution - sum(m%h_xy,1)

    ! read in from estimated decentralized model for starting values
    CALL readFromCSV("data/h_xy.dat",m%h_xy)
    CALL readFromCSVvec("data/u_x.dat",m%u_x)
    CALL readFromCSVvec("data/v_y.dat",m%v_y)
    CALL readFromCSV("data/S_xy.dat",m%S_xy)

    m%i_y = 0.0_wp
    !----------------------------------------------------
    ! 			    INITIALIZE MASSES
    !----------------------------------------------------
    m%u = sum(m%u_x)
    m%v = sum(m%v_y)

    !----------------------------------------------------
    ! 	       INITIALIZE PRODUCTION FUNCTION
    !----------------------------------------------------
    call createQuantileProductionFunction(m%f_xy,p)

    ! call writeToCSV('f_xy_test.dat', m%f_xy)

    !----------------------------------------------------
    ! 	       INITIALIZE SURPLUS FUNCTION
    !----------------------------------------------------
    ! m%S_xy =  m%f_xy / (p%r + p%chi + p%delta + p%zeta)


    !----------------------------------------------------
    ! 	       INITIALIZE MATCHING INDICATOR
    !----------------------------------------------------
    m%M_xy =  merge(1.0_wp, 0.0_wp, m%S_xy >= 0.0_wp)
 

    !----------------------------------------------------
    ! 	       INITIALIZE PLANNER'S MATCHING INDICATOR
    !----------------------------------------------------
    
    m%M_planner_xy =  1.0_wp
    
    !----------------------------------------------------
    ! 	       INITIALIZE TRANSITION MATRIX
    !----------------------------------------------------

    !call createTransactionMatrix(m%Q_yy,p)
    m%Q_yy = 1.0_wp/p%gridSizeFirms


    !----------------------------------------------------
    !          INITIALIZE to NONBINDING MINIMUM WAGE
    !----------------------------------------------------

     m%M_minw_xy = 1.0_wp

	! print *, "Starting Values and some Fixed Functions"

	!	print *,"Q_yy"
	!    call printMat(m%Q_yy)
	!	print *,"f_xy"
	!    call printMat(m%f_xy)
	!
	!	print *,"S_xy"
	!    call printMat(m%S_xy)
	!    print *,"h_xy"
	!    call printMat(m%h_xy)
	!
	!    print *, "u_x"
	!    call printVec(m%u_x)
	!    print *, "v_x"
	!    call printVec(m%v_y)
	!    print *, "i_x"
	!    call printVec(m%i_y)



  end subroutine InitModel

  subroutine FreeModel(m)
    implicit none
    type (Model), intent(inout) :: m

    deallocate( m%W1_xyw  )
    deallocate( m%S_xy )
    deallocate( m%h_xy )
    deallocate( m%f_xy )
    deallocate( m%M_xy )
    deallocate( m%M_planner_xy )
    deallocate( m%M_minw_xy )
    deallocate( m%QbarM_xy )
    deallocate( m%Q_yy )
    deallocate( m%u_x )
    deallocate( m%i_y )
    deallocate( m%v_y )
    deallocate( m%W0_x )
    deallocate( m%PI0_y )

  end subroutine FreeModel

  function matchingFunction(U,V,p)
    !*FD Calculates the matching function for
    !*FD     : u, scalar effective measure of  searchers
    !*FD     : v, scalar measure of vacancies
    !*FD     : X, structure containing model parameters
    !*FDRV     : m, scalar of measure of meetings rate
    implicit none
    real(wp), intent(IN) :: U 
    real(wp), intent(IN) :: V 
    type (ExogenousParameters), intent(IN) :: p
    real(wp) :: matchingFunction

    ! Note:  This matching function is cobb-douglas with constnat
    ! returns to scale

    matchingFunction = p%alpha * ( U ** p%gamma ) * ( V ** (1.0_wp - p%gamma) ) 

  end function matchingFunction


! !!! BETA TYPE VERSION
!  subroutine createQuantileProductionFunction(f_xy,p)
!    implicit none
!    real(wp), intent(INOUT) :: f_xy(:,:)
!    type (ExogenousParameters), intent(IN) :: p
!    integer x,y
!    real(wp) tx, ty    ! temporary variables
!
!    ! x and y are U[0,1], The production function function is CES in the inverse function for the Pareto
!    ! distribution.  One interpretation is the types are Pareto distributed and the production function is CES
!    ! However, all we can really talk about is the marginal product of a quantile, and this is one way to be more flexible.
!
!    ! f(x,y) = ( f1*x^(f5/f2) + f3*y^(f5/f4) )^(1/f5)
!
!    if ( abs(p%f7) .gt. 0.0001_wp ) then
!
!        ! CES productions
!        do x = 1 , p%gridSizeWorkers
!           do y = 1 , p%gridSizeFirms
!
!              f_xy(x,y) = ( ( p%f1*betainv(p%xvals_x(x), p%f2, p%f3) )**p%f7 + ( p%f4*betainv(p%yvals_y(y), p%f5, p%f6) )**p%f7 )**(1/p%f7)
!
!           end do
!        end do
!    else
!        ! COBB-DOUGLAS Production
!        do x = 1 , p%gridSizeWorkers
!           do y = 1 , p%gridSizeFirms
!
!              f_xy(x,y) = ( ( p%f1*betainv(p%xvals_x(x), p%f2, p%f3) ) * ( p%f4*betainv(p%yvals_y(y), p%f5, p%f6) ) )
!
!           end do
!        end do
!    end if
!  end subroutine createQuantileProductionFunction
!
!  subroutine createBenefit(benefit_xy,p)
!    implicit none
!    real(wp), intent(INOUT) :: benefit_xy(:,:)
!    type (ExogenousParameters), intent(IN) :: p
!    integer x,y
!    real(wp) tx, ty    ! temporary variables
!
!    ! x and y are U[0,1], The flow benefit to the worker is interpreted can be interpreted as home production.
!    ! for a given b \in [0,1], the benefit is interpreted as a worker can produce at home as if she were matched with a firm at
!    ! the b-quantile (think McDonalds)
!
!    do x = 1 , p%gridSizeWorkers
!       do y = 1 , p%gridSizeFirms
!
!          benefit_xy(x,y) = ( ( p%f1*betainv(p%xvals_x(x), p%f2, p%f3) )**p%f7 + ( p%f4*betainv(p%b, p%f5, p%f6) )**p%f7 )**(1/p%f7)
!
!       end do
!    end do
!  end subroutine createBenefit
!
!  subroutine createVacancyCost(vacancy_cost_xy,p)
!    implicit none
!    real(wp), intent(INOUT) :: vacancy_cost_xy(:,:)
!    type (ExogenousParameters), intent(IN) :: p
!    integer x,y
!    real(wp) tx, ty    ! temporary variables
!
!    ! The vacancy cost is just a constant for all firm types.  We use the production function here just to keep a reasonable scale
!    ! as the other parameters change
!    !
!    ! Here we scale the posting cost to production between the median worker and the c-quantile firm (this has no interpretation,
!    ! it is a numerical trick to keep relative posting cost constant when moving production and type-distribution parameters around)
!
!    do x = 1 , p%gridSizeWorkers
!       do y = 1 , p%gridSizeFirms
!
!         vacancy_cost_xy(x,y) = ( ( p%f1*betainv(0.5_wp, p%f2, p%f3) )**p%f7 + ( p%f4*betainv(p%c, p%f5, p%f6) )**p%f7 )**(1/p%f7)
!
!       end do
!    end do
!  end subroutine createVacancyCost

! !!! END BETA TYPE VERSION



! !!! PARETO TYPE VERSION
!  subroutine createQuantileProductionFunction(f_xy,p)
!    implicit none
!    real(wp), intent(INOUT) :: f_xy(:,:)
!    type (ExogenousParameters), intent(IN) :: p
!    integer x,y
!    real(wp) tx, ty    ! temporary variables
!
!    ! x and y are U[0,1], The production function function is CES in the inverse function for the Pareto
!    ! distribution.  One interpretation is the types are Pareto distributed and the production function is CES
!    ! However, all we can really talk about is the marginal product of a quantile, and this is one way to be more flexible.
!
!    ! f(x,y) = ( f1*x^(f5/f2) + f3*y^(f5/f4) )^(1/f5)
!
!    if ( abs(p%f7) .gt. 0.0001_wp ) then
!
!        ! CES productions
!        do x = 1 , p%gridSizeWorkers
!           do y = 1 , p%gridSizeFirms
!              f_xy(x,y) = ( p%f1*(p%f2 - p%xvals_x(x))**(-p%f7/p%f3) + p%f4* (p%f5-p%yvals_y(y))**(-p%f7/p%f6) )**(1.0_wp/p%f7)
!           end do
!        end do
!    else
!        ! COBB-DOUGLAS Productio
!        do x = 1 , p%gridSizeWorkers
!		   do y = 1 , p%gridSizeFirms
!		      f_xy(x,y) = ( p%f1*(p%f2 - p%xvals_x(x))**(-1.0_wp/p%f3) ) * ( p%f4*(p%f5-p%yvals_y(y))**(-1.0_wp/p%f6) )
!		   end do
!		end do
!    end if
!  end subroutine createQuantileProductionFunction
!
!  subroutine createBenefit(benefit_xy,p)
!    implicit none
!    real(wp), intent(INOUT) :: benefit_xy(:,:)
!    type (ExogenousParameters), intent(IN) :: p
!    integer x,y
!    real(wp) tx, ty    ! temporary variables
!
!    ! x and y are U[0,1], The flow benefit to the worker is interpreted can be interpreted as home production.
!    ! for a given b \in [0,1], the benefit is interpreted as a worker can produce at home as if she were matched with a firm at
!    ! the b-quantile (think McDonalds)
!
!    do x = 1 , p%gridSizeWorkers
!       do y = 1 , p%gridSizeFirms
!          !benefit_xy(x,y) = ( (p%f1*p%xvals_x(x))**(p%f5/p%f2) + (p%f3*p%b)**(p%f5/p%f4) )**(1.0_wp/p%f5)
!
!          benefit_xy(x,y) = ( p%f1*(p%f2 - p%xvals_x(x))**(-p%f7/p%f3) + p%f4* (p%f5-p%b)**(-p%f7/p%f6) )**(1.0_wp/p%f7)
!
!       end do
!    end do
!  end subroutine createBenefit
!
!  subroutine createVacancyCost(vacancy_cost_xy,p)
!    implicit none
!    real(wp), intent(INOUT) :: vacancy_cost_xy(:,:)
!    type (ExogenousParameters), intent(IN) :: p
!    integer x,y
!    real(wp) tx, ty    ! temporary variables
!
!    ! The vacancy cost is just a constant for all firm types.  We use the production function here just to keep a reasonable scale
!    ! as the other parameters change
!    !
!    ! Here we scale the posting cost to production between the median worker and the c-quantile firm (this has no interpretation,
!    ! it is a numerical trick to keep relative posting cost constant when moving production and type-distribution parameters around)
!
!    do x = 1 , p%gridSizeWorkers
!       do y = 1 , p%gridSizeFirms
!         !vacancy_cost_xy(x,y) = ( (p%f1*0.5_wp)**(p%f5/p%f2) + (p%f3*p%c)**(p%f5/p%f4) )**(1.0_wp/p%f5)
!
!         vacancy_cost_xy(x,y) = ( p%f1*(p%f2 - 0.5_wp)**(-p%f7/p%f3) + p%f4* (p%f5-p%c)**(-p%f7/p%f6) )**(1.0_wp/p%f7)
!
!       end do
!    end do
!  end subroutine createVacancyCost

! !!! END PARETO TYPE VERSION

!!$ !!! LOG NORMAL VERSION
!!$  subroutine createQuantileProductionFunction(f_xy,p)
!!$    implicit none
!!$    real(wp), intent(INOUT) :: f_xy(:,:)
!!$    type (ExogenousParameters), intent(IN) :: p
!!$    integer x,y
!!$    real(wp) tx, ty    ! temporary variables
!!$
!!$    ! x and y are U[0,1], The production function function is CES in the inverse function for the Pareto
!!$    ! distribution.  One interpretation is the types are Pareto distributed and the production function is CES
!!$    ! However, all we can really talk about is the marginal product of a quantile, and this is one way to be more flexible.
!!$
!!$        do x = 1 , p%gridSizeWorkers
!!$          tx = exp( p%f3 * p%f4 * norminv(p%xvals_x(x)) )  
!!$            do y = 1 , p%gridSizeFirms
!!$              ty = exp( p%f3 * p%f5 * norminv(p%yvals_y(y)) ) 
!!$              if ( abs(p%f3) .ge. 0.001 ) then 
!!$                 f_xy(x,y) = exp(p%f1) * ( p%f2*tx + (1.0_wp-p%f2)*ty )**(1.0_wp / p%f3)
!!$              else
!!$                 f_xy(x,y) = exp( p%f1 + p%f2 * p%f4 * norminv(p%xvals_x(x)) + (1.0_wp-p%f2) * p%f5 * norminv(p%yvals_y(y)) ) 
!!$              end if
!!$
!!$           end do
!!$        end do
!!$
!!$  end subroutine createQuantileProductionFunction
!!$
!!$  subroutine createBenefit(benefit_xy,p)
!!$    implicit none
!!$    real(wp), intent(INOUT) :: benefit_xy(:,:)
!!$    type (ExogenousParameters), intent(IN) :: p
!!$    integer x,y
!!$    real(wp) tx, ty    ! temporary variables
!!$
!!$    ! x and y are U[0,1], The flow benefit to the worker is interpreted can be interpreted as home production.
!!$    ! for a given b \in [0,1], the benefit is interpreted as a worker can produce at home as if she were matched with a firm at
!!$    ! the b-quantile (think McDonalds)
!!$
!!$    ty = exp( p%f4 * norminv(p%b) ) 
!!$    do x = 1 , p%gridSizeWorkers
!!$      tx = exp( p%f2 * norminv(p%xvals_x(x)) )  
!!$      do y = 1 , p%gridSizeFirms
!!$         if ( abs(p%f5) .ge. 0.001 ) then 
!!$            benefit_xy(x,y) = exp(p%f1) * ( p%f2*tx + (1.0_wp-p%f2)*ty )**(1.0_wp / p%f3)
!!$         else
!!$            benefit_xy(x,y) = exp( p%f1 + p%f2 * p%f4 * norminv(p%xvals_x(x)) + (1.0_wp-p%f2) * p%f5 * norminv(p%yvals_y(y)) ) 
!!$         end if
!!$      end do
!!$    end do
!!$
!!$  end subroutine createBenefit
!!$
!!$  subroutine createVacancyCost(vacancy_cost_xy,p)
!!$    implicit none
!!$    real(wp), intent(INOUT) :: vacancy_cost_xy(:,:)
!!$    type (ExogenousParameters), intent(IN) :: p
!!$    integer x,y
!!$    real(wp) tx, ty    ! temporary variables
!!$
!!$    ! The vacancy cost is just a constant for all firm types.  We use the production function here just to keep a reasonable scale
!!$    ! as the other parameters change
!!$    !
!!$    ! Here we scale the posting cost to production between the median worker and the c-quantile firm (this has no interpretation,
!!$    ! it is a numerical trick to keep relative posting cost constant when moving production and type-distribution parameters around)
!!$
!!$    ty = exp( p%f3 + p%f4 * norminv(p%c) ) 
!!$    tx = exp( p%f1 + p%f2 * norminv(0.5_wp) ) 
!!$    do x = 1 , p%gridSizeWorkers
!!$       do y = 1 , p%gridSizeFirms
!!$
!!$         vacancy_cost_xy(x,y) = ( tx**p%f5 + ty**p%f5 )**(1.0_wp/p%f5)
!!$
!!$       end do
!!$    end do
!!$  end subroutine createVacancyCost
!!$
!!$! END LOG NORMAL VERSION

!!!!!
!!!!!
 !!! BETA TYPE VERSION - impose thin right tail using definition of beta parameters
  subroutine createQuantileProductionFunction(f_xy,p)
    implicit none
    real(wp), intent(INOUT) :: f_xy(:,:)
    type (ExogenousParameters), intent(IN) :: p
    integer x,y
    real(wp) tx, ty    ! temporary variables
    
    ! x and y are U[0,1], The production function function is CES in the inverse Beta function 
    ! One interpretation is the types are B-distributed and the production function is CES
    ! However, all we can really talk about is the marginal product of a quantile
    
    do x = 1 , p%gridSizeWorkers
       tx = betainv(p%xvals_x(x), p%f4, p%f6)
       do y = 1 , p%gridSizeFirms
          ty = betainv(p%yvals_y(y), p%f5, p%f7) 
          if ( abs(p%f3) .ge. 0.001 ) then 
             f_xy(x,y)=exp(p%f1)*( p%f2*(tx**p%f3)+(1.0_wp-p%f2)*(ty**p%f3))**(1.0_wp/p%f3)
          else
             f_xy(x,y)=exp(p%f1)*(tx**p%f2)*(ty**(1.0_wp-p%f2))
          end if
       end do
    end do
    
  end subroutine createQuantileProductionFunction

  subroutine createBenefit(benefit_xy,p)
    implicit none
    real(wp), intent(INOUT) :: benefit_xy(:,:)
    type (ExogenousParameters), intent(IN) :: p
    real(wp) tx, ty    ! temporary variables
    integer x
    ! Just fix b to be a constant
    ! the following just makes it proportional to median flow output
    ty = betainv(p%b, p%f5, p%f7) 
    
    do x = 1 , p%gridSizeWorkers
       tx = betainv(p%xvals_x(x), p%f4, p%f6)
       if ( abs(p%f3) .ge. 0.001 ) then 
          benefit_xy(x,:) = exp(p%f1)*( p%f2*(tx**p%f3)+(1.0_wp-p%f2)*(ty**p%f3))**(1.0_wp/p%f3)
       else
          benefit_xy(x,:)=exp(p%f1)*(tx**p%f2)*(ty**(1.0_wp-p%f2))
       end if
    end do
    
  end subroutine createBenefit

  subroutine createVacancyCost(vacancy_cost_xy,p)
    implicit none
    real(wp), intent(INOUT) :: vacancy_cost_xy(:,:)
    type (ExogenousParameters), intent(IN) :: p
    real(wp) tx, ty    ! temporary variables
    
    tx = betainv(0.5_wp, p%f4, p%f6)
    ty = betainv(0.5_wp, p%f5, p%f7) 
  
    ! The vacancy cost is just a constant for all firm types. 
    ! the following just makes it proportional to median flow output
    if ( abs(p%f3) .ge. 0.001 ) then 
       vacancy_cost_xy = p%c * exp(p%f1)*( p%f2*(tx**p%f3)+(1.0_wp-p%f2)*(ty**p%f3))**(1.0_wp/p%f3)
    else
       vacancy_cost_xy = p%c * exp(p%f1)*(tx**p%f2)*(ty**(1.0_wp-p%f2))
    end if
  end subroutine createVacancyCost

! END Beta type version

!!!!!
!!!!!



  subroutine createTransactionMatrix(Q_yy,p)
    use copula
    implicit none
    real(wp), intent(INOUT) :: Q_yy(:,:)
    type (ExogenousParameters), intent(IN) :: p
    real(wp) :: weight(size(Q_yy,1),size(Q_yy,2))

    call generateTCopulaMatrix(p%yvals_y,p%rho,p%nu,Q_yy)
    weight = spread( sum( Q_yy, 1) , 2 , size(Q_yy,2))
    Q_yy = transpose(Q_yy / weight)

  end subroutine createTransactionMatrix

end module modeldef
