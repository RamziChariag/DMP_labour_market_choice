module params
! This module defines the parameters that will be used in solving the model

  use glob
  use output
  use array_helper
  implicit none

  public
  type ExogenousParameters

     ! Calibrated parameters for Low Skilled workers
     real(wp) s0
     real(wp) s1
     real(wp) zeta

     ! Effective discount rates
     real(wp) r
     real(wp) delta
     real(wp) chi

     ! Parameters of Markov transiton on y: q(y'|y) represented by a t-copula
     real(wp) rho
     real(wp) nu

     ! Parameters of the matching function 
     real(wp) gamma  !parameter in Cobb-Douglas matching function
     real(wp) alpha 	! parameter multiplying matching function

     ! Policy parameter
     real(wp) tau    ! Firing tax

     ! Worker bargaining power
     real(wp) beta 	! Bargaing power of workers

     ! flow cost and benefit of firm and worker
     real(wp) c     	! job posting cost
     real(wp) b     	! home production, produce as if matched to 0.1 quantile of firm distn

     real(wp) mu        ! probability of the market share being cpatured by another firm

     ! The distribution of x and y are not separately identified from the production funciton.
     ! we will use a reduced form for a CES produciton function and log normal distributions for x and y

     ! f(x,y) = f1 ( f2*exp( f3 * f4 * invPhi(x) ) + (1-f2)*exp( f3 * f5 * invPhi(y) ) )^(1/f3)
     real(wp) f1
     real(wp) f2
     real(wp) f3
     real(wp) f4
     real(wp) f5
     real(wp) f6
     real(wp) f7

     ! standard deviation of measurement error used in estimation
     real(wp) sigma 
     ! auto correlation of measurment error between months 
     real(wp) corr 

     real(wp) c_mean  !mean value for distribution of constraints
     real(wp) c_std   !std for the distribution of constraints

     integer gridSizeWorkers      ! number of grid points for worker type distribution
     integer gridSizeFirms        ! number of grid points for firm type distribution
     integer gridSizeWages        ! Number of grid points for wages
     integer gridSizeConstraints  ! Number of grid points for wages

     real(wp) xmin   ! upper and lower bounds of productivity distribution
     real(wp) xmax   ! x and y here will refer to the quantile of the distribution
     real(wp) ymin   ! this min and max means when we work with unbounded distn
     real(wp) ymax   ! (such as log normal) we truncate the bottom and top 1%
	 integer cmin
	 integer cmax
     ! for the purposes of representing it on the computer

     ! Size of labour force and potential number of firms
     real(wp) nWorkers         !25000;            % Size of the labour force
     real(wp) nFirms           !50000;            % Dummy number of firms
     real(wp) nFirms_decentralized ! number of firms from decentralized economy
     real(wp) rKost                ! amortized cost of indreasing firm size
     ! the planner must pay rKost * ( nFirms - nFirms_decentralized ) 
     ! forever to obtain a once and for all increase in the number of 
     ! firms to nFirms

     ! Create grid of worker and firm types
     real(wp) gridWidthWorkers   ! grid width for worker skills
     real(wp) gridWidthFirms   ! grid width for firm productivities

     real(wp), allocatable :: yvals_y(:)  ! vector of quantiles to evaluate ANY distn at
     real(wp), allocatable :: xvals_x(:)  ! This ensures that regardless of the distribution of types,
     ! we always space the grid points with
     ! equal mass between them ( F^-1(x) ).
     real(wp), allocatable :: wvals_w(:)
     integer, allocatable ::  cvals_w(:)

     ! vectors to hold the planner's reservation types
     real(wp), allocatable :: x_reserve(:)
     real(wp), allocatable :: y_reserve(:)

     real(wp) Z    ! It may be easier to control the location of
     ! the mean wage by multiplying the productin
     ! function by X.Z than by moving either the mean
     ! of x or the mean of y
     ! (this turned out not to be the case, so Z is not used)

     real(wp) y
     real(wp), allocatable :: x(:);

     ! Create vector of distribtuion of worker types
     real(wp), allocatable :: workersDistribution(:)

     ! Create vector of distribution of firm productivities
     real(wp), allocatable :: firmsDistribution(:)

     !! Number of periods and workers to simulate
     integer nPeriodSimulation      ! Number of periods to simulate
     integer nWorkerSimulation      ! Number of workers t simulate
     !integer years  ! number of years to use for moments
     integer momentsMonths  ! number of years to use for moments
     integer nConditionalGroups ! number of groups to condition for x;
     
     ! PARAMS FOR FIRM SIMULATION
     integer FirmSimulation_nperiods
     integer FirmSimulation_nfirms
     integer FirmSimulation_nprodlines
     
     !! Numerical Methods Parameters
     ! Relaxation parameter for pseudo-contraction
     real(wp) relax_S
     real(wp) relax_h
     real(wp) relax_v

     ! Tolerance for convervence of surplus function and distributions
     real(wp) tol_S
     real(wp) tol_h
     real(wp) tol_u
     real(wp) tol_v
     real(wp) tol_W
     real(wp) tol_0

     ! Chebychev Basis used for fitting the Planners' matching set
     integer gridSizeBasis
     real(wp), allocatable :: BASIS(:,:)

     real(wp) planner1
     real(wp) planner2
     real(wp) planner3
     real(wp) planner4
     real(wp) planner5
     real(wp) planner6
     real(wp) planner7
     real(wp) planner8
     real(wp) planner9
     real(wp) planner10
     real(wp) planner11
     real(wp) planner12

  end type ExogenousParameters
  
  interface Init
     module procedure InitExogenousParameters
  end interface
contains
  subroutine InitExogenousParameters(p)
    implicit none
    type(ExogenousParameters), intent(INOUT) :: p
    real(wp) :: theta(16), theta_rescaled(16)
    call readFromCSVvec('data/starting_val.raw', theta )
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



    p%s0   =   1.0_wp
    p%s1   =   theta_rescaled(2);
    p%zeta =   theta_rescaled(3);  ! expected working life of 40 years (480 months)

    ! Effective discount rates
    p%r     = 0.05_wp/12.0_wp;     ! discount rate of 5% per year
    p%delta = theta_rescaled(9); ! 4.0_wp/12.0_wp;        ! Poisson probability of productivity shock (expect 1 per quarter)
    p%chi   = theta_rescaled(16)     ! mortality rate (40 year working life)

    ! Parameters of Markov transiton on y: q(y'|y) represented by a t-copula
    !p%rho   =  0.2_wp;
    !p%nu    =  2.0_wp;

    ! Parameters of the matching function (from J-M Robin on aggregate data)
    p%gamma = 0.50_wp;        ! parameter in Cobb-Douglas matching function
    p%alpha = theta_rescaled(1)        ! parameter multiplying matching function

    ! Policy parameter
    p%tau   = 0.0_wp;         ! Firing tax

    ! Worker bargaining power
    p%beta  = theta_rescaled(11);        ! Bargaing power of workers

    ! flow cost and benefit of firm and worker
    p%c     = theta_rescaled(13);        ! job posting cost
    p%b     = theta_rescaled(12);        ! home production, produce as if matched to 0.1 quantile of firm distn

     ! f(x,y) = f1 ( f2*exp( f3 * f4 * invPhi(x) ) + (1-f2)*exp( f3 * f5 * invPhi(y) ) )^(1/f3)
     p%f1   =  theta_rescaled(4)
     p%f2   =  theta_rescaled(5)
     p%f3   =  theta_rescaled(6)
     p%f4   =  theta_rescaled(7)
     p%f5   =  theta_rescaled(8)
     p%f6   =  theta_rescaled(14)
     p%f7   =  theta_rescaled(15)

     
     ! standard deviation of measurement error
     ! p%sigma = 0.3400408900_wp
     p%sigma = theta_rescaled(10)
     p%corr =  0.90_wp

    !distribution of constraints
    !p%c_mean = 20.0_wp
    !p%c_std  = 5.0_wp

    !firm dynamics
    !p%mu = 0.1_wp   !probability of the market share being captured

    p%gridSizeWorkers     = 100          ! number of grid points for worker type distribution
    p%gridSizeFirms       = 100          ! number of grid points for firm type distribution
    p%gridSizeWages       = 25          ! Number of grid points for wages
    ! p%gridSizeWages       = 2000          ! Number of grid points for wages
    ! I use 25 during estimation and 2000 for plotting the wage bands
    p%gridSizeConstraints = 30          ! Number of grid points firm max sizes

!!$    p%gridSizeWorkers     = 10          ! number of grid points for worker type distribution
!!$    p%gridSizeFirms       = 10          ! number of grid points for firm type distribution
!!$    write(*,*) 'REDUCED GRID FOR DEBUGGING'

    p%xmin  = 0.01_wp       ! upper and lower bounds of productivity distribution
    p%xmax  = 0.99_wp       ! x and y here will refer to the quantile of the distribution
    p%ymin  = 0.01_wp       ! this min and max means when we work with unbounded distn
    p%ymax  = 0.99_wp       ! (such as log normal) we truncate the bottom and top 1%
    p%cmin = 0
    p%cmax = 1000 ! for the purposes of representing it on the computer

    !Size of labour force and potential number of firms
    p%nWorkers     =  10000_wp !25000;       ! Size of the labour force
    p%nFirms       =  10145.8710967797_wp   !50000;       ! Dummy number of firms
    p%nFirms_decentralized = 10145.8710967797_wp ! number of firms from estiamtion
    p%rKost        = (0.05_wp/12.0_wp)*39053.4716076489_wp     ! flow cost (permanent) to increase number of firms

    allocate( p%yvals_y( p%gridSizeFirms ) )
    allocate( p%xvals_x( p%gridSizeWorkers ) )
    allocate( p%wvals_w( p%gridSizeWages ) )
    !allocate( p%cvals_c( p%gridSizeConstraints ) )

    allocate( p%y_reserve( p%gridSizeFirms ) )
    allocate( p%x_reserve( p%gridSizeWorkers ) )

    call linspace(p%yvals_y , p%ymin , p%ymax) 	! vector of quantiles to evaluate ANY distn at
    call linspace(p%xvals_x , p%xmin , p%xmax)  ! This ensures that regardless of the distribution of types,
!    call linspace(p%cvals_c , real(p%cmin,wp) , real(p%cmax,wp))  ! This ensures that regardless of the distribution of types,

    ! we always space the grid points with
    ! equal mass between them ( F^-1(x) ).

    p%Z      = 0.0_wp                    ! It may be easier to control the location of
    ! the mean wage by multiplying the productin
    ! function by X.Z than by moving either the mean
    ! of x or the mean of y
    ! (this turned out not to be the case, so Z is not used)

    allocate ( p%workersDistribution ( p%gridSizeWorkers ) )
    allocate ( p%firmsDistribution   ( p%gridSizeFirms ) )

    ! Create vector of distribtuion of worker types
    p%workersDistribution = (real(p%nWorkers,wp) / real(p%gridSizeWorkers,wp));

    ! Create vector of distribuion of firm productivities
    p%firmsDistribution = (real(p%nFirms,wp) / real(p%gridSizeFirms,wp));


    !!! Use for estimation basd on a birth cohrt
    !  Number of periods and workers to simulate
    p%nPeriodSimulation      = 241    ! Number of periods (months) to simulate
    p%nWorkerSimulation      = 10000  ! Number of workers to simulate for estimation 
    
!!$    p%nWorkerSimulation      = 100  ! Number of workers to simulate for estimation 
!!$    write(*,*) 'REDUCED NUMBER OF SIMULATED WORKERS FOR DEBUGGING'

    ! p%nWorkerSimulation      = 400000  ! Number of workers to simulate for simulation of steady state 
    p%momentsMonths  		 = 241     ! number of months to use for moments (drop first 6 months)


!!$    !!! Use to simulate a stationar equilibrium
!!$    p%nPeriodSimulation      = 2400+241  ! Number of periods (months) to simulate
!!$    p%nWorkerSimulation      = 10000  ! Number of workers to simulate for estimation 
!!$    p%momentsMonths  		 = 241     ! number of months to use for moments (drop first 6 months)
!!$
!!$	p%nConditionalGroups     = 1 ! use 5 groups to condition for x
!!$
!!$	p%FirmSimulation_nperiods   = 400
!!$	p%FirmSimulation_nfirms     = 200
!!$	p%FirmSimulation_nprodlines = 20000

    ! Numerical Methods Parameters
    ! Relaxation parameter for pseudo-contraction
    p%relax_S = 0.9_wp;
    p%relax_h = 0.9_wp;
    p%relax_v = 0.9_wp;

    ! Tolerance for convervence of surplus function and distributions
    p%tol_S = 1E-6_wp
    p%tol_h = 1E-6_wp
    p%tol_u = 1E-6_wp
    p%tol_v = 1E-6_wp
    p%tol_W = 1E-6_wp
    p%tol_0 = 1E-3_wp

    ! Basis for Planners's matching set
    p%gridSizeBasis = 11
    allocate ( p%BASIS( p%gridSizeWorkers, p%gridSizeBasis ) )

    p%planner1          = 0.0_wp
    p%planner2          = 0.0_wp
    p%planner3          = 0.0_wp
    p%planner4          = 0.0_wp
    p%planner5          = 0.0_wp
    p%planner6          = 0.0_wp
    p%planner7          = 0.0_wp
    p%planner8          = 0.0_wp
    p%planner9          = 0.0_wp
    p%planner10         = 0.0_wp
    p%planner11         = 0.0_wp
    p%planner12         = 0.0_wp

  end subroutine InitExogenousParameters

  subroutine FreeExogenousParameters(p)
    implicit none
    type(ExogenousParameters), intent(INOUT) :: p

    deallocate( p%yvals_y )
    deallocate( p%xvals_x )
    deallocate( p%wvals_w )
    deallocate ( p%workersDistribution )
    deallocate ( p%firmsDistribution )
    deallocate ( p%BASIS )

    deallocate( p%y_reserve )
    deallocate( p%x_reserve )

  end subroutine FreeExogenousParameters

  !defines a simplified verion of the model
  subroutine simplifyParam(p)
    implicit none
    type(ExogenousParameters), intent(INOUT) :: p
    ! The grid size for workers and firms should be the same size (there is code elsewhere that assumes this to be so)
    p%gridSizeWorkers     = 100;          ! number of grid points for worker type distribution
    p%gridSizeFirms       = 100;          ! number of grid points for firm type distribution
    p%gridSizeWages       = 20;           ! Number of grid points for wages

  end subroutine simplifyParam

  subroutine scaleWages(p,w_min,w_max)
    implicit none
    type(ExogenousParameters), intent(INOUT) :: p
    real(wp), intent(in) :: w_min, w_max

    call linspace(p%wvals_w , log( p%beta * w_min) , log(w_max))
    p%wvals_w = exp(p%wvals_w);

  end subroutine scaleWages


end module params
