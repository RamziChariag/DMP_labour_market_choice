module simulate_firms_mod
! given a distribution of constraints in the economy
! and the value functions, we can simulate a panel of firm
! that have a comple work force and hire / fire / gets productivity  shocks

  use glob
  use modeldef
  use params
  use stat_helper
  use array_helper
  use output

! the idea is to have a pool of production line. When a given production line sproots
! a new one, I use an empty slot and give it to the firm that just created it

 type SimulationFirmPanel
     integer, allocatable :: prod_line_worker_type_lt(:,:)  ! type of the worker in this production line
     integer, allocatable :: prod_line_worker_wage_lt(:,:)  ! wage of the worker in this production line
	 integer, allocatable :: firm_type_ft(:,:)              ! productivity level of the firm in this production line
     integer, allocatable :: prod_line_firmid_lt(:,:)       ! if of the firm owning this production line
	 real(wp), allocatable:: firm_output_ft(:,:)            !

  end type SimulationFirmPanel

contains

  subroutine initSimulationFirmPanel(sf,p)
    implicit none
    type (SimulationFirmPanel) , intent(INOUT) :: sf
    type (ExogenousParameters), intent(in) :: p
	integer :: nf, nt, nl

    nf = p%FirmSimulation_nfirms
    nt = p%FirmSimulation_nperiods
	nl = p%FirmSimulation_nprodlines

	allocate(sf%prod_line_worker_type_lt(nl , nt))
	allocate(sf%prod_line_worker_wage_lt(nl , nt))
    allocate(sf%prod_line_firmid_lt     (nl , nt ))
	allocate(sf%firm_type_ft  (nf , nt ))
	allocate(sf%firm_output_ft(nf , nt ))

  end subroutine initSimulationFirmPanel


  subroutine printFirmToFiles(sf,p,firm_index)
    implicit none
    integer, intent(IN) ::firm_index
    type (SimulationFirmPanel) , intent(INOUT) :: sf
    type (ExogenousParameters),    intent(in)    :: p
    integer :: nl, nt, nf        ! R individuals, T time periods
    integer :: it, il, if        ! R individuals, T time periods
	real(wp),allocatable :: wms(:,:)
	real(wp),allocatable :: tms(:,:)
	real(wp),allocatable :: firm_size_vs_out(:,:)

	nl = size(sf%prod_line_worker_type_lt,1)
	nt = size(sf%prod_line_worker_type_lt,2)

	allocate(wms(nt,5))
	allocate(tms(nt,5))
	allocate(firm_size_vs_out(nt,2))

	!call writeToCSV("firm_worker_type.dat",real(sf%prod_line_worker_type_rtl(firm_index,:,:),wp))
	!call writeToCSV("firm_worker_wage.dat",real(sf%prod_line_worker_wage_rtl(firm_index,:,:),wp))
	!call writeToCSVVec("firm_type.dat",real(sf%firm_type_rt(firm_index,:),wp))

	!firm_size_vs_out(:,1) = sum( merge(1.0_wp ,0.0_wp, sf%prod_line_worker_wage_rtl(firm_index,:,:) .ne. 0),2)

	!firm_size_vs_out(:,2) = merge(0.0_wp  , real(sf%firm_output_rt(firm_index,:),wp) / firm_size_vs_out(:,1) , &
	!firm_size_vs_out(:,1) .eq. 0.0_wp)

	!call writeToCSV("firm_size_vs_output.dat",firm_size_vs_out)

	!compute mean wage and mean type
	do it=1,nt
!		wms(it,1)=it
!		tms(it,1)=it
		!call momentsVecWithMask(real(sf%prod_line_worker_wage_rtl(firm_index,tt,:),wp), &
		!						sf%prod_line_worker_wage_rtl(firm_index,tt,:) .ne. 0, &
		!						wms(tt,2), wms(tt,3))

		!call momentsVecWithMask(real(sf%prod_line_worker_type_rtl(firm_index,tt,:),wp), &
		!						sf%prod_line_worker_type_rtl(firm_index,tt,:) .ne. 0, &
		!						tms(tt,2), tms(tt,3))

		!tms(tt,4) = minval(merge(sf%prod_line_worker_type_rtl(firm_index,tt,:),200, &
		!			 sf%prod_line_worker_type_rtl(firm_index,tt,:) .ne. 0  ))
		!tms(tt,5) = maxval(sf%prod_line_worker_type_rtl(firm_index,tt,:))
	end do
	tms(:,3) = 0.5*sqrt(tms(:,3))
	wms(:,3) = 0.5*sqrt(wms(:,3))

	! call writeToCSV("firm_worker_type_moms.dat",tms)
	! call writeToCSV("firm_worker_wage_moms.dat",wms)
	! call writeToCSVvec("firm_size.dat",sum( merge(1.0_wp ,0.0_wp, sf%prod_line_worker_wage_rtl(firm_index,:,:) .ne. 0),2))

    deallocate(wms )
    deallocate(tms )
    deallocate(firm_size_vs_out )


  end subroutine printFirmToFiles

  subroutine SimulateFirmsHistory(sf, m, p )
	implicit none

    ! -------------- INPUTS --------------
    type (Model),               intent(in)    :: m
	type (ExogenousParameters), intent(in) :: p
    ! ------------- OUTPUTS --------------
    type (SimulationFirmPanel) , intent(INOUT) :: sf

    ! ------------- RANDS --------------
    real(wp),allocatable :: rand_x_r(:)
    real(wp),allocatable :: rand_delta_f(:), rand_zeta_l(:) ,rand_y_rt(:,:,:), rand_lambda_l(:)
	real(wp),allocatable :: rand_yshock_f(:) ,rand_yext_l(:), rand_xu_l(:) ,rand_theta_l(:) , rand_mu_l(:)
	integer,allocatable :: rand_x_hxy_l(:) ,rand_y_hxy_l(:)

    real(wp) ::lambda0,lambda1,lambda2

	integer :: nl, nx, ny, nw, nf, nt
	integer :: il, ix, iy, iw, if, it ,ill
	integer :: i,y,c,yprime ,w ,x, y_prev
	integer :: firm_id
	integer,allocatable :: highest_wage(:,:), lowest_wage(:,:)

    ! CDF to draw new values
    real(wp), allocatable :: Qcdf_yy(:,:), Vcdf_y(:), Lcdf_x(:) , Ccdf_c(:), Ucdf_x(:)

	nx = p%gridSizeWorkers
    ny = p%gridSizeFirms
    nw = p%gridSizeWages
    nf = p%FirmSimulation_nfirms
    nt = p%FirmSimulation_nperiods
	nl = p%FirmSimulation_nProdlines

	allocate(highest_wage(nx, ny))
	allocate(lowest_wage(nx, ny))
	call computeHighLowWageMatrices(lowest_wage , highest_wage , m)

    allocate( Qcdf_yy(ny ,ny))
    allocate( Vcdf_y(ny))
    allocate( Lcdf_x(nx))
    allocate( Ucdf_x(nx))

    allocate(rand_yshock_f(nf))
    allocate(rand_yext_l(nl))
    allocate(rand_x_hxy_l(nl))
    allocate(rand_y_hxy_l(nl))
	allocate(rand_delta_f (nf))
	allocate(rand_lambda_l(nl))
	allocate(rand_xu_l(nl))
	allocate(rand_zeta_l(nl))
	allocate(rand_theta_l(nl))
	allocate(rand_mu_l(nl))

    call pdf2cdf(m%v_y / m%v , Vcdf_y)
    call pdf2cdf(p%workersDistribution / p%nWorkers , Lcdf_x)
    call pdf2cdf(m%u_x / m%u , Ucdf_x)

    do i=1,ny
       call pdf2cdf(m%Q_yy(:,i) , Qcdf_yy(:,i) )
    end do

	! call writeToCSV("cumulative_y_shocks.dat" , Qcdf_yy)

    lambda0 = p%s0 * m%kappa * m%U
    lambda1 = p%s1 * m%kappa * m%V
    lambda2 = p%s1 * m%kappa * sum(m%h_xy)

	! write(*,*) "lambda0:" , lambda0
	! write(*,*) "lambda1:" , lambda1
	! write(*,*) "lambda2:" , lambda2

	call randvec(rand_delta_f)
    sf%prod_line_worker_type_lt = 0
    sf%prod_line_worker_wage_lt = 0
    sf%firm_type_ft             = 0
    sf%prod_line_firmid_lt      = 0

    !---------------------------------
    !         INITIAL VALUES
    !---------------------------------
    !give an empty line to each firm

    do if = 1,nf
        sf%firm_type_ft(if,1) = y
        sf%prod_line_firmid_lt(if,1)   = if
        sf%prod_line_worker_type_lt = 0
        sf%prod_line_worker_wage_lt = 0
    end do

    !---------------------------------
    !         FOR EACH PERIOD
    !---------------------------------
    do it =2,nt

        !---------------------------------
        !           GET DRAWS
        !---------------------------------
        call randVec(rand_yshock_f)
        call randVec(rand_yext_l)
        call randVec(rand_lambda_l)
        call randVec(rand_xu_l)
        call randVec(rand_zeta_l)
        call randVec(rand_mu_l)
        call randVec(rand_theta_l)
        call gibbsBivariateGenerate( m%h_xy , rand_x_hxy_l ,rand_y_hxy_l , 100)

        !---------------------------------
        !         FOR EACH FIRM
        !---------------------------------
        !GETTING THE NEW PRODUCTIVITY
        do if = 1,nf-1
            if (rand_delta_f(if) .le. ( p%delta ) ) then
               call locate(Qcdf_yy(:,sf%firm_type_ft(if,it-1)), rand_yshock_f(if),yprime) ! location for y' draw
               yprime = min(yprime+1, ny)
               sf%firm_type_ft(if,it) = yprime
            end if
        end do

	    !---------------------------------
	    !         FOR EACH LINE
	    !---------------------------------
        do il = 1,nl

            firm_id = sf%prod_line_firmid_lt(il,it-1)
            sf%prod_line_firmid_lt(il,it) = firm_id

			y_prev  = sf%firm_type_ft(firm_id,it-1)
			y       = sf%firm_type_ft(firm_id,it)

			! IF PRODUCTION LINE IS NOT ASSOCIATED WITH A FIRM, JUST PASS
            if (firm_id == 0) then
                continue
            end if

            ! CHECK IF LINE IS DESTROYED BY ANOTHER FIRM CAPTURING THE MARKET SHARE
            ! with probabilty mu, the line is destroyed
			if (rand_mu_l(il) < p%mu) then
				 sf%prod_line_firmid_lt(il,it) = 0
				 continue
			end if
			!reserve the line for next period
			sf%prod_line_firmid_lt(il,it+1) = firm_id

            ! CHECK IF LINE SPROUTS A NEW LINE
			if (rand_theta_l(il) < p%mu .and. sf%prod_line_worker_type_lt(il,it)>0 ) then
				!find an empty line and open it for the next period
				do ill=1,nl
					if ( sf%prod_line_firmid_lt(ill,it+1) == 0 ) then
						sf%prod_line_firmid_lt(ill,it+1) = firm_id
						sf%prod_line_worker_type_lt(ill,it+1)= 0
					end if
				end do
			end if

            w = sf%prod_line_worker_wage_lt(il,it-1)
            x = sf%prod_line_worker_type_lt(il,it-1)

            !---------------PRODUCTIVITY SHOCK FOR THE FIRM--------------------!
            if (y_prev .ne. y) then
				!---- NEED TO CORRECT CONTRACT ----!

				!---- NOT ACTIVE-----!
				if (w .eq. 0) then
					!NOTHING
				!---- DESTROYED-----!
				elseif (m%S_xy(x,y)<0.0_wp) then
					sf%prod_line_worker_type_lt(il,it) = 0
					sf%prod_line_worker_wage_lt(il,it) = 0

				!---- RENEG UP - WORKER GETS ENTIRE SURPLUS-----!
				elseif (m%W1_xyw(x,y,w) - m%W0_x(x) <0) then

					sf%prod_line_worker_type_lt(il,it) = x
					sf%prod_line_worker_wage_lt(il,it) = highest_wage(x,y)

				!---- RENEG DOWN - FIRMS GETS ENTIRE SURPLUS-----!
				elseif (m%W1_xyw(x,y,w) - m%W0_x(x) > m%S_xy(x,y)) then
					sf%prod_line_worker_type_lt(il,it) = x
					sf%prod_line_worker_wage_lt(il,it) = lowest_wage(x,y)

				!---- NOTHING-----!
				else
					sf%prod_line_worker_type_lt(il,it) = x
					sf%prod_line_worker_wage_lt(il,it) = w
				end if
			else
            !---------------NO PRODUCTIVITY SHOCK FOR THE FIRM--------------------!

				!### SKIP INACTIVE ONE ###
				if (w .eq. 0) then
				!### exogenous destruction ####
				elseif (rand_zeta_l(il) .le. p%zeta) then
					sf%prod_line_worker_type_lt(il,it) = 0
					sf%prod_line_worker_wage_lt(il,it) = 0

				!### worker gets outside offer ####
				elseif (rand_lambda_l(il) .le. lambda1) then
		      		call locate(Vcdf_y, rand_yext_l(il), yprime) ! find position for y' draw
		      		yprime = min(yprime+1, ny)

					!EXTERNAL FIRM BETTER - WORKER LEAVES
					if ( m%S_xy(x,yprime) > m%S_xy(x,y)) then
						sf%prod_line_worker_type_lt(il,it) = 0
						sf%prod_line_worker_wage_lt(il,it) = 0

					!EXTERNAL FIRM GOOD - WORKER GET RAISE
					else if (m%S_xy(x,yprime) > m%W1_xyw(x,y,w) - m%W0_x(x)) then
						sf%prod_line_worker_type_lt(il,it) = x
						sf%prod_line_worker_wage_lt(il,it) = find_wage(x,y,m%S_xy(x,yprime),m)

					!EXTERNAL FIRM BAD - NOTHING
					else
						sf%prod_line_worker_type_lt(il,it) = x
						sf%prod_line_worker_wage_lt(il,it) = w
					end if
				end if

				!###  LINE EMPTY WITH VACANCY ###
				if ( (w .eq. 0) .and. (m%PI0_y(y) > 0)) then

					!### new worker from other firm ####
					if (rand_lambda_l(il) .le. lambda2) then
						!DRAW SOMEONE FROM h_xy
						x = rand_x_hxy_l(il)
						yprime = rand_y_hxy_l(il)

						if (m%S_xy(x,y) > m%S_xy(x,yprime)) then
							sf%prod_line_worker_type_lt(il,it) = x
							sf%prod_line_worker_wage_lt(il,it)= find_wage(x,y,m%S_xy(x,yprime),m)
						else
							sf%prod_line_worker_type_lt(il,it) = 0
							sf%prod_line_worker_wage_lt(il,it) = 0
						end if

					!### new worker from unemployement ####
					else if (rand_lambda_l(il) .le. lambda0) then
						call locate(Ucdf_x, rand_xu_l(il), x)
						x = min(x+1, nx)

						if (m%S_xy(x,y)>0) then
							sf%prod_line_worker_type_lt(il,it) = x
							sf%prod_line_worker_wage_lt(il,it) = lowest_wage(x,y)
						else
							sf%prod_line_worker_type_lt(il,it) = 0
							sf%prod_line_worker_wage_lt(il,it) = 0
						end if

					end if
				end if ! FINISHED LOOKING AT DYNAMICS IN PRODUCTION LINES
			end if ! IF OF PRODUCTIVITY SHOCK

	    end do ! CLOSING LOOP ON FIRMS
    end do ! CLOSING LOOP ON TIME

    ! clean up
    deallocate(highest_wage )
    deallocate(lowest_wage )
    deallocate( Qcdf_yy )
    deallocate( Vcdf_y )
    deallocate( Lcdf_x )
    deallocate( Ucdf_x )
    deallocate(rand_yshock_f )
    deallocate(rand_yext_l )
    deallocate(rand_x_hxy_l )
    deallocate(rand_y_hxy_l )
    deallocate(rand_delta_f  )
    deallocate(rand_lambda_l )
    deallocate(rand_xu_l )
    deallocate(rand_zeta_l )
    deallocate(rand_theta_l )
    deallocate(rand_mu_l )

 	end subroutine SimulateFirmsHistory

	subroutine find_hilo_wage(x,y,low,hi,m)
	  	implicit none
	  	integer,intent(in) :: x,y
	  	integer,intent(out) :: hi,low
      	type (Model),intent(in)    :: m
	  	integer :: nw, w

	  	nw = size(m%W1_xyw,3)
		low = 1
		hi = nw

		do w=1,nw
			if ((m%W1_xyw(x,y,w) - m%W0_x(x) .ge. 0.0_wp) .and. (w < low)) then
				low = w
			end if

			if ((m%S_xy(x,y) - m%W1_xyw(x,y,w) + m%W0_x(x) .ge. 0.0_wp) .and. (w > hi)) then
				hi = w
			end if
		end do
	end subroutine find_hilo_wage

	subroutine computeHighLowWageMatrices(lowest_wage, highest_wage , m)
		implicit none
		integer,intent(inout) ::lowest_wage(:,:) , highest_wage(:,:)
		integer :: nx, ny , ix, iy
      	type (Model),intent(in)    :: m

		nx = size(lowest_wage,1)
		ny = size(lowest_wage,2)

		lowest_wage = 0
		highest_wage = 0

		do ix = 1,nx
			do iy= 1, ny
				if (m%S_xy(ix, iy)>0) then
					call find_hilo_wage(ix, iy, lowest_wage(ix,iy), highest_wage(ix,iy),m )
				end if
			end do
		end do
	end subroutine computeHighLowWageMatrices

	function find_wage(x,y,u,m)
		implicit none
		real(wp),intent(IN) :: u
		integer, intent(IN) :: x,y
      	type (Model),intent(in)    :: m
      	integer :: find_wage
      	integer :: nw, w

	  	nw = size(m%W1_xyw,3)
		find_wage = 1

		do w=nw,1
			if ((     m%W1_xyw(x,y,w) - m%W0_x(x) .ge. u) .and. (w < find_wage)) then
				find_wage = w
			end if
		end do
	end function find_wage

end module simulate_firms_mod
