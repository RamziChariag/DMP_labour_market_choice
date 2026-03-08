module compute_theoritical_moments_mod

  use glob
  use modeldef
  use params
  use stat_helper
  use array_helper
  use output
  use moments_mod

contains

subroutine computeTheoreticalMoments(m,p,moms)
    implicit none

    ! -------------- INPUTS --------------
    type (Model),                  intent(in)    :: m
    type (ExogenousParameters),    intent(in)    :: p
    ! ------------- OUTPUTS --------------
    type (Moments) , intent(INOUT) :: moms

    ! -------------- LOCALS --------------
    integer :: nr, nt        ! R individuals, T time periods
    integer :: nx, ny, nw  ! dims of surplus funciton
    real(wp) ::lambda0,lambda1
	real(wp) :: res

	call computeJob2JobTransitionRate(m,p,res)
	!write(*,*) "theoretical J2J: " , res

	call computeE2UTransitionRate(m,p,res)
	!write(*,*) "theoretical E2U: " , res

	call computeU2ETransitionRate(m,p,res)
	!write(*,*) "theoretical U2E: " , res


    end subroutine computeTheoreticalMoments

subroutine computeJob2JobTransitionRate(m,p,res)

    ! -------------- INPUTS --------------
    type (Model),                  intent(in)    :: m
    type (ExogenousParameters),    intent(in)    :: p

    ! ------------- OUTPUTS --------------
    real(wp) , intent(OUT) :: res

    ! -------------- LOCALS --------------
	integer :: nx, ny ! bounds
	integer :: ix, iy , iyp ! indices
	real(wp) :: integral

	nx = p%gridSizeWorkers
	ny = p%gridSizeFirms

	integral = 0.0_wp
	do ix = 1,nx
		do iy = 1,ny
			do iyp =1,ny
				if (m%S_xy(ix,iyp) .ge. m%S_xy(ix,iy)) then
					integral = integral + m%h_xy(ix,iy) * m%v_y(iyp)
				end if
			end do
		end do
	end do

	res = p%s1 * m%kappa * integral / (sum(m%h_xy))

end subroutine computeJob2JobTransitionRate

subroutine computeE2UTransitionRate(m,p,res)

    ! -------------- INPUTS --------------
    type (Model),                  intent(in)    :: m
    type (ExogenousParameters),    intent(in)    :: p

    ! ------------- OUTPUTS --------------
    real(wp) , intent(OUT) :: res

    ! -------------- LOCALS --------------
	integer :: nx, ny ! bounds
	integer :: ix, iy , iyp ! indices
	real(wp) :: integral

	nx = p%gridSizeWorkers
	ny = p%gridSizeFirms

	integral = 0.0_wp
	do ix = 1,nx
		do iy = 1,ny
			do iyp =1,ny
				!The surplus must be negative in the new y p level
				if (m%S_xy(ix,iyp) .le. 0.0_wp) then
					integral = integral + m%h_xy(ix,iy) * m%q_yy(iy,iyp)
				end if
			end do
		end do
	end do

	res = p%delta * integral / (sum(m%h_xy))

end subroutine computeE2UTransitionRate


subroutine computeU2ETransitionRate(m,p,res)

    ! -------------- INPUTS --------------
    type (Model),                  intent(in)    :: m
    type (ExogenousParameters),    intent(in)    :: p

    ! ------------- OUTPUTS --------------
    real(wp) , intent(OUT) :: res

    ! -------------- LOCALS --------------
	integer :: nx, ny ! bounds
	integer :: ix, iy , iyp ! indices
	real(wp) :: integral

	nx = p%gridSizeWorkers
	ny = p%gridSizeFirms

	integral = 0.0_wp
	do ix = 1,nx
		do iy = 1,ny
			if (m%S_xy(ix,iy) .ge. 0.0_wp) then
				integral = integral + m%u_x(ix) * m%v_y(iy)
			end if
		end do
	end do

	res = p%s0 * m%kappa * integral / (sum(m%u_x))

end subroutine computeU2ETransitionRate

end module compute_theoritical_moments_mod

