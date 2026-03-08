module stat_helper
  use glob
  use biomath_constants_mod
  use cdf_t_mod
  use cdf_normal_mod
  use cdf_beta_mod
  use array_helper
  implicit none

contains

  function norminv(x)
    implicit none
    real(wp), intent(IN) :: x
    real(wp) :: norminv
    real(dpkind) :: tmp
    norminv = 1.1_wp

    !norminv = real( fgsl_cdf_ugaussian_Pinv( real( x , fgsl_double ) ), sp)
    !norminv = nag_normal_deviate('L',real(x,8))
    norminv = real(inv_normal(cum = real(x , dpkind)) , wp)
  end function norminv

  function normpdf(x)
    implicit none
    real(wp), intent(IN) :: x
    real(wp) :: normpdf
    real(dpkind) :: tmp
    normpdf = 0.5_wp

    !norminv = real( fgsl_cdf_ugaussian_Pinv( real( x , fgsl_double ) ), sp)
    !norminv = nag_normal_deviate('L',real(x,8))
     call normal_01_pdf(real(x , dpkind) , tmp)
     normpdf = real(tmp,wp)
  end function normpdf

  function tpdf(x,nu)
    implicit none
    real(wp), intent(IN) :: x
    real(wp), intent(IN) :: nu
    real(wp) :: tpdf
    real(wp) :: pi
    real(wp) :: tcopulapdf , tcopulapdf1 , tcopulapdf2
    pi = 2*ACOS(0.0_wp)
    tpdf = 1.1_wp

    !tpdf = real( fgsl_ran_tdist_pdf( real( x , fgsl_double ) , real( nu , fgsl_double ) ), sp)
    tpdf = gamma( (nu + 1.0_wp )/2.0_wp ) / (sqrt( nu * pi) * gamma( nu / 2.0_wp) ) *(1.0_wp +x**2 / nu ) ** (- (nu+1.0_wp)/2.0_wp)
  end function tpdf

  function tinv(x,nu)
    implicit none
    real(wp), intent(IN) :: x
    real(wp), intent(IN) :: nu
    real(wp) :: tinv
    real(dpkind) :: tmp
    tinv = 1.1_wp

    !tinv = real( fgsl_cdf_tdist_Pinv( real( x , fgsl_double ) , real( nu , fgsl_double ) ), sp)
    !tinv = nag_t_deviate('L',real(x,8),real(nu,8))
    tmp = inv_t(cum = real(x, dpkind) , df = real(nu , dpkind))
    tinv = real(tmp, wp)
  end function tinv

  function betainv(x,aa,bb)
    implicit none
    real(wp), intent(IN) :: x, aa, bb
    real(wp) :: betainv
    real(dpkind) :: tmp
    betainv = 1.1_wp
    tmp = inv_beta(cum = real(x, dpkind), ccum = real(1.0_wp - x, dpkind), a = real(aa, dpkind), b = real(bb, dpkind) )
    betainv = real(tmp, wp)
  end function betainv

  function gamma(x)
    implicit none
    real(wp),intent(IN) :: x
    real(wp) :: gamma
    real(kind=8) r8_gamma
    gamma = 1.1_wp

    !gamma = nag_gamma(real(x,8))
    gamma = r8_gamma(real(x,8))
  end function gamma

  subroutine pdf2cdf(pdf , cdf )
    real(wp),intent(IN) :: pdf(:)
    real(wp),intent(INOUT) :: cdf(:)
    integer n,i

    n = size(pdf)
    cdf(1) = pdf(1)

    do i=2,n
       cdf(i) = cdf(i-1) + pdf(i)
    end do

  end subroutine pdf2cdf

  subroutine initSeed_lmr(seed)
    implicit none
    integer, intent(IN) :: seed

    call ZBQLINI(seed)
  end subroutine initSeed_lmr

  ! fills in a vector with draws from
  ! a normal distribution
  subroutine randnVec(res)
    implicit none
    real(wp), intent(INOUT) :: res(:)
    real(kind=8) ZBQLNOR
    integer n,i

    n = size(res)

    do i=1,n
       res(i) = ZBQLNOR(0.0_wp,1.0_wp)
    end do

  end subroutine randnVec


  ! fills in a vector with draws from
  ! a uniform [0,1] distribution
  subroutine randVec_lmr(res)
    implicit none
    real(wp), intent(INOUT) :: res(:)
    real(kind=8) ZBQLU01
    integer n,i

    n = size(res)

    do i=1,n
       res(i) = ZBQLU01(0.0_wp)
    end do

  end subroutine randVec_lmr

  subroutine randMat_lmr(res)
    implicit none
    real(wp), intent(INOUT) :: res(:,:)
    real(kind=8) ZBQLU01
    integer n,m,i,j

    n = size(res,1)
    m = size(res,2)

    do i=1,n
        do j=1,m
            res(i,j) = ZBQLU01(0.0_wp)
        end do
    end do

  end subroutine randMat_lmr

  subroutine gibbsBivariateGenerate(pdf, X,Y , burnout)
  	implicit none
  	real(wp), intent(IN) :: pdf(:,:)
  	integer, intent(IN) :: burnout
  	integer,intent(INOUT) :: x(:), y(:)
	integer nx, ny ,nr ,i
	real(wp),allocatable :: y_conditional_xy(: ,: ), x_conditional_xy(: ,: )
	real(wp),allocatable :: rand_x_r(:) ,  rand_y_r(:)
	integer :: xprime, yprime

	nx = size(pdf, 1)
	ny = size(pdf, 2)
	nr = size(X,1)

	allocate(y_conditional_xy(nx,ny))
	allocate(x_conditional_xy(nx,ny))
	allocate(rand_x_r(nr))
	allocate(rand_y_r(nr))

	call randVec_lmr( rand_x_r)
	call randVec_lmr( rand_y_r)

    ! compute conditionals
    do i=1,ny
       call pdf2cdf(pdf(:,i )/sum(pdf(:,i )), x_conditional_xy(:,i))
    end do

    do i=1,nx
       call pdf2cdf(pdf(i,:)/sum(pdf(i,:)) , y_conditional_xy(i,:))
    end do

	! get uniform draws

	Y(1) = 1
	X(1) = 1

	do i=2,nr
		call locate(x_conditional_xy(:,Y(i-1)), rand_x_r(i) ,xprime)
		X(i) = xprime
		call locate(y_conditional_xy(X(i),:),rand_y_r(i),yprime)
		Y(i) = yprime
	end do

	! clean up
	deallocate(y_conditional_xy )
    deallocate(x_conditional_xy )
    deallocate(rand_x_r )
    deallocate(rand_y_r )

 end subroutine gibbsBivariateGenerate

  subroutine gibbsBivariateGenerateMat(pdf, X,Y , burnout)
  	implicit none
  	real(wp), intent(IN) :: pdf(:,:)
  	integer, intent(IN) :: burnout
  	integer,intent(INOUT) :: x(:,:), y(:,:)
	integer nx, ny ,ni,nj ,i,j
	real(wp),allocatable :: y_conditional_xy(: ,: ), x_conditional_xy(: ,: )
	real(wp),allocatable :: rand_x_r(:,:) ,  rand_y_r(:,:)
	integer :: xprime, yprime , xprev, yprev

	nx = size(pdf, 1)
	ny = size(pdf, 2)
	ni = size(X,1)
	nj = size(X,2)

	allocate(y_conditional_xy(nx,ny))
	allocate(x_conditional_xy(nx,ny))
	allocate(rand_x_r(ni,nj))
	allocate(rand_y_r(ni,nj))

	call randMat_lmr( rand_x_r)
	call randMat_lmr( rand_y_r)

    ! compute conditionals
    do i=1,ny
       call pdf2cdf(pdf(:,i )/sum(pdf(:,i )), x_conditional_xy(:,i))
    end do

    do i=1,nx
       call pdf2cdf(pdf(i,:)/sum(pdf(i,:)) , y_conditional_xy(i,:))
    end do

	! get uniform draws

	xprev = 1
	yprev = 1

	do i=1,ni
	  do j=1,nj
		call locate(x_conditional_xy(:,yprev), rand_x_r(i,j) ,xprime)
		xprime = min(xprime+1 , nx)
		X(i,j) = xprime
		xprev = xprime
		call locate(y_conditional_xy(xprev,:),rand_y_r(i,j),yprime)
		yprime = min(yprime+1,ny)
		Y(i,j) = yprime
		yprev = yprime
	  end do
	end do

	! clean up
    deallocate(y_conditional_xy )
    deallocate(x_conditional_xy )
    deallocate(rand_x_r )
    deallocate(rand_y_r )

 end subroutine gibbsBivariateGenerateMat

end module stat_helper
