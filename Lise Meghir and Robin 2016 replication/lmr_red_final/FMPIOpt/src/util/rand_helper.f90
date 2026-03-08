module rand_helper
  use fmpioptglob
  use array_helper
  implicit none

contains
  subroutine initSeed(seed)
    implicit none
    integer, intent(IN) :: seed

    call ZBQLINI(seed)
  end subroutine

  subroutine pdf2cdf(pdf , cdf )
    real(wpfmpi),intent(IN) :: pdf(:)
    real(wpfmpi),intent(INOUT) :: cdf(:)
    integer n,i

    n = size(pdf)
    cdf(1) = pdf(1)

    do i=2,n
       cdf(i) = cdf(i-1) + pdf(i)
    end do

  end subroutine pdf2cdf

  ! fills in a vector with draws from
  ! a normal distribution
  subroutine randnVec(res)
    implicit none
    real(wpfmpi), intent(INOUT) :: res(:)
    real(kind=8) ZBQLNOR
    integer n,i

    n = size(res)

    do i=1,n
       res(i) = ZBQLNOR(0.0_wpfmpi,1.0_wpfmpi)
    end do

  end subroutine randnVec


  ! fills in a vector with draws from
  ! a uniform [0,1] distribution
  subroutine randVec(res)
    implicit none
    real(wpfmpi), intent(INOUT) :: res(:)
    real(kind=8) ZBQLU01
    integer n,i

    n = size(res)

    do i=1,n
       res(i) = ZBQLU01(0.0_wpfmpi)
    end do

  end subroutine randVec

  subroutine randMat(res)
    implicit none
    real(wpfmpi), intent(INOUT) :: res(:,:)
    real(kind=8) ZBQLU01
    integer n,m,i,j

    n = size(res,1)
    m = size(res,2)

    do i=1,n
        do j=1,m
            res(i,j) = ZBQLU01(0.0_wpfmpi)
        end do
    end do

  end subroutine randMat

  subroutine gibbsBivariateGenerate(pdf, X,Y , burnout)
  	implicit none
  	real(wpfmpi), intent(IN) :: pdf(:,:)
  	integer, intent(IN) :: burnout
  	integer,intent(INOUT) :: x(:), y(:)
	integer nx, ny ,n ,i,j
	real(wpfmpi),allocatable :: y_conditional_xy(: ,: ), x_conditional_xy(: ,: )
	real(wpfmpi),allocatable :: rand_x_r(:) ,  rand_y_r(:)
	integer :: xprime, yprime , xprev, yprev

	nx = size(pdf, 1)
	ny = size(pdf, 2)
	n = size(X)

	allocate(y_conditional_xy(nx,ny))
	allocate(x_conditional_xy(nx,ny))
	allocate(rand_x_r(n))
	allocate(rand_y_r(n))

	call randVec( rand_x_r)
	call randVec( rand_y_r)

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

	do i=1,n
		call locate(x_conditional_xy(:,yprev), rand_x_r(i) ,xprime)
		xprime = min(xprime+1 , nx)
		X(i) = xprime
		xprev = xprime
		call locate(y_conditional_xy(xprev,:),rand_y_r(i),yprime)
		yprime = min(yprime+1,ny)
		Y(i) = yprime
		yprev = yprime
	end do
 end subroutine gibbsBivariateGenerate

  subroutine gibbsBivariateGenerateMat(pdf, X,Y , burnout)
  	implicit none
  	real(wpfmpi), intent(IN) :: pdf(:,:)
  	integer, intent(IN) :: burnout
  	integer,intent(INOUT) :: x(:,:), y(:,:)
	integer nx, ny ,ni,nj ,i,j
	real(wpfmpi),allocatable :: y_conditional_xy(: ,: ), x_conditional_xy(: ,: )
	real(wpfmpi),allocatable :: rand_x_r(:,:) ,  rand_y_r(:,:)
	integer :: xprime, yprime , xprev, yprev

	nx = size(pdf, 1)
	ny = size(pdf, 2)
	ni = size(X,1)
	nj = size(X,2)

	allocate(y_conditional_xy(nx,ny))
	allocate(x_conditional_xy(nx,ny))
	allocate(rand_x_r(ni,nj))
	allocate(rand_y_r(ni,nj))

	call randMat( rand_x_r)
	call randMat( rand_y_r)

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
 end subroutine gibbsBivariateGenerateMat

 end module rand_helper
