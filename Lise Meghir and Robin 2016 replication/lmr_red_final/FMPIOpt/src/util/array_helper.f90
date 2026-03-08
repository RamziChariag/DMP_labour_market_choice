module array_helper
  use fmpioptglob
  implicit none
contains
  subroutine linspace(z,start,end)
    implicit none
    real(wpfmpi), intent(inout) :: z(:)
    real(wpfmpi), intent(in) :: start, end
    real(wpfmpi) step;
    integer n,i

    n = SIZE(z);
    step = (end-start)/ (n-1)
    z(1) = real(start,wpfmpi)
    do i = 2,n
       z(i) = z(i-1) + step
    end do
  end subroutine linspace

  subroutine kron_on_vects(z,x,y)
    implicit none
    real(wpfmpi), intent(inout) :: z(:,:)
    real(wpfmpi), intent(in) :: x(:)
    real(wpfmpi), intent(in) :: y(:)
  end subroutine kron_on_vects

  subroutine kron_on_matrices(z,x,y)
    implicit none
    real(wpfmpi), intent(inout) :: z(:,:)
    real(wpfmpi), intent(in) :: x(:,:)
    real(wpfmpi), intent(in) :: y(:)
  end subroutine kron_on_matrices

  subroutine printArray(A)
    implicit none
    real(wpfmpi), intent(in) :: A(:,:)
    integer :: n,m
    CHARACTER(len=200) :: myFormat
    character(len=10) :: ci
    n = size(A,2)
    write(ci,*) n
    myFormat = '( a / ( ' // ci // '(f5.4,1x)) )'
    print myFormat , "matching set",  A

  end subroutine printArray

  !Given an array xx(1:n), and given a value x, returns a value j
  !such that x is between xx(j) and xx(j+1).  xx(1:n) must be
  !monotonic, either increasing or decreasing.  j=0 or j=n is
  !returned to indicate that x is out of range.
  subroutine locate(xx,x,j)
    implicit none
    integer,intent(out) :: j
    real(wpfmpi),intent(in) :: x, xx(:)
    integer jl, jm, ju ,n

    n=size(xx)
    jl=0  !initialize lower limit
    ju=n+1  !initialize upper limit

    !we start with highest and lowest value
    !and pick the value in the middle to
    !separate the interval in 2.
    do while (ju-jl .gt. 1)
       jm=(ju+jl)/2
       if((xx(n).ge.xx(1)) .eqv. (x.ge.xx(jm))) then
          jl=jm
       else
          ju=jm
       endif
    end do

    if (x.eq.xx(1)) then
       j=1
    else if (x.eq.xx(n)) then
       j=n-1
    else
       j=jl
    endif

  end subroutine locate

  function meanVec(X)
    implicit none
    real(wpfmpi),intent(in) :: X (:)
    real(wpfmpi) :: meanVec
    integer n

    n = size(X)
    meanVec = sum(X) / real(n,wpfmpi)

  end function meanVec

  subroutine momentsVecWithMask(X,mask,lmean, lvar)
    implicit none
    real(wpfmpi),intent(in) :: X (:)
    real(wpfmpi),intent(out) :: lmean , lvar
    logical, intent(in) :: mask(:)
    real(wpfmpi) :: meanVec
    integer n,i,lcount

    n = size(X)
    lcount=0
    lmean = 0.0_wpfmpi
    lvar = 0.0_wpfmpi

    do i=1,n
        if (mask(i)) then
            lmean = lmean + X(i)
            lvar  = lvar + X(i) * X(i)
            lcount = lcount + 1
         end if
    end do

    lmean = lmean / real(lcount,wpfmpi)
    lvar = lvar / real(lcount,wpfmpi) - lmean * lmean

    if (lcount .eq. 0) then
    	lmean = 0
    	lvar = 0
    end if

  end subroutine momentsVecWithMask

  function covWithMask(X,Y,mask)
    implicit none
    real(wpfmpi),intent(in) :: X (:), Y(:)
    real(wpfmpi) :: mean_x , mean_y , prod_xy
    logical, intent(in) :: mask(:)
    real(wpfmpi) :: covWithMask
    integer n,i,lcount

    n = size(X)
    lcount=0
    mean_x = 0
    mean_y = 0
    prod_xy = 0

    do i=1,n
        if (mask(i)) then
            mean_x = mean_x + X(i)
            mean_y = mean_y + Y(i)
            prod_xy  = prod_xy + X(i) * Y(i)
            lcount = lcount + 1
         end if
    end do

    mean_x = mean_x / real(lcount,wpfmpi)
    mean_y = mean_y / real(lcount,wpfmpi)

    covWithMask = prod_xy / real(lcount,wpfmpi) - mean_x * mean_y

  end function covWithMask


!  function floatMaskMat(X)
!    implicit none
!    logical, intent(in) :: X(:,:)
!    real(wp) :: floatMaskMat(:,:)
!
!    floatMaskMat = merge(1.0_wp , 0.0_wp , X)
!
!  end function floatMaskMat

  subroutine computeDiscreteHistogram(X,hist)
  	implicit none
  	integer,intent(in) :: X(:)
  	integer,intent(inout) :: hist(:)
  	integer n,k

  	n = size(X)
  	hist = 0
  	do k = 1,n
  		hist(X(k))=hist(X(k)) + 1
  	end do

  end subroutine computeDiscreteHistogram

  subroutine computeRank(X,R)
    implicit none
    real(wpfmpi), intent(IN) :: X(:)
	integer, intent(INOUT) :: R(:)
	integer :: n,i,j

	R=1
	n=size(X)
	do i=1,n
	  do j=1,n
		if (X(i) .gt. X(j)) then
			R(i) = R(i) + 1
		end if
		if ((X(i) .eq. X(j)) .and. (i>j)) then
			R(i) = R(i) + 1
		end if
	  end do
	end do

  end subroutine computeRank

	subroutine compute2DHist(X,Y,hist)
	    implicit none
		integer,intent(IN) :: X(:), Y(:)
		real(wpfmpi), intent(INOUT) :: hist(:,:)

		integer :: n,lcount
		integer :: i

		n = size(X)
		hist = 0.0_wpfmpi
		lcount = 0

		do i = 1,n
			hist(X(i),Y(i)) = hist(X(i),Y(i)) + 1.0_wpfmpi
					lcount = lcount +1
		end do

		hist = hist / real(lcount,wpfmpi)

	end subroutine compute2DHist



end module array_helper
