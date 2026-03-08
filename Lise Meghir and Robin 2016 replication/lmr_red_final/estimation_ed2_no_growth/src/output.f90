module output
  use glob

contains

  subroutine readFromCSV(filename , mat)
  	real(wp) , intent(INOUT) :: mat(:,:)
  	character (*), intent(IN) :: filename

	integer n,m;
    n = size(mat,1)
    m = size(mat,2)

	open(13,file=filename,form='formatted')

	do i=1,n
 		read(13,*) (mat(i,j),j=1,m) !--   <- implicit in-line do loop
	end do
    close(13)

  end subroutine readFromCSV

  subroutine readFromCSVvec(filename , vec)
    real(wp) , intent(INOUT) :: vec(:)
    character (*), intent(IN) :: filename

    integer n;
    n = size(vec)

    open(13,file=filename,form='formatted')

    do i=1,n
        read(13,*) (vec(i)) !--   <- implicit in-line do loop
    end do
    close(13)

  end subroutine readFromCSVvec

  subroutine writeToCSV(filename , mat)
    implicit none
    real(wp), intent(IN) :: mat(:,:)
    real(wp),allocatable :: mat_tr(:,:)
    CHARACTER (*), intent(IN) :: filename
    integer :: n,m,x,y
    character(len=40)     :: fstr

    n = size(mat,1)
    m = size(mat,2)

    allocate( mat_tr(m,n))

    mat_tr = transpose(mat)

    open(unit=11,file=filename)

    write(fstr,'( "(",i4,"(ES18.8E3,1x))" )') m
    write(11,fstr) mat_tr

    close(11)
    deallocate(mat_tr)

  end subroutine writeToCSV

  subroutine writeToCSVVec(filename , vec)
    implicit none
    real(wp), intent(IN) :: vec(:)
    CHARACTER (*), intent(IN) :: filename
    integer :: n,m,x,y
    character(len=40)     :: fstr

    m = size(vec)

    open(unit=11,file=filename)
    write(fstr,'( "(",i4,"(ES18.8E3,1x))" )') m

    do x=1,m
    	write(11,"(ES18.8E3)") vec(x)
    end do

    close(11)
  end subroutine writeToCSVVec

  subroutine printMat(mat)
    implicit none
    real(wp), intent(IN) :: mat(:,:)
    real(wp),allocatable :: mat_tr(:,:)
    integer :: n,m,x,y
    character(len=40)     :: fstr

    n = size(mat,1)
    m = size(mat,2)

    allocate( mat_tr(m,n))
    mat_tr = transpose(mat)

    write(fstr,'( "(",i4,"(ES18.8E2,1x))" )') m
    write(*,fstr) mat_tr

    deallocate(mat_tr)

  end subroutine printMat

  subroutine printVec(vec)
    implicit none
    real(wp), intent(IN) :: vec(:)
    integer :: n
    character(len=40)     :: fstr
    n = size(vec)

    write(fstr,'( "(",i4,"(ES18.8E2,1x))" )') n
    write(*,fstr) vec

  end subroutine printVec

  subroutine log4mat_debug_real(sig , level, str, val)
    implicit none
    integer, intent(IN) :: level
    real(wp), intent(IN) :: val
    CHARACTER (*), intent(IN) :: str
	CHARACTER (*), intent(IN) :: sig

	write(*,*) "[" , sig , "] ", str , " " , val

  end subroutine log4mat_debug_real
end module output
