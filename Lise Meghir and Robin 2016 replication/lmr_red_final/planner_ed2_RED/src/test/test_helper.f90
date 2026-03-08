module test_helper
  use glob

contains

  subroutine generateArray(seed, res_nm)
    implicit none
    real(wp), intent(in) :: seed
    real(wp), intent(inout) :: res_nm(:,:)
    integer :: n,m,x,y

    n = size(res_nm,1)
    m = size(res_nm,2)
    
    do x = 1,n
       do y= 1,m
          res_nm(x,y) = cos(10.0_wp * real(x * y,wp) + seed)
       end do
    end do
  end subroutine generateArray

  subroutine generateVec(seed, res_nm)
    implicit none
    real(wp), intent(in) :: seed
    real(wp), intent(inout) :: res_nm(:)
    integer :: n,x

    n = size(res_nm)
    
    do x = 1,n
       res_nm(x) = cos(10.0_wp * real(x ,wp) + seed)
    end do
  end subroutine generateVec

  subroutine generateDistibutionVec(seed, res_n)
    implicit none
    real(wp), intent(in) :: seed
    real(wp), intent(inout) :: res_n(:)
    integer :: n,x

    n = size(res_n)
    
    do x = 1,n
       res_n(x) = abs(cos(10.0_wp * real(x ,wp) + seed))
    end do
    
    res_n = res_n / sum(res_n)
  end subroutine generateDistibutionVec

  subroutine generateDistributionArray(seed, res_nm)
    implicit none
    real(wp), intent(in) :: seed
    real(wp), intent(inout) :: res_nm(:,:)
    integer :: n,m,x,y

    n = size(res_nm,1)
    m = size(res_nm,2)
    
    do x = 1,n
       do y= 1,m
          res_nm(x,y) = cos(10.0_wp * real(x * y,wp) + seed)
       end do
    end do

    res_nm = abs(res_nm)
    res_nm = res_nm / sum(res_nm)

  end subroutine generateDistributionArray

end module test_helper
