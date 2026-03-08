module copula
  use stat_helper
  use glob
  implicit none

contains
  function tcopulapdf(x,y,rho,nu)
    implicit none
    real(wp), intent(IN) :: x, y , rho
    real(wp), intent(IN) :: nu
    real(wp) :: xi , yi ,xj,yj
    real(wp) :: pi
    real(wp) :: tcopulapdf , tcopulapdf1 , tcopulapdf2
    pi = 2*ACOS(0.0_wp)
    
    !compute the inverse t distribution
    xi = tinv(x,nu);
    yi = tinv(y,nu);
    
    !compute the pdf of the copula
    tcopulapdf = 1.0_wp/ ( (2.0_wp*pi) * (sqrt(1.0_wp-rho**2)) ) &
         * ( 1.0_wp + (xi**2 -2.0_wp*yi*xi*rho+yi**2)/(nu*(1.0_wp-rho**2))  )**(-(real(nu,wp)+2)/2) &
         / (tpdf(xi,nu) * tpdf(yi,nu)) ;
    
  end function tcopulapdf

  subroutine generateTCopulaMatrix(y_y,rho,nu,Q)
    implicit none
    real(wp), intent(IN), dimension(:) :: y_y
    real(wp), intent(IN) :: rho
    real(wp), intent(IN) :: nu
    real(wp), intent(INOUT) :: Q(:,:)
    integer :: x,y
    
    do x = 1, size(y_y)
       do y = 1, size(y_y)
          Q(x,y) = tcopulapdf( y_y(x) , y_y(y) ,rho,nu)
       end do
    end do
                
  end subroutine generateTCopulaMatrix

 
end module copula
