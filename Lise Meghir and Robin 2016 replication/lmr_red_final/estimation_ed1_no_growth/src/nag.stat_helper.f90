module stat_helper
  use glob
  use nag_normal_dist
  use nag_gamma_fun
  use nag_t_dist
  implicit none

contains

  function norminv(x)
    implicit none
    real(wp), intent(IN) :: x
    real(wp) :: norminv
    
    !norminv = real( fgsl_cdf_ugaussian_Pinv( real( x , fgsl_double ) ), sp)
    norminv = nag_normal_deviate('L',real(x,8))
  end function norminv

  function tpdf(x,nu)
    implicit none
    real(wp), intent(IN) :: x
    real(wp), intent(IN) :: nu
    real(wp) :: tpdf
    real(wp) :: pi
    real(wp) :: tcopulapdf , tcopulapdf1 , tcopulapdf2
    pi = 2*ACOS(0.0_wp)
    
    !tpdf = real( fgsl_ran_tdist_pdf( real( x , fgsl_double ) , real( nu , fgsl_double ) ), sp)
    tpdf = gamma( (nu + 1.0_wp )/2.0_wp ) / (sqrt( nu * pi) * gamma( nu / 2.0_wp) ) *(1.0_wp +x**2 / nu ) ** (- (nu+1.0_wp)/2.0_wp)
  end function tpdf
  
  function tinv(x,nu)
    implicit none
    real(wp), intent(IN) :: x
    real(wp), intent(IN) :: nu
    real(wp) :: tinv
    
    !tinv = real( fgsl_cdf_tdist_Pinv( real( x , fgsl_double ) , real( nu , fgsl_double ) ), sp)
    tinv = nag_t_deviate('L',real(x,8),real(nu,8))
  end function tinv
  
  function gamma(x)
    implicit none
    real(wp),intent(IN) :: x
    real(wp) :: gamma
    
    gamma = nag_gamma(real(x,8))
  end function gamma

end module stat_helper
