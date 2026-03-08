module moments_mod

  use glob
  use modeldef
  use params
  use stat_helper
  use array_helper
  use output

    type Moments

        real(wp) :: E(20)                 ! Employment rate 
        real(wp) :: E2U(20)               ! Emploment to Unemployment transition rate
        real(wp) :: U2E(20)               ! Unemployment to Employment transition rate
        real(wp) :: J2J(20)               ! Job to Job transiton rate

        real(wp) :: w(20)                 ! mean log wage 
        real(wp) :: Dw(20)                ! mean wage growth
        real(wp) :: Dw_EE(20)             ! mean wage change on the job
        real(wp) :: Dw_DJ(20)             ! mean wage change at job change

        real(wp) :: w2(20)                ! variance log wage
        real(wp) :: Dw2(20)               ! var wage growth
        real(wp) :: Dw2_EE(20)            ! var  wage change on the job
        real(wp) :: Dw2_DJ(20)            ! var wage change at job change

        real(wp) :: V_U                   ! vacancy to unemployment ratio

  end type Moments

contains

 subroutine initMoments(moms)
    implicit none
    type (Moments) , intent(INOUT) :: moms

 end subroutine initMoments

 subroutine freeMoments(moms)
    implicit none
    type (Moments) , intent(INOUT) :: moms

 end subroutine freeMoments


end module moments_mod
