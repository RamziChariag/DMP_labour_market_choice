module Compute_Surplus_Helper_Mod
  use glob
  implicit none

contains
  subroutine VM1M2(res_xyw, S_xy , W1m0_xyw, v_y)
    implicit none
    real(wp), intent(IN) :: S_xy(:,:), v_y(:), W1m0_xyw(:,:,:)
    real(wp), intent(INOUT) ::  res_xyw(:,:,:)
    integer nx, ny, nw
    integer h, i, j, k

    nx = size(res_xyw , 1)
    ny = size(res_xyw , 2)
    nw = size(res_xyw , 3)

    ! Note that the following condition is {M1(x,y) union M2(w,x,y)}
    ! since if S(x,y') > S(x,y) then by definition S(x,y') >
    ! W1(w,x,y)-W0(x) > 0
    !

    do h=1,nw
       do i=1,nx
          do j=1,ny
             res_xyw(i,j,h) = 0.0_wp
             if (S_xy(i,j) .ge. 0.0) then
                do k=1,ny
                   if ( (W1m0_xyw(i,j,h)) .lt. S_xy(i,k) .and. S_xy(i,k) .ge. 0.0 ) then
                      res_xyw(i,j,h) = res_xyw(i,j,h) +  v_y(k)
                   end if
                end do
             end if
          end do
       end do
    end do
  end subroutine VM1M2

!!$  subroutine IntCdQ(res_xyw, S_xy, W1m0_xyw, Q_yy)
!!$    implicit none
!!$    real(wp), intent(IN) :: S_xy(:,:), W1m0_xyw(:,:,:), Q_yy(:,:)
!!$    real(wp), intent(INOUT) ::  res_xyw(:,:,:)
!!$    integer nx, ny, nw
!!$    integer h, i, j, k
!!$
!!$    nx = size(res_xyw , 1)
!!$    ny = size(res_xyw , 2)
!!$    nw = size(res_xyw , 3)
!!$
!!$    do h=1,nw
!!$       do i=1,nx
!!$          do j=1,ny
!!$             res_xyw(i,j,h) = 0.0_wp
!!$             if (S_xy(i,j) .gt. 0.0) then
!!$                do k=1,ny
!!$
!!$                   if ((S_xy(i,k) .ge. W1m0_xyw(i,k,h) .and. (W1m0_xyw(i,k,h) .ge. 0.0)) then
!!$                      res_xyw(i,j,h) = res_xyw(i,j,h) +  W1m0_xyw(i,k,h) * Q_yy(k,j)
!!$                   end if
!!$
!!$                end do
!!$             end if
!!$          end do
!!$       end do
!!$    end do
!!$    return
!!$  end subroutine IntCdQ
!!$  
!!$  subroutine IntBarCdQ(res_xyw, S_xy, W1m0_xyw, Q_yy)
!!$    implicit none
!!$    real(wp), intent(IN) :: S_xy(:,:), W1m0_xyw(:,:,:), Q_yy(:,:)
!!$    real(wp), intent(INOUT) ::  res_xyw(:,:,:)
!!$    integer nx, ny, nw
!!$    integer h, i, j, k
!!$
!!$    nx = size(res_xyw , 1)
!!$    ny = size(res_xyw , 2)
!!$    nw = size(res_xyw , 3)
!!$
!!$    do h=1,nw
!!$       do i=1,nx
!!$          do j=1,ny
!!$             res_xyw(i,j,h) = 0.0_wp
!!$             if (S_xy(i,j) .gt. 0.0) then
!!$                do k=1,ny
!!$                   if ( (S_xy(i,k) .gt. 0.0_wp) .and. (W1m0_xyw(i,k,h) .gt. S_xy(i,k)) ) then
!!$                      res_xyw(i,j,h) = res_xyw(i,j,h) + S_xy(i,k) * Q_yy(k,j)
!!$                   end if
!!$                end do
!!$             end if
!!$          end do
!!$       end do
!!$    end do
!!$    return
!!$  end subroutine IntBarCdQ

  ! this next subroutine should be able to replace both of the previous subroutines
  ! the number of calculations should be at least cut in half as a result.
  subroutine IntW1m0dQ(res_xyw, S_xy, W1m0_xyw, Q_yy)
    implicit none
    real(wp), intent(IN) :: S_xy(:,:), W1m0_xyw(:,:,:), Q_yy(:,:)
    real(wp), intent(INOUT) ::  res_xyw(:,:,:)
    integer nx, ny, nw
    integer h, i, j, k

    nx = size(res_xyw , 1)
    ny = size(res_xyw , 2)
    nw = size(res_xyw , 3)

    res_xyw = 0.0_wp

    do h=1,nw
       do i=1,nx
          do j=1,ny
             do k=1,ny
                res_xyw(i,j,h) = res_xyw(i,j,h) &
                     + (max(min(S_xy(i,k), W1m0_xyw(i,k,h)), 0.0_wp )) * Q_yy(k,j)
             end do
          end do
       end do
    end do
    return
  end subroutine IntW1m0dQ

!!$  subroutine IntM1dV(res_xyw, S_xy, V_y, beta)
!!$    implicit none
!!$    real(wp), intent(IN) :: S_xy(:,:), V_y(:), beta
!!$    real(wp), intent(INOUT) ::  res_xyw(:,:,:)
!!$    integer nx, ny, nw
!!$    integer h, i, j, k
!!$
!!$    nx = size(res_xyw , 1)
!!$    ny = size(res_xyw , 2)
!!$    nw = size(res_xyw , 3)
!!$
!!$    do h=1,nw
!!$       do i=1,nx
!!$          do j=1,ny
!!$             res_xyw(i,j,h) = 0.0
!!$             if (S_xy(i,j) .ge. 0.0) then
!!$                do k=1,ny
!!$                   if ((S_xy(i,k) .gt. S_xy(i,j))) then
!!$                      res_xyw(i,j,h) = res_xyw(i,j,h) + (beta*S_xy(i,k) + (1.0_wp-beta)*S_xy(i,j)) * V_y(k)
!!$                   end if
!!$                end do
!!$             end if
!!$          end do
!!$       end do
!!$    end do
!!$    return
!!$  end subroutine IntM1dV
!!$
!!$  subroutine IntM2dV(res_xyw, S_xy, W1m0_xyw, V_y , beta)
!!$    implicit none
!!$    real(wp), intent(IN) :: S_xy(:,:), W1m0_xyw(:,:,:), V_y(:), beta
!!$    real(wp), intent(INOUT) ::  res_xyw(:,:,:)
!!$    integer nx, ny, nw
!!$    integer h, i, j, k
!!$    
!!$    nx = size(res_xyw , 1)
!!$    ny = size(res_xyw , 2)
!!$    nw = size(res_xyw , 3)
!!$    
!!$    do h=1,nw
!!$       do i=1,nx
!!$          do j=1,ny
!!$             res_xyw(i,j,h) = 0.0_wp
!!$             if (S_xy(i,j) .ge. 0.0_wp) then
!!$                do k=1,ny
!!$                   if ((S_xy(i,j) .gt. S_xy(i,k)) &
!!$                        .and. (S_xy(i,k) .gt. 0.0_wp) &
!!$                        .and. (S_xy(i,k) .gt. W1m0_xyw(i,j,h) ) &
!!$                        ) then
!!$                      
!!$                      res_xyw(i,j,h) = res_xyw(i,j,h) &
!!$                           + ((beta*S_xy(i,j) + (1.0_wp-beta)*S_xy(i,k)) * v_y(k))
!!$                      
!!$                   end if
!!$                end do
!!$             end if
!!$          end do
!!$       end do
!!$    end do
!!$    return
!!$  end subroutine IntM2dV


  ! this subroutine should replace the two above with many fewer calculation
  subroutine IntW1m0dV(res_xyw, S_xy, W1m0_xyw, V_y , beta)
    implicit none
    real(wp), intent(IN) :: S_xy(:,:), W1m0_xyw(:,:,:), V_y(:), beta
    real(wp), intent(INOUT) ::  res_xyw(:,:,:)
    integer nx, ny, nw
    integer h, i, j, k
    
    nx = size(res_xyw , 1)
    ny = size(res_xyw , 2)
    nw = size(res_xyw , 3)

    res_xyw = 0.0_wp
    
    do h=1,nw
       do i=1,nx
          do j=1,ny
             do k=1,ny
                if ( W1m0_xyw(i,j,h) .LT. S_xy(i,k) ) then
                   res_xyw(i,j,h) = res_xyw(i,j,h) &
                        + ( min( S_xy(i,j),S_xy(i,k)) & 
                        + beta*max( S_xy(i,k)-S_xy(i,j),0.0_wp ) &
                        - W1m0_xyw(i,j,h) ) * v_y(k)
                end if
             end do
          end do
       end do
    end do
    return
  end subroutine IntW1m0dV
  
end module Compute_Surplus_Helper_Mod
