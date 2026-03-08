module Compute_Surplus_Mod
	use glob
    use modeldef
 	use params
 	use compute_surplus_helper_mod
    use output

contains
  subroutine ComputeSurpluses(m,p)
    ! Script file to compute W(x,y,w).  The value associated with any given
    ! wage contract w
    implicit none
    type (ExogenousParameters), intent(in)  :: p
    type(model), intent(INOUT) :: m
    integer nx,ny,nw
    real(wp),allocatable :: wages_xyw(:,:,:) , W0_rep_xyw(:,:,:), S_rep_xyw(:,:,:)
    real(wp),allocatable :: M0_rep_xyw(:,:,:), W1m0_0_xyw(:,:,:) , W1m0_1_xyw(:,:,:)
    real(wp),allocatable :: VM1M2_xyw(:,:,:) 
    real(wp),allocatable :: IntW1m0dQ_xyw(:,:,:), IntW1m0dV_xyw(:,:,:)
!!$    IntCdQ_xyw(:,:,:), IntBarCdQ_xyw(:,:,:) ,  IntM1dV_xyw(:,:,:),  IntM2dV_xyw(:,:,:)
    real(wp) W1m0_dist, discounting

    integer Niter , i

    nx = p%gridSizeWorkers
    ny = p%gridSizeFirms
    nw = p%gridSizeWages

    !--------------------------------------
    !         ALLOCATING ARRAYS
    !--------------------------------------
    allocate( wages_xyw(nx,ny,nw) )
    allocate( W0_rep_xyw(nx,ny,nw) )
    allocate( S_rep_xyw(nx,ny,nw) )
    allocate( M0_rep_xyw(nx,ny,nw) )
    allocate( W1m0_0_xyw(nx,ny,nw) )
    allocate( W1m0_1_xyw(nx,ny,nw) )
    allocate( VM1M2_xyw(nx,ny,nw) )
    allocate( IntW1m0dQ_xyw(nx,ny,nw) )
    allocate( IntW1m0dV_xyw(nx,ny,nw) )
!!$    allocate( IntCdQ_xyw(nx,ny,nw) )
!!$    allocate( IntBarCdQ_xyw(nx,ny,nw) )
!!$    allocate( IntM1dV_xyw(nx,ny,nw) )
!!$    allocate( IntM2dV_xyw(nx,ny,nw) )

    !--------------------------------------
    !         STARTING LOOPS
    !--------------------------------------

    ! Create a nx-ny-nw array of wages
    wages_xyw = spread(spread( p%wvals_w , 1 , nx ) , 2 , ny)

    ! Expand W0 to be n-m-k array
    W0_rep_xyw = spread(spread(m%W0_x,2,ny), 3, nw)

    ! Expand M0 to be n-m-k
    M0_rep_xyw = spread(m%m_xy, 3 , nw)

    ! Initialize W1_0 if no value provided
    W1m0_0_xyw = M0_rep_xyw * W0_rep_xyw  
    W1m0_1_xyw = W1m0_0_xyw

    ! max number of iteration
    Niter = 1000

    !This integral is independent of W1(w,x,y)
    S_rep_xyw = spread(m%S_xy, 3 , nw)

    do i = 1 , Niter
     call VM1M2(VM1M2_xyw ,m%S_xy, W1m0_0_xyw , m%v_y)

!!$     call IntCdQ(IntCdQ_xyw ,m%S_xy , W1m0_0_xyw , m%Q_yy)
!!$     call IntBarCdQ(IntBarCdQ_xyw , m%S_xy , W1m0_0_xyw , m%Q_yy)
     
     call IntW1m0dQ(IntW1m0dQ_xyw , m%S_xy , W1m0_0_xyw , m%Q_yy)

!!$     call IntM1dV(IntM1dV_xyw , m%S_xy, m%v_y, p%beta)
!!$     call IntM2dV(IntM2dV_xyw , m%S_xy, W1m0_0_xyw, m%v_y , p%beta)

     call IntW1m0dV(IntW1m0dV_xyw, m%S_xy, W1m0_0_xyw, m%v_y, p%beta)
     
     ! I add W1m0 to both sides to gurentee convergence (contracting operator)

     discounting = 1.0_wp / ( 1.0_wp + p%r + p%chi + p%delta + p%zeta )

     W1m0_1_xyw = M0_rep_xyw *  &
          discounting * ( wages_xyw - (p%r + p%chi)*W0_rep_xyw &
          + p%delta * IntW1m0dQ_xyw &
          + p%s1 * m%kappa * IntW1m0dV_xyw &
          + W1m0_0_xyw ) 
      
     W1m0_dist =  maxval( abs(W1m0_1_xyw - W1m0_0_xyw) / maxval( abs(W1m0_1_xyw(:,:,nw/2))))**2
     
     if (mod(i,10) == -1) then  ! change to == 0 to see printed results
        print*, "---------------------------------  LOOPING ----------------------------"
        
        !print '( a , (ES9.2E2))', "denominator: " , maxval( abs(W1_1_xyw(:,:,nw/2)))
        print '( a , (i9) )', "number of iterations " , i
        print '( a , (ES9.2E2) , a , (ES9.2E2) )', "diff W1m0_xyw : " , W1m0_dist , " tol: " ,p%tol_W
        
     end if
     
     if ( W1m0_dist < p%tol_W) then
        exit
     end if
     
     W1m0_0_xyw = W1m0_1_xyw
  end do
  
  ! print '( a , (i9), a )', "Value Function W1 converged after " , i , " iterations "
  
  !Saving the value function
  m%W1_xyw = W1m0_1_xyw + spread(spread(m%W0_x,2,ny), 3, nw)
  
  !--------------------------------------
  !         DEALLOCATING ARRAYS
  !--------------------------------------
  deallocate( wages_xyw )
  deallocate( W0_rep_xyw )
  deallocate( S_rep_xyw )
  deallocate( M0_rep_xyw )
  deallocate( W1m0_0_xyw )
  deallocate( W1m0_1_xyw )
  deallocate( VM1M2_xyw )
!!$  deallocate( IntCdQ_xyw )
!!$  deallocate( IntBarCdQ_xyw )
!!$  deallocate( IntM1dV_xyw )
!!$  deallocate( IntM2dV_xyw )
  deallocate( IntW1m0dQ_xyw )
  deallocate( IntW1m0dV_xyw )
  
end subroutine ComputeSurpluses

end module compute_surplus_mod
