module solver_helper
  use glob
  implicit none

contains
  !Integral Computation
  subroutine IntSmSpdHxy(S,h,res)
    ! res(y) = \int_{y'|S(x,y')>S(x,y)} [S(x,y')-S(x,y)] v(y') dy'
    ! S was positive anyway so we can also condition on that
    implicit none
    real(wp), intent(IN) :: S(:,:)
    real(wp), intent(IN) :: h(:,:)
    real(wp), intent(INOUT)  :: res(:)
    integer nx,ny;
    integer y,x,y_prime;
    
    nx = size(S,1);
    ny = size(S,2);

    !with respect to Jeremy's code
    ! m <-> nx
    ! n <-> ny
    ! i <-> y
    ! j <-> x
    ! k <-> y_prime
    
    do y= 1 , ny
       res(y) = 0.0_wp
       do y_prime=1, ny
          do  x = 1 , nx
             if ( (S(x,y_prime)>= 0.0_wp) .and. (S(x,y) > S(x,y_prime))) then
                res(y) = res(y)+((S(x,y) - S(x,y_prime)) * h(x,y_prime))
             endif
          end do
       end do
    end do
  end subroutine IntSmSpdHxy

  !Integral Computation
  subroutine IntSdVy(S,v,res)
    ! res(x,y) = \int_{y'|S(x,y')>S(x,y)} S(x,y') v(y') dy'
    implicit none
    real(wp),  intent(IN) :: S(:,:)
    real(wp),  intent(IN) :: v(:)
    real(wp),  intent(INOUT)  :: res(:,:)
    integer nx,ny;
    integer y,x,y_prime;
    
    nx = size(S,1);
    ny = size(S,2);

    !with respect to Jeremy's code
    ! m <-> nx
    ! n <-> ny
    ! i <-> x
    ! j <-> y
    ! k <-> y_prime
    
    do x= 1 , nx
       do y=1 , ny
          res(x,y)=0.0_wp
          do  y_prime = 1 , ny
             if ( (S(x,y) >= 0.0_wp) .and. (S(x,y_prime) > S(x,y))) then
                res(x,y) = res(x,y)+S(x,y_prime) * v(y_prime)
             endif
          end do
       end do
    end do
  end subroutine IntSdVy
  
  !Integral Computation
  subroutine VM1(S,v,res)
    ! res(x,y) = \int_{y'|S(x,y')>S(x,y)} v(y') dy'
    implicit none
    real(wp),  intent(IN) :: S(:,:)
    real(wp),  intent(IN) :: v(:)
    real(wp),  intent(INOUT)  :: res(:,:)
    integer nx,ny;
    integer y,x,y_prime;
    
    nx = size(S,1);
    ny = size(S,2);
    
    do x= 1 , nx
       do y=1 , ny
          res(x,y)=0.0_wp
          do  y_prime = 1 , ny
             if ( (S(x,y)>= 0.0_wp) .and. (S(x,y_prime) > S(x,y))) then
                res(x,y) = res(x,y) + v(y_prime)
             endif
          end do
       end do
    end do
  end subroutine VM1


  !Integral Computation
  subroutine IntbarM1dHyp(S,h,res)
    ! res(x,y) = \int_{y'|S(x,y')>S(x,y)} v(y') dy'
    implicit none
    real(wp), intent(IN) :: S(:,:)
    real(wp), intent(IN) :: h(:,:)
    real(wp), intent(INOUT)  :: res(:,:)
    integer nx,ny;
    integer y,x,y_prime;
    
    nx = size(S,1);
    ny = size(S,2);
    
    do x= 1 , nx
       do y=1 , ny
          res(x,y)=0.0_wp
          do  y_prime = 1 , ny
             if ( (S(x,y_prime)>= 0.0_wp) .and. (S(x,y) > S(x,y_prime))) then
                res(x,y) = res(x,y) + h(x,y_prime)
             endif
          end do
       end do
    end do
  end subroutine IntbarM1dHyp
  
end module solver_helper

