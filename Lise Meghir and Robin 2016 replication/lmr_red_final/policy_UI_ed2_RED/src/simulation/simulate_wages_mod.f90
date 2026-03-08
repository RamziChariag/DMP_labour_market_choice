module simulate_wages_mod
  use glob
  use modeldef
  use params
  use stat_helper
  use array_helper
  use output

  type SimulationWorkerPanel
     integer,  allocatable :: wage_index_rt(:,:)   !wage index for given individual at given time
     integer,  allocatable :: firm_type_rt(:,:)    !firm for given individual at given time
     integer,  allocatable :: firm_id_rt(:,:)      !firm for given individual at given time
     integer,  allocatable :: worker_type_r(:)     !worker type for given individual
     integer,  allocatable :: status_rt(:,:)       !satus for given individual at given time
     real(wp), allocatable :: wage_rt(:,:)         !wage for given individual at given time
     real(wp), allocatable :: renegociate_rt(:,:)  !whever renegociation occures
     real(wp), allocatable :: W1_rt(:,:)           !surplus for given individual at given time
     real(wp), allocatable :: W0_r(:)              !surplus for given individual at given time
     real(wp), allocatable :: S_r(:)               ! value of surplus used when calculating the wage bands 
  end type SimulationWorkerPanel

contains


  subroutine printCrossSectionToFile(s,p,filename,truncate)
    implicit none
    integer,optional,intent(in) :: truncate
    type (SimulationWorkerPanel) , intent(INOUT) :: s
    type (ExogenousParameters),    intent(in)    :: p
    character (*), intent(IN) :: filename
    integer :: nr, nt        ! R individuals, T time periods
    integer :: ii, tt    ! R individuals, T time periods (jj firm types in wage bounds)
    character(len=40)     :: fstr

    nr = p%nWorkerSimulation
    nt = p%nPeriodSimulation

	if(present(truncate)) then
		nr = min(nr,truncate)
	end if

    open(unit=24,file=filename)

    do ii = 1, nr
        do tt = 1, nt
            write(24, '(i5 ,1x , i5, 1x , i4 , 1x , i4 , 1x , i4 , 1x , ES18.8E3, 1x)')  &
            ii , tt , &
            s%worker_type_r(ii) , &
            s%firm_type_rt(ii,tt) ,  &
            s%status_rt(ii,tt), &
            s%wage_rt(ii,tt)
        end do
    end do

    close(24)

  end subroutine printCrossSectionToFile

  subroutine initSimulationWorkerPanel(s,p)
    implicit none
    type (SimulationWorkerPanel) , intent(INOUT) :: s
    type (ExogenousParameters),    intent(in)    :: p
    integer :: nr, nt        ! R individuals, T time periods

    nr = p%nWorkerSimulation
    nt = p%nPeriodSimulation

    allocate ( s%wage_rt(nr,nt) )
    allocate ( s%wage_index_rt(nr,nt) )
    allocate ( s%firm_type_rt(nr,nt) )
    allocate ( s%firm_id_rt(nr,nt) )
    allocate ( s%status_rt(nr,nt) )
    allocate ( s%renegociate_rt(nr,nt) )
    allocate ( s%W1_rt(nr,nt) )
    allocate ( s%W0_r(nr) )

    allocate ( s%worker_type_r(nr) )

  end subroutine initSimulationWorkerPanel

  subroutine initWorkerBounds(s,p)
    implicit none
    type (SimulationWorkerPanel) , intent(INOUT) :: s
    type (ExogenousParameters),    intent(in)    :: p
    integer :: nr, nt        ! R individuals, T time periods

    nr = p%gridSizeWorkers * p%gridSizeFirms
    nt = 2 

    allocate ( s%wage_rt(nr,nt) )
    allocate ( s%wage_index_rt(nr,nt) )
    allocate ( s%firm_type_rt(nr,1) )
    allocate ( s%firm_id_rt(nr,nt) )
    allocate ( s%status_rt(nr,nt) )
    allocate ( s%renegociate_rt(nr,nt) )
    allocate ( s%W1_rt(nr,nt) )
    allocate ( s%W0_r(nr) )
    allocate ( s%S_r(nr) )   
    allocate ( s%worker_type_r(nr) )

  end subroutine initWorkerBounds

  subroutine freeSimulationWorkerPanel(s)
    implicit none
    type (SimulationWorkerPanel) , intent(INOUT) :: s
    deallocate ( s%wage_rt )
    deallocate ( s%wage_index_rt )
    deallocate ( s%firm_type_rt )
    deallocate ( s%firm_id_rt )
    deallocate ( s%status_rt )
    deallocate ( s%renegociate_rt )
    deallocate ( s%W1_rt )
    deallocate ( s%W0_r )
    deallocate ( s%worker_type_r )
  end subroutine freeSimulationWorkerPanel

  subroutine freeWorkerBounds(s)
    implicit none
    type (SimulationWorkerPanel) , intent(INOUT) :: s
    deallocate ( s%wage_rt )
    deallocate ( s%wage_index_rt )
    deallocate ( s%firm_type_rt )
    deallocate ( s%firm_id_rt )
    deallocate ( s%status_rt )
    deallocate ( s%renegociate_rt )
    deallocate ( s%W1_rt )
    deallocate ( s%W0_r )
    deallocate ( s%worker_type_r )
    deallocate ( s%S_r )   
  end subroutine freeWorkerBounds

  subroutine SimulateWages(s, m, p )
    implicit none

    ! -------------- INPUTS --------------
    type (Model),                  intent(in)    :: m
    type (ExogenousParameters),    intent(in)    :: p
    ! ------------- OUTPUTS --------------
    type (SimulationWorkerPanel) , intent(INOUT) :: s

    ! -------------- LOCALS --------------
    integer :: nr, nt        ! R individuals, T time periods
    integer :: nx, ny, nw  ! dims of surplus funciton
    real(wp) ::lambda0,lambda1

    real(wp) :: temp_val

    ! CDF to draw new values
    real(wp), allocatable :: Qcdf_yy(:,:), Vcdf_y(:), Lcdf_x(:)  !Q(y'|y)

    ! others
    integer :: i,ii, tt          ! index workers and time
    integer :: x               ! the current worker type
    integer :: boundry, w_index  ! used to indicate wage is on the boundry

    real(wp) :: W1_tmp      ! W1xyw used
    real(wp),allocatable :: W1xywTmp_rt(:,:)
    integer :: renegotiate
    integer :: yprime

    nx = p%gridSizeWorkers
    ny = p%gridSizeFirms
    nw = p%gridSizeWages
    nr = p%nWorkerSimulation
    nt = p%nPeriodSimulation

    lambda0 = p%s0 * m%kappa * m%V
    lambda1 = p%s1 * m%kappa * m%V
    ! write(*,*) "lambda0: ", lambda0
    ! write(*,*) "lambda1: ", lambda1


    allocate( Qcdf_yy(ny ,ny))
    allocate( Vcdf_y(ny))
    allocate( Lcdf_x(nx))

    allocate( W1xywTmp_rt(nr,nt) )

    !--------------------------------------
    !           COMPUTE CDFS
    !--------------------------------------
    call pdf2cdf(m%v_y / m%v , Vcdf_y)
    call pdf2cdf(p%workersDistribution / p%nWorkers , Lcdf_x)

    do i=1,ny
       call pdf2cdf(m%Q_yy(:,i) , Qcdf_yy(:,i) )
    end do


    !--------------------------------------
    !        FOR EACH INDIVIDUAL
    !--------------------------------------
    do ii = 1,nr

       !---------------------------------
       !         INITIAL VALUES
       !---------------------------------
       call locate(Lcdf_x, global_rand_x_r(ii), x)
       x = min(x + 1, nx)
       s%worker_type_r(ii) = x
       s%W0_r(ii) = m%W0_x(x)
       s%W1_rt(ii,1) = s%W0_r(ii)
       s%firm_type_rt(ii,1) = 0
       s%status_rt(ii,1) = 0
       s%firm_id_rt(ii,1) = 0
       s%wage_index_rt(ii,1) = 1
       s%wage_rt(ii,1) = p%wvals_w(1)

       !---------------------------------
       !         FOR EACH PERIOD
       !---------------------------------
       do tt =2,nt
          renegotiate = 0

          !---------------------------------
          !       EMPLOYED WORKER
          !---------------------------------
          if (s%status_rt(ii, tt-1) .eq. 1) then
             
             !---------- FIRMS GETS PRODUCTIVITY SHOCK --------!
             if (global_rand_delta_rt(ii,tt) .le. p%delta) then
                call locate(Qcdf_yy(:,s%firm_type_rt(ii,tt-1)),global_rand_y_rt(ii,tt),yprime) ! location for y' draw
                yprime = min(yprime+1, ny)
                      
                !computing the surplus at the new productivity level for the worker
                W1xywTmp_rt(ii,tt) = m%W1_xyw(x, yprime ,s%wage_index_rt(ii ,tt-1) ) & 
                     + ( m%W1_xyw(x, yprime ,s%wage_index_rt(ii ,tt-1)+1)            & 
                     - m%W1_xyw(x ,yprime ,s%wage_index_rt(ii ,tt-1)  ) )            & 
                     * ( s%wage_rt(ii,tt-1) -  p%wvals_w(s%wage_index_rt(ii,tt -1))) & 
                     / (  p%wvals_w( s%wage_index_rt(ii, tt-1) +1)                   & 
                     - p%wvals_w( s%wage_index_rt(ii, tt-1)   ) )
                      
                !---------- MATCH SURPLUS IS NEGATIVE GIVEN NEW FIRM PRODUCTIVITY --------!
                !---------- -> MATCH IS DESTROYED --------!
                if (m%S_xy(x,yprime) .lt. 0.0) then
                   s%W1_rt(ii,tt) = s%W0_r(ii)
                   s%firm_type_rt(ii,tt) = 0
                   s%status_rt(ii,tt) = 0
                   s%firm_id_rt(ii,tt) = 0
                   renegotiate = 0
                   
                   !---------- CURRENT WORKER WAGE IS NOW UNOFFORDABLE FOR THE FIRM -------!
                   !---------- -> WAGE IS RENEGOCIATED / WORKER GETS TOTAL SURPLUS --------!
                else if ( ((W1xywTmp_rt(ii,tt) - s%W0_r(ii)) .ge. m%S_xy(x,yprime) )) then
                   s%W1_rt(ii,tt) = m%S_xy(x,yprime)+s%W0_r(ii)
                   s%firm_type_rt(ii,tt) = yprime
                   s%status_rt(ii,tt) = 1
                   s%firm_id_rt(ii,tt) = s%firm_id_rt(ii,tt-1)
                   renegotiate = 1
                         
                   !---------- CURRENT WORKER BETTER OFF UNEMPLOYED --------!
                   !---------- UNLESS WAGE CHANGES --------!
                else if (W1xywTmp_rt(ii,tt) .le. s%W0_r(ii) ) then
                   s%W1_rt(ii,tt) = s%W0_r(ii)
                   s%firm_type_rt(ii,tt) = yprime
                   s%status_rt(ii,tt) = 1
                   s%firm_id_rt(ii,tt) = s%firm_id_rt(ii,tt-1)
                   renegotiate = 1
                   
                   !---------- VALUE CHANGES BUT WAGE STAYS THE SAME --------!
                else
                   s%W1_rt(ii,tt) = W1xywTmp_rt(ii,tt)
                   s%firm_type_rt(ii,tt) = yprime
                   s%status_rt(ii,tt) = 1
                   s%firm_id_rt(ii,tt) = s%firm_id_rt(ii,tt-1)
                   renegotiate = 0
                   
                end if
             else

                !---------- EXOGENOUS DESTRUCTION OF THE MATCH --------!
                !---------- -> MATCH IS DESTROYED --------!
                if (global_rand_zeta_rt(ii,tt) .le. p%zeta) then
                   s%W1_rt(ii,tt) = s%W0_r(ii)
                   s%firm_type_rt(ii,tt) = 0
                   s%status_rt(ii,tt) = 0
                   s%firm_id_rt(ii,tt) = 0
                   renegotiate = 0
                else
                
                   !############# WORKER GETS OUTSIDE OFFER #############!
                   if (global_rand_lambda_rt(ii,tt) .le. lambda1) then
                      call locate(Vcdf_y, global_rand_y_rt(ii,tt), yprime) ! find position for y' draw
                      yprime = min(yprime+1, ny)
                      
                      !---------- THE NEW FIRM IS BETTER - WORKER LEAVES --------!
                      if (m%S_xy(x, yprime) .gt. m%S_xy(x, s%firm_type_rt(ii, tt-1))) then
                         s%W1_rt(ii,tt) =  p%beta*m%S_xy(x, yprime) + (1-p%beta) *m%S_xy(x,s%firm_type_rt(ii, tt-1)) + s%W0_r(ii)
                         s%firm_type_rt(ii,tt) = yprime
                         s%status_rt(ii,tt) = 1
                         s%firm_id_rt(ii,tt) = yprime
                         renegotiate = 1
                      
                         !---------- CURRENT FIRM IS BETTER BUT NEED TO OFFER WAGE INCREASE --------!
                      else if (  (m%S_xy(x,s%firm_type_rt(ii,tt-1)) .ge. m%S_xy(x,yprime))  &
                           .and.(m%S_xy(x,yprime) .ge. (s%W1_rt(ii,tt-1) - s%W0_r(ii)))) then
                         s%W1_rt(ii,tt) = m%S_xy(x,yprime) + s%W0_r(ii)
                         
                         s%firm_type_rt(ii,tt) = s%firm_type_rt(ii,tt-1)
                         s%status_rt(ii,tt) = 1
                         s%firm_id_rt(ii,tt) = s%firm_id_rt(ii,tt-1)
                         renegotiate = 1
                         
                         !---------- CURRENT FIRM IS BETTER, NO NEED TO OFFER WAGE INCREASE --------!
                      else
                         s%W1_rt(ii,tt) = s%W1_rt(ii,tt-1)
                         s%firm_type_rt(ii,tt) = s%firm_type_rt(ii,tt-1)
                         s%status_rt(ii,tt) = 1
                         s%firm_id_rt(ii,tt) = s%firm_id_rt(ii,tt-1)
                         renegotiate = 0
                         
                      end if
                   else
                      !---------- NOthing changes --------!
                      s%W1_rt(ii,tt) = s%W1_rt(ii,tt-1)
                      s%firm_type_rt(ii,tt) = s%firm_type_rt(ii,tt-1)
                      s%status_rt(ii,tt) = 1
                      s%firm_id_rt(ii,tt) = s%firm_id_rt(ii,tt-1)
                      renegotiate = 0
                   end if
                end if
             end if
          end if
          
          !---------------------------------
          !       UNEMPLOYED WORKER
          !---------------------------------
          if  (s%status_rt(ii, tt-1) .eq. 0) then
             s%W1_rt(ii,tt) = s%W0_r(ii)
             s%firm_type_rt(ii,tt) = 0
             s%status_rt(ii,tt) = 0
             s%firm_id_rt(ii,tt) = 0
             renegotiate = 0
             
             if (global_rand_lambda_rt(ii,tt) .le. lambda0) then
                call locate(Vcdf_y, global_rand_y_rt(ii,tt), yprime)
                yprime = min(yprime+1, ny)
                if (m%S_xy(x,yprime) .ge. 0.0_wp) then
                   s%W1_rt(ii,tt) = p%beta * m%S_xy(x,yprime) + s%W0_r(ii)
                   s%firm_type_rt(ii,tt) = yprime
                   s%firm_id_rt(ii,tt) = yprime
                   s%status_rt(ii,tt) = 1
                   renegotiate = 1
                end if
             end if
          end if
          

            !-------------------------------------------
            !               COMPUTING WAGE
            !-------------------------------------------
          
          if (s%status_rt(ii,tt) .eq. 1) then
             if (renegotiate .eq. 1) then
                
                !locate the correct w_index
                call locate(m%W1_xyw(x, s%firm_type_rt(ii,tt),:), s%W1_rt(ii,tt), w_index)
                
                if (s%W1_rt(ii,tt) .lt. m%W1_xyw(x,s%firm_type_rt(ii,tt),1)) then
                   w_index = 1
                else if (s%W1_rt(ii,tt) .gt. m%W1_xyw(x,s%firm_type_rt(ii,tt),nw)) then
                   w_index = nw-1
                end if
                s%wage_index_rt(ii,tt) = max(1, w_index)
                if (w_index .gt. 1) then

                   s%wage_rt(ii,tt) = p%wvals_w(s%wage_index_rt(ii,tt)) + &
                        (p%wvals_w(s%wage_index_rt(ii,tt)+1) - p%wvals_w(s%wage_index_rt(ii,tt) )) &
                        * (s%W1_rt(ii,tt) - m%W1_xyw(x,s%firm_type_rt(ii,tt),s%wage_index_rt(ii,tt) )) &
                        / (m%W1_xyw(x ,s%firm_type_rt(ii,tt),s%wage_index_rt(ii,tt)+1) &
                        - m%W1_xyw(x,s%firm_type_rt(ii ,tt),s%wage_index_rt(ii,tt)) )

                else
                   s%wage_rt(ii,tt) = p%wvals_w(1)
                end if

             else
                s%wage_rt(ii,tt) = s%wage_rt(ii,tt-1)
                s%wage_index_rt(ii,tt) = s%wage_index_rt(ii,tt-1)
             endif

          else
             s%wage_rt(ii,tt) = p%wvals_w(1)   ! For unemployed workers
             s%wage_index_rt(ii,tt) = 1
          end if

       end do
    end do

!!$    !EXPORTING THE DATA
!!$    call writeToCSVvec('worker_type.dat',  real(s%worker_type_r,wp))
!!$    call writeToCSV('panel_wage.dat', s%wage_rt)
!!$    call writeToCSV('panel_firm_id.dat', real(s%firm_id_rt,wp))
!!$    call writeToCSV('panel_firm_type.dat', real(s%firm_type_rt,wp))
!!$    call writeToCSV('panel_status.dat', real(s%status_rt,wp))
    

    ! clean up
    deallocate( Qcdf_yy )
    deallocate( Vcdf_y )
    deallocate( Lcdf_x )
    deallocate( W1xywTmp_rt )
    
    return
  end subroutine SimulateWages

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine WagesBounds(s, m, p )
    implicit none

    ! -------------- INPUTS --------------
    type (Model),                  intent(in)    :: m
    type (ExogenousParameters),    intent(in)    :: p
    ! ------------- OUTPUTS --------------
    type (SimulationWorkerPanel) , intent(INOUT) :: s

    ! -------------- LOCALS --------------
    integer :: nr, nt        ! R individuals, T time periods
    integer :: nx, ny, nw  ! dims of surplus funciton
    !    real(wp) ::lambda0,lambda1

    ! random numbers for simulation
    !    real(wp),allocatable :: global_rand_x_r(:)
    !    real(wp),allocatable :: global_rand_delta_rt(:,:), global_rand_zeta_rt(:,:) ,global_rand_y_rt(:,:), global_rand_lambda_rt(:,:)
    !    real(wp) :: temp_val

    ! CDF to draw new values
    !    real(wp), allocatable :: Qcdf_yy(:,:), Vcdf_y(:), Lcdf_x(:)  !Q(y'|y)

    ! others
    integer :: i,ii, tt  , jj  ! index workers and time (index firm s by jj)
    integer :: x               ! the current worker type
    integer :: boundry, w_index  ! used to indicate wage is on the boundry

    real(wp) :: W1_tmp      ! W1xyw used
    ! real(wp),allocatable :: W1xywTmp_rt(:,:)
    integer :: renegotiate
    integer :: yprime

    nx = p%gridSizeWorkers
    ny = p%gridSizeFirms
    nw = p%gridSizeWages

    nr = nx * ny ! number of (x,y) possible matches
    nt = 2       ! calculate lowest and highest wage for (x,y)

    !    lambda0 = p%s0 * m%kappa * m%V
    !    lambda1 = p%s1 * m%kappa * m%V
    ! write(*,*) "lambda0: ", lambda0
    ! write(*,*) "lambda1: ", lambda1

!!$    allocate( global_rand_x_r(nr) )
!!$    allocate( global_rand_delta_rt(nr,nt) )
!!$    allocate( global_rand_zeta_rt(nr,nt) )
!!$    allocate( global_rand_lambda_rt(nr,nt) )
!!$    allocate( global_rand_y_rt(nr,nt) )
!!$
!!$    allocate( Qcdf_yy(ny ,ny))
!!$    allocate( Vcdf_y(ny))
!!$    allocate( Lcdf_x(nx))
!!$
!!$    allocate( W1xywTmp_rt(nr,nt) )

!!$    !--------------------------------------
!!$    !           COMPUTE CDFS
!!$    !--------------------------------------
!!$    call pdf2cdf(m%v_y / m%v , Vcdf_y)
!!$    call pdf2cdf(p%workersDistribution / p%nWorkers , Lcdf_x)
!!$
!!$    do i=1,ny
!!$       call pdf2cdf(m%Q_yy(:,i) , Qcdf_yy(:,i) )
!!$    end do
!!$
!!$    !--------------------------------------
!!$    !           GENERATE DRAWS
!!$    !--------------------------------------
!!$    call initSeed(1234567)
!!$
!!$    call randVec( global_rand_x_r(1:nr/2) )
!!$    global_rand_x_r(1+nr/2:nr) = 1.0_wp - global_rand_x_r(1:nr/2)
!!$
!!$	call randMat( global_rand_delta_rt )
!!$    !call randMat( global_rand_delta_rt(1:nr/2 ,:) )
!!$    !global_rand_delta_rt((1+(nr/2)):nr,1:nt) = 1.0_wp - global_rand_delta_rt(1:nr/2 ,:)
!!$
!!$	call randMat( global_rand_zeta_rt)
!!$    !call randMat( global_rand_zeta_rt(1:nr/2 ,:) )
!!$    !global_rand_zeta_rt(1+nr/2:nr,:) = 1.0_wp - global_rand_zeta_rt(1:nr/2 ,:)
!!$
!!$	call randMat( global_rand_lambda_rt )
!!$    !call randMat( global_rand_lambda_rt(1:nr/2 ,:) )
!!$    !global_rand_lambda_rt(1+nr/2:nr,:) = 1.0_wp - global_rand_lambda_rt(1:nr/2 ,:)
!!$
!!$	call randMat( global_rand_y_rt )
!!$    !call randMat( global_rand_y_rt(1:nr/2 ,:) )
!!$    !global_rand_y_rt(1+nr/2:nr,:) = 1.0_wp - global_rand_y_rt(1:nr/2 ,:)
    
    !--------------------------------------
    !        FOR EACH INDIVIDUAL
    !--------------------------------------
    do ii = 1,nx

       !--------------------------------------
       !        FOR EACH INDIVIDUAL
       !--------------------------------------
       do jj = 1,ny

          !---------------------------------
          !         INITIAL VALUES
          !---------------------------------
          
          s%worker_type_r( (ii-1)*ny + jj ) = ii
          s%firm_type_rt( (ii-1)*ny + jj   ,1) = jj 

          ! worker outside option and surplus
          s%W0_r( (ii-1)*ny + jj ) = m%W0_x( ii )
          s%S_r( (ii-1)*ny + jj ) = m%S_xy( ii,jj )
          
          !-------------------------------------------
          !               COMPUTING WAGE
          !-------------------------------------------
          
          ! Compute lowest wage at (x,y) match (share beta of surplus to worker)
          !locate the correct w_index for starting wage
          call locate(m%W1_xyw(ii, jj, :), p%beta*m%S_xy( ii,jj )+m%W0_x( ii ), w_index)
          
          if (  p%beta*m%S_xy( ii,jj )+m%W0_x( ii ) .lt. m%W1_xyw(ii, jj, 1) ) then
             w_index = 1
          else if (   p%beta*m%S_xy( ii,jj )+m%W0_x( ii ) .gt. m%W1_xyw(ii, jj, nw) ) then
             w_index = nw-1
          end if
          w_index = max(1, w_index)         
          if (w_index .gt. 1) then
             
             s%wage_rt( (ii-1)*ny + jj ,1) = p%wvals_w(w_index) &
                  + (p%wvals_w(w_index+1) - p%wvals_w(w_index)) &
                  * ( p%beta*m%S_xy( ii,jj ) + m%W0_x( ii ) - m%W1_xyw(ii, jj, w_index) ) &
                  / (m%W1_xyw(ii, jj, w_index+1) - m%W1_xyw(ii, jj, w_index ) )
             
          else
             s%wage_rt(ii,tt) = p%wvals_w(1)
          end if
          
          
          ! Compute highest wage at (x,y) match (all surplus to worker)
          !locate the correct w_index for starting wage
          call locate(m%W1_xyw(ii, jj, :), m%S_xy( ii,jj )+m%W0_x( ii ), w_index)
          
          if ( m%S_xy( ii,jj )+m%W0_x( ii ) .lt. m%W1_xyw(ii, jj, 1) ) then
             w_index = 1
          else if ( m%S_xy( ii,jj )+m%W0_x( ii ) .gt. m%W1_xyw(ii, jj, nw) ) then
             w_index = nw-1
          end if
          w_index = max(1, w_index)         
          if (w_index .gt. 1) then

             s%wage_rt( (ii-1)*ny + jj ,2) = p%wvals_w(w_index) &
                  + (p%wvals_w(w_index+1) - p%wvals_w(w_index)) &
                  * ( m%S_xy( ii,jj ) + m%W0_x( ii ) - m%W1_xyw(ii, jj, w_index) ) &
                  / ( m%W1_xyw(ii, jj, w_index+1) - m%W1_xyw(ii, jj, w_index ) )
             
          else
             s%wage_rt(ii,tt) = p%wvals_w(1)
          end if
          
       
       end do
    end do

    !EXPORTING THE DATA
    call writeToCSVvec('bound_worker_type.dat',  real(s%worker_type_r,wp) )
    call writeToCSV('bound_firm.dat', real(s%firm_type_rt, wp) )
    call writeToCSVvec('bound_surplus.dat', s%S_r )
    call writeToCSV('bound_wage.dat', s%wage_rt )
    
!!$    ! clean up
!!$    deallocate( global_rand_x_r )
!!$    deallocate( global_rand_delta_rt )
!!$    deallocate( global_rand_zeta_rt )
!!$    deallocate( global_rand_lambda_rt )
!!$    deallocate( global_rand_y_rt )
!!$    deallocate( Qcdf_yy )
!!$    deallocate( Vcdf_y )
!!$    deallocate( Lcdf_x )
!!$    deallocate( W1xywTmp_rt )
!!$    
    return
  end subroutine WagesBounds


end module simulate_wages_mod
