module compute_moments_mod

  use glob
  use modeldef
  use params
  use stat_helper
  use array_helper
  use output
  use simulate_wages_mod
  use moments_mod
  use sort_mod
  !  use MKL95_LAPACK, only: POTRF, POTRI    !  This should be sufficient to bring in the intel MKL LAPACK routines

contains

subroutine computeMoments(s, p, moms, COV, update)
    implicit none
    type (Moments) , intent(INOUT) :: moms
    type (SimulationWorkerPanel) , intent(IN)    :: s
    type (ExogenousParameters),    intent(in)    :: p
    real(wp), intent(INOUT) :: COV(:,:)
    logical, intent(IN) :: update

    integer :: nr, nt ,ns, ntmp        ! R individuals, T time periods
    integer :: years
    integer :: i, j, y, m
    real(wp),allocatable :: wage(:,:),  wage_y(:,:), mnths_empl_y(:,:)
    integer,allocatable  ::  E_monthly(:,:), F_monthly(:,:), cnts_empl_y(:,:), chng_job_y(:,:)
    real(wp) local_LF, local_E, local_U, tmp1, tmp2, tmp3, tmp4

    nr = p%nWorkerSimulation
    nt = p%nPeriodSimulation  ! total of nt months in simulation
    ns = p%momentsMonths      ! use the ns last months for moments
 
    allocate( wage(nr,ns) )
    allocate( E_monthly(nr,ns) )
    allocate( F_monthly(nr,ns) )

    allocate( wage_y(nr,20) )
    allocate( cnts_empl_y(nr,20) )
    allocate( chng_job_y(nr,20) )
    allocate( mnths_empl_y(nr,20) )

    
    ! wage used here is simulated wage times measurement error
    ! here we can change the variance of the measurement error
    wage = merge( (s%wage_rt(:,(nt-ns+1):nt))*(exp(p%sigma * global_merror ) ), 0.0_wp, s%status_rt(:,(nt-ns+1):nt) .eq. 1 )

    ! Employment indicator
    E_monthly =     s%status_rt(:,(nt - ns + 1):nt)
    ! Firm number
    F_monthly =     s%firm_id_rt(:,(nt - ns + 1):nt)
    
    ! initialize all moments to 0
    moms%E      = 0.0_wp  !!!!!
    moms%E2U    = 0.0_wp  !!!!!
    moms%U2E    = 0.0_wp  !!!!!
    moms%J2J    = 0.0_wp  !!!!!
    moms%w      = 0.0_wp  !!!!!
    moms%w2     = 0.0_wp  !!!!!
    moms%Dw     = 0.0_wp  !!!!!
    moms%Dw2    = 0.0_wp  !!!!!
    moms%Dw_EE  = 0.0_wp  !!!!!
    moms%Dw2_EE = 0.0_wp  !!!!!
    moms%Dw_DJ  = 0.0_wp  !!!!!
    moms%Dw2_DJ = 0.0_wp  !!!!!
    moms%V_U    = 0.0_wp  ! Need to calculate this somewhere else.

    ! initialize all annual values for workers to zero
    wage_y     = 0.0_wp
    mnths_empl_y = 0.0_wp
    cnts_empl_y = 1       ! initialize to 1 and change to zero if any spell of unemployment in the year
    chng_job_y  = 0       ! initialize to 0 and change to 1 if there is any job change during the 
    
    ! We calculate moments for each of the first 20 years of labour market experience
    do y=1,20
       local_LF = 0.0_wp ! count number of worker months
       local_E = 0.0_wp  ! count number of employed worker months (for exposure to job change or job loss)
       local_U = 0.0_wp  ! count number of unemployed worker months (for exposure to job finding)
       ! loop over all months in the current year y
       do m=1+12*(y-1),12*y
          ! loop over all workers
          do i=1,nr
             ! add up worker/month observations within the year
             local_LF = local_LF + 1.0_wp
             ! check if the worker is employed this month
             if ( E_monthly(i,m) .eq. 1 ) then
                ! add one to employment
                local_E = local_E + 1.0_wp
                moms%E(y) = moms%E(y) + 1.0_wp
                ! add up months wage for individual worker
                wage_y(i,y) = wage_y(i,y) + wage(i,m)
                mnths_empl_y(i,y) = mnths_empl_y(i,y) + 1.0_wp
                ! check if worker is also employed in the next period
                if ( (E_monthly(i,m) .eq. 1) .and. (E_monthly(i,m+1) .eq. 1) ) then
                   ! check if worker is employed at same firm in next period
                   if ( F_monthly(i,m) .ne. F_monthly(i,m+1) ) then
                      ! set indicator if worker changes job
                      chng_job_y(i,y) = 1                 ! indicator that this worker has changed jobs this year
                      moms%J2J(y) = moms%J2J(y) + 1.0_wp
                   end if
                      ! add one for those who are not employed next period (but were employed this period)
                else
                   moms%E2U(y) = moms%E2U(y) + 1.0_wp
                end if
             else 
                local_U = local_U + 1.0_wp
                if ( m .gt. 1 ) then ! the simulations records everyone as unemployed in month 1 so nobocy could be continuously empllyed in the first year if we include the first simulation month.
                   cnts_empl_y(i,y) = 0 ! change to 0 if there is any spell of unemployment during the year for this worker   
                end if
                if ( (E_monthly(i,m) .eq. 0) .and. (E_monthly(i,m+1) .eq. 1) ) then
                   moms%U2E(y) = moms%U2E(y) + 1.0_wp
                end if
             end if
          end do
       end do
          
       moms%E2U(y)    = moms%E2U(y) / local_E
       moms%J2J(y)    = moms%J2J(y) / local_E              
       moms%U2E(y)    = moms%U2E(y) / local_U             
       moms%E(y)      = moms%E(y)   / local_LF
       
    end do
    
    ! Now create wage moments
    ! replace sum of monthly wages with yearly average of monthly wages
    wage_y = merge( wage_y / mnths_empl_y, 0.0_wp, mnths_empl_y .gt. 0.0_wp )

    ! We will only use the first 19 years for estimation as year 20 data is very noisy
    do y=1,19
       ! local counters to do the conditioning
       tmp1 = 0.0_wp ! use to count number of workers continuously employed in both years
       tmp2  = 0.0_wp ! use to count the number of workers who are continuously employed in both years and change jobs
       tmp3  = 0.0_wp ! use to count the number of observations used in mean log wage (excludes those unemployed for entire year)
       tmp4  = 0.0_wp ! use to count the number of observations used in Dw (excluded those unemployed for entire year in either t or t+1)
       
       ! means
       do i=1,nr
          if ( wage_y(i,y) .gt. 0.0_wp ) then
             tmp3 = tmp3 + 1.0_wp
             moms%w(y) = moms%w(y) + log( wage_y(i,y) )
             if ( wage_y(i,y) .gt. 0.0_wp .and. wage_y(i,y+1) .gt. 0.0_wp ) then
                tmp4 = tmp4 + 1.0_wp
                moms%Dw(y) = moms%Dw(y) + ( log( wage_y(i,y+1) ) - log( wage_y(i,y) ) )
                if ( cnts_empl_y(i,y) .and. cnts_empl_y(i,y+1) ) then
                   moms%Dw_EE(y) = moms%Dw_EE(y) + (log( wage_y(i,y+1))-log(wage_y(i,y)))
                   tmp1 = tmp1 + 1.0_wp
                   if ( chng_job_y(i,y) .or. chng_job_y(i,y+1) ) then
                      moms%Dw_DJ(y) = moms%Dw_DJ(y) + (log(wage_y(i,y+1))-log(wage_y(i,y)))
                      tmp2 = tmp2 + 1.0_wp
                   end if
                end if
             end if
          end if
       end do
       
       ! the conditioning is necessary to guard against the rare events of no job changes etc
       if (tmp3 .gt. 0.0_wp) then
          moms%w(y) = moms%w(y) / tmp3
       end if
       if (tmp4 .gt. 0.0_wp) then
          moms%Dw(y) = moms%Dw(y) / tmp4
       end if
       if (tmp1 .ge. 0.0_wp) then 
          moms%Dw_EE(y) = moms%Dw_EE(y) / tmp1
       end if
       if (tmp2 .ge. 0.0_wp) then
          moms%Dw_DJ(y) = moms%Dw_DJ(y) / tmp2
       end if
       
       ! variances
       do i=1,nr
          if ( wage_y(i,y) .gt. 0.0_wp ) then
             moms%w2(y) = moms%w2(y) + ( log( wage_y(i,y) ) - moms%w(y) )**2
             if ( wage_y(i,y+1) .gt. 0.0_wp ) then
                moms%Dw2(y) = moms%Dw2(y) + &
                     ( log( wage_y(i,y+1) ) - log( wage_y(i,y) ) - moms%Dw(y) )**2
                if ( cnts_empl_y(i,y) .and. cnts_empl_y(i,y+1) ) then
                   moms%Dw2_EE(y) = moms%Dw2_EE(y) + &
                        ( log( wage_y(i,y+1) ) - log( wage_y(i,y) ) -  moms%Dw_EE(y) )**2
                   if ( chng_job_y(i,y) .or. chng_job_y(i,y+1) ) then
                      moms%Dw2_DJ(y) = moms%Dw2_DJ(y) + &
                           ( log( wage_y(i,y+1) ) - log( wage_y(i,y) ) - moms%Dw_DJ(y) )**2
                   end if
                end if
             end if
          end if
       end do
       
       if (tmp3 .gt. 0.0_wp) then
          moms%w2(y) = moms%w2(y) / tmp3
       end if
       if (tmp4 .gt. 0.0_wp) then
          moms%Dw2(y) = moms%Dw2(y) / tmp4
       end if
       if (tmp1 .ge. 0.0_wp) then 
          moms%Dw2_EE(y) = moms%Dw2_EE(y) / tmp1
       end if
       if (tmp2 .ge. 0.0_wp) then
          moms%Dw2_DJ(y) = moms%Dw2_DJ(y) / tmp2      
       end if
       
    end do

    ! just fill in year 20 with year 19 data as a place holder (not used in estimation)
    moms%w(20) = moms%w(19) 
    moms%Dw(20) = moms%Dw(19) 
    moms%Dw_EE(20) = moms%Dw_EE(19) 
    moms%Dw_DJ(20) = moms%Dw_DJ(19) 
    moms%w2(20) = moms%w2(19) 
    moms%Dw2(20) = moms%Dw2(19) 
    moms%Dw2_EE(20) = moms%Dw2_EE(19) 
    moms%Dw2_DJ(20) = moms%Dw2_DJ(19) 
    
    if (update .EQ. .TRUE. ) then
       
       COV = 0.0_wp  ! we don't use optimal weighting in this paper.
       
    end if
       
    ! clean up
    deallocate( wage )
    deallocate( E_monthly )
    deallocate( F_monthly )
    deallocate( wage_y ) 
    deallocate( cnts_empl_y )
    deallocate( chng_job_y )
    deallocate( mnths_empl_y )
    
  end subroutine computemoments
  
end module compute_moments_mod

