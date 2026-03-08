module mpi_mcmc_mod

  ! this is a stripped down version of
  ! http://www.bris.ac.uk/boris/jobs/ads?ID=83993

  use fmpioptglob
  use output
  use rand_helper
  use sort_mod
  include 'mpif.h'

  type MCMCParams
     integer :: parameter_size     ! size of the parameter vector
     integer :: chain_count        ! number of parrallel chains
     integer :: chain_length       ! number of parrallel chains
     integer :: delta			  ! number of elements in the difference to compute candidates
     real(wpfmpi),allocatable :: prior_lower_bound(:) ! bounds to use for the prior uniform distribution
     real(wpfmpi),allocatable :: prior_upper_bound(:) ! bounds to use for the prior uniform distribution
     real(wpfmpi) :: CR 			  ! cretaria for how dimension to update
     integer :: max_iteration
     integer :: stat_display_rate
     real(wpfmpi) :: shock_add_std
     real(wpfmpi) :: shock_mult_std
     real(wpfmpi) :: initial_posterior_value
     logical 	 :: dump_chain
  end type MCMCParams

  type MCMCChains
     real(wpfmpi),allocatable :: thetas_knl(:,:,:) ! bounds to use for the prior uniform distribution
     real(wpfmpi),allocatable :: posts_nl(:,:) ! bounds to use for the prior uniform distribution
  end type MCMCChains

  save

  integer :: procId !proc id on the mpi grid
  integer :: nprocs !number of procs

contains

  subroutine MpiMCMCInit
    implicit none
    integer :: ierror

    call MPI_INIT(ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, procId, ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierror)
  end subroutine MpiMCMCInit

  subroutine MCMCParamInit(op)
    implicit none
    type(MCMCParams), intent(IN) :: op
  end subroutine MCMCParamInit

  subroutine MCMCChainsInit(ch,op)
    implicit none
    type(MCMCParams), intent(IN) :: op
    type(MCMCChains), intent(INOUT) :: ch
    integer :: K, L, N
    K = op%parameter_size
    L = op%chain_length
    N = op%chain_count

    allocate(ch%thetas_knl(K,N,L))
    allocate(ch%posts_nl(N,L))
  end subroutine MCMCChainsInit

  ! This methods recieve a list of parameter values in theta.
  ! For each of the values, it sends it to a processor, then wait
  ! for the results that are returned in the same order in vals
  ! IN    thetas(:,:)  (nparams x number_of_evaluations)
  ! INOUT vals(:)      (number_of_evaluations)
  !
  ! The idea here is to do blocking send/recieve everywhere
  ! but on the recieve side of the master
  subroutine Master_ComputeInParrallel(thetas , vals, op)
    implicit none
    type(MCMCParams), intent(IN) :: op
    real(wpfmpi), intent(IN) :: thetas(:,:)
    real(wpfmpi), intent(INOUT) :: vals(:)
    real :: tmp_val
    real,allocatable :: theta_tmp(:)

    integer,allocatable :: requests(:)
    integer :: ntasks,itask
    integer :: left_to_recieve
    integer :: ierror

    ntasks = size(thetas,2)
    allocate( requests(ntasks))
    allocate( theta_tmp(op%parameter_size))

    !SENDING THE REQUESTS
    !eventually we should do this in a smarter way where we
    !we queue requests
    !write(*,*) "[mpi_mcmc_mod:master] submitting " , ntasks , " tasks"
    do itask=1,ntasks
       !write(*,*) "[mpi_mcmc_mod:master] sending theta to " , itask
       theta_tmp = real(thetas(:,itask))
       !call printVec(real(theta_tmp,wpfmpi))
       call MPI_Send(theta_tmp, op%parameter_size, MPI_REAL, itask, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
       !write(*,*) "[mpi_mcmc_mod:master] done sending"
    end do
    left_to_recieve = ntasks

    !FOR NOW BLOKING RECIEVE
    do itask=1,ntasks
       !write(*,*) "[mpi_mcmc_mod:master] ready to receive value from : " , itask
       call MPI_Recv(tmp_val, 1, MPI_REAL, itask, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
       vals(itask) = real(tmp_val,wpfmpi)
       !write(*,*) "[mpi_mcmc_mod:master] got value ", tmp_val , " from:" , itask
    end do
    !write(*,*) "[mpi_mcmc_mod:master] received " , ntasks , " values"

    !DONE - CLEANING
    deallocate(requests)

  end subroutine Master_ComputeInParrallel

  subroutine Slaves_ComputeValues_Loop(op,objfunc)
    implicit none
    type(MCMCParams), intent(IN) :: op
    real,allocatable :: theta(:)
    real(wpfmpi) :: value
    integer :: ierror
    interface
       subroutine objfunc(params,res)
         use fmpioptglob
         real(wpfmpi) :: params(:)
         real(wpfmpi) :: res
       end subroutine objfunc
    end interface

    ! INIT PARAMETERS
    allocate (theta(op%parameter_size))

    do while (.TRUE.)
       !write(*,*) "[mpi_mcmc_mod:slave] " ,procid , ": ready to receive"
       call MPI_Recv(theta, op%parameter_size, MPI_REAL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
       !write(*,*) "[mpi_mcmc_mod:slave] " , procid , ": Recieved a parameter"

       if (sum(abs(theta)) .eq. 0.0) then
          write(*,*) "[mpi_mcmc_mod:slave] ", procid, " received null param, so exits"
          return
       end if

       ! CALLING THE OBJECTIVE FUNCTION
       call objfunc(real(theta,wpfmpi),value)
       !call sleep (1)

       !write(*,*) "[mpi_mcmc_mod:slave] " , procid , "value: " ,  value
       call MPI_Send(real(value) , 1, MPI_REAL, 0 , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
       !write(*,*) "[mpi_mcmc_mod:slave] " , procid , ": value sent back"
    end do
  end subroutine Slaves_ComputeValues_Loop

  !  subroutine StartOptimizer(initial_theta ,solution_theta,ch, op, objfunc)
  !	implicit none
  !	real(wpfmpi),intent(IN) :: initial_theta(:)
  !	real(wpfmpi),intent(INOUT) :: solution_theta(:)
  !	type(MCMCParams), intent(IN) :: op
  !    type(MCMCChains), intent(INOUT) :: ch
  !	interface
  !      subroutine objfunc(params,res)
  !      	use fmpioptglob
  !        real(wpfmpi) :: params(:)
  !        real(wpfmpi) :: res
  !      end subroutine objfunc
  !    end interface
  !
  !    if(procId .ne. 0) then
  !	    call Slaves_ComputeValues_Loop(op,objfunc)
  !	    write(*,*) "salve is done"
  !	  else
  !	  	call StartOptimizerMaster(initial_theta ,solution_theta,ch, op)
  !	    call MasterReleaseSlaves(op)
  !	  end if
  !
  !	  write(*,*) procid , " is Done with the simulation!"
  !  end subroutine StartOptimizer

  !##########################################################################
  ! 				OPTIMIZATION PROCEDURE
  !##########################################################################

  subroutine StartOptimizerMaster(initial_theta ,solution_theta,ch, op)
    implicit none
    type(MCMCParams), intent(IN) :: op
    type(MCMCChains), intent(INOUT) :: ch

    real(wpfmpi),intent(IN) :: initial_theta(:)
    real(wpfmpi),intent(INOUT) :: solution_theta(:)

    integer :: I, P, D, C ,J
    integer :: K, L, N
    integer :: loop_count
    real(wpfmpi),allocatable :: all_population_knl(:,:,:) ! a cyclical buffer to store the population
    real(wpfmpi),allocatable :: one_population_kn(:,:)    ! a cyclical buffer to store the population
    real(wpfmpi),allocatable :: candidate_population_kn(:,:)    ! a cyclical buffer to store the population
    real(wpfmpi),allocatable :: param_candidate(:)            ! a vector use to locally store a param vector
    real(wpfmpi),allocatable :: one_population_posteriors_n(:)  ! storing the posterior for the population
    real(wpfmpi),allocatable :: all_population_posteriors_nl(:,:)  ! storing the posterior for the population
    real(wpfmpi),allocatable :: candidate_population_posteriors_n(:)  ! storing the posterior for the population
    real(wpfmpi),allocatable :: populations_logmean_n(:)
    integer,allocatable :: index_logmean_n(:)

    real(wpfmpi), allocatable :: alpha(:)         ! acceptance probability
    real(wpfmpi), allocatable :: rand_acceptance(:)

    real(wpfmpi) :: gamma , IQR, Q3, Q1
    integer :: best_chain

    integer, allocatable :: vec_index(:)

    logical, allocatable :: last_half_mask_l(:)     ! mask that captures the last 50 %
    logical, allocatable :: logmask_nl(:,:)
    integer, allocatable :: all_acceptance_results_nl(:,:)
    logical, allocatable :: all_outliner_results_nl(:,:)
    logical, allocatable :: within_prior_mask_n(:)

    integer, allocatable :: est_max_loc(:)
    character(len=40)     :: Lfilename
    integer :: buffer_loop

    K = op%parameter_size
    L = op%chain_length
    N = op%chain_count
    P = nprocs

    allocate( all_acceptance_results_nl(N,L))
    allocate( all_outliner_results_nl(N,L))
    allocate( all_population_knl(K,N,L))
    allocate( one_population_kn(K,N))
    allocate( candidate_population_kn(K,N))
    allocate( one_population_posteriors_n(N))
    allocate( populations_logmean_n(N))
    allocate( index_logmean_n(N))
    allocate( within_prior_mask_n(N))
    allocate( candidate_population_posteriors_n(N))
    allocate( all_population_posteriors_nl(N,L))

    allocate( param_candidate(K))

    allocate (alpha(N))
    allocate (rand_acceptance(N))
    allocate (vec_index(L))
    allocate (last_half_mask_l(L))
    allocate (logmask_nl(N,L))

    allocate (est_max_loc(2))

    do I=1,L
       vec_index(I)=I
    end do

    all_acceptance_results_nl = 0
    all_outliner_results_nl = .false.
    all_population_posteriors_nl = op%initial_posterior_value

    !------------------------------------------------------
    !CREATE THE FIRST POPULATION
    !------------------------------------------------------
    I = 1
    !call randMat( one_population_kn )
    !call printMat(one_population_kn)
    !one_population_kn = spread(op%prior_lower_bound, 2 , N) + &
    !					(spread(op%prior_upper_bound - op%prior_lower_bound, 2 , N)) * one_population_kn

    one_population_kn = spread(initial_theta,2,N)

    !call printVec(op%prior_upper_bound)
    !call printVec(op%prior_lower_bound)
    all_population_knl(:,:,I) = one_population_kn;
    call printMat(one_population_kn)

    !EVALUATE EACH MENMBER OF THE CURRENT POPULATION
    call log4mat_debug_real("mpi_mcmc_mod", 1, "evaluation of first the population", 0.0_wpfmpi)
    call Master_ComputeInParrallel(one_population_kn, one_population_posteriors_n,op)
    all_population_posteriors_nl(:,I) = one_population_posteriors_n
    !call printvec(one_population_posteriors_n)

    !-----------------------------------------------------
    !			      CHAINS EVOLUTION
    !-----------------------------------------------------
    ! REPEATING FOR EVER!
    loop_count = 0
    buffer_loop=0
    DO WHILE (loop_count < op%max_iteration)
       loop_count = loop_count + 1

       one_population_kn = all_population_knl(:,:,I)
       one_population_posteriors_n = all_population_posteriors_nl(:,I)
       ! ******* COMPUTING NEW CANDIDATES ******
       do C =1,N
          do J=1,100 ! TODO: better to reflect - what happens after 20?
             call ComputeParamCandidate(one_population_kn,param_candidate,C,op, mod(I,10) .eq. 0 )
             candidate_population_kn(:,C) = param_candidate

             ! check if candidate is in prior
             if (all(param_candidate > op%prior_lower_bound) .and. &
                  all(param_candidate < op%prior_upper_bound)) then
                candidate_population_kn(:,C) = param_candidate
                EXIT
             end if
          end do

          if (J > 90) then
            write(*,*) "!!!!!! cannot find a guess within the prior: ", J
          end if

          !write(*,*) "Prev and candidate"
          !call PrintVec(	one_population_kn(:,C))
          !call PrintVec(	param_candidate)
       end do

       ! ******* COMPUTING POSTERIORS FOR CANDIDATES ******
       ! call log4mat_debug_real("mpi_mcmc_mod", 1, "evaluation posterior for candidates", 0.0_wpfmpi)
       ! write(*,*) "candidates:"
       ! call printMat(candidate_population_kn)
       call Master_ComputeInParrallel(candidate_population_kn, candidate_population_posteriors_n, op)
       !write(*,*) "results:"
       !call printVec(candidate_population_posteriors_n)
       !call log4mat_debug_real("mpi_mcmc_mod", 1, "evaluation of posteriors done", 0.0_wpfmpi)

       !multiply candidate by prior
       !should check before hand and not evaluat!!!!!!!
       within_prior_mask_n = &
            all((candidate_population_kn > spread(op%prior_lower_bound, 2,N) ) .and. &
            (candidate_population_kn < spread(op%prior_upper_bound, 2,N) ) ,1)

       !INCREMENT I
       I = mod(I,L) + 1

       ! ******* COMPUTING ACCEPTS / REJECTS ******
       !
       !! THIS IS AN ATTEMPT TO ADD SIMULATED ANNEALING AT THE BEGINNING
!!$               alpha = merge(exp( (candidate_population_posteriors_n -  one_population_posteriors_n) &
!!$                                   / max(1.0_wpfmpi, real(op%max_iteration - loop_count, wpfmpi) )) , &
!!$                  1.1_wpfmpi ,  one_population_posteriors_n .GT. -100000000000.0_wpfmpi )

       ! USE THE NEXT LINE TO DO TRUE MCMC
        alpha = merge(exp(candidate_population_posteriors_n -  one_population_posteriors_n) , &
	       1.1_wpfmpi ,  one_population_posteriors_n .GT. -100000000000.0_wpfmpi )
!!$
       ! Use exp(X_t - X_{t-1}) if (X_t - X_{t-1}) < 0, and 1.1 otherwise.
       !alpha = merge(candidate_population_posteriors_n / one_population_posteriors_n , 1.1_wpfmpi ,one_population_posteriors_n > 0)
       ! REJECT IF OUTSIDE OF PRIOR
       alpha = merge(0.0_wpfmpi , alpha ,  within_prior_mask_n .eq. .false.)
       !max at 1
       alpha = min(1.1_wpfmpi , alpha)

       call randvec(rand_acceptance)

       do C = 1,N

          !write(*,*) "pop, candidates, give posterior: " ,  candidate_population_posteriors_n(C), " vs ", one_population_posteriors_n(C)
          !call printVec(one_population_kn(:,C))
          !call printVec(candidate_population_kn(:,C))

          ! CHECK IF WE REJECT THE NEW POSTERIOR
          if (rand_acceptance(C) .ge. alpha(C)) then
             !if (alpha(C) < 1 ) then
             candidate_population_posteriors_n(C) = one_population_posteriors_n(C)
             candidate_population_kn(:,C) = one_population_kn(:,C)
             if ( alpha(C) > 0.0_wpfmpi) then
                all_acceptance_results_nl(C,I) = 0
             else
                all_acceptance_results_nl(C,I) = -1
             end if
          else
             if ( alpha(C) > 1.0_wpfmpi) then
                all_acceptance_results_nl(C,I) = 3
             elseif ( alpha(C) == 1.0_wpfmpi) then
                all_acceptance_results_nl(C,I) = 2
             else
                all_acceptance_results_nl(C,I) = 1
             end if
          end if
       end do

       !call log4mat_debug_real("mpi_mcmc_mod", 1, "saving the populations", 0.0_wpfmpi)
       !SAVE THE POPULATION AND INCREMENT I
       all_population_knl(:,:,I)         = candidate_population_kn
       all_population_posteriors_nl(:,I) = candidate_population_posteriors_n

       !-----------------------------------------------------
       !			      CHAINS REPLACEMENTS
       !-----------------------------------------------------

       ! ******* COMPUTING IQR STATISTICS ******
       !first get the means of the last 50 % of the cyclical buffer
       if (I <= (L/2)) then
          last_half_mask_l = (vec_index > I) .and. (vec_index < I + L/2)
       else
          last_half_mask_l = (vec_index > I) .or. (vec_index < I - L/2)
       end if

       !		logmask_nl = spread(last_half_mask_l, 1 ,N) .and. (all_population_posteriors_nl > 0.0_wpfmpi)
       ! Since I am already returning the log probability, all will be < 0, so remove this > 0 check
       logmask_nl = spread(last_half_mask_l, 1 ,N)

       ! Note, again that since we already use log probability we want to use 0, not 1 if not in mask.

       populations_logmean_n = sum( (merge(all_population_posteriors_nl , 0.0_wpfmpi , logmask_nl)) , 2) / &
            sum(merge(1.0_wpfmpi , 0.0_wpfmpi , logmask_nl) , 2)

       !write(*,*) "population log means"
       !call PrintVec(populations_logmean_n)

       !sorting to get quantiles
       !write(*,*) "[mpi_mcmc_mod] computing IQR stats"
       call qsortd(populations_logmean_n , index_logmean_n , N)
       Q3 = populations_logmean_n(index_logmean_n( (3*N) / 4))
       Q1 = populations_logmean_n(index_logmean_n(N/4))
       IQR = Q3 - Q1

       call qsortd(one_population_posteriors_n , index_logmean_n , N)
       best_chain = index_logmean_n(N)

       !REPLACING OULTLINER WITH BEST GUESS
       do C = 1,N
          if (populations_logmean_n(C) < Q1 - 2.0_wpfmpi * IQR) then
             all_outliner_results_nl(C,I) = .true.
             all_population_knl(:,C,:) = all_population_knl(:,best_chain,:)
             all_population_posteriors_nl(C,:) = all_population_posteriors_nl(best_chain,:)
          end if
       end do

       !PRINT SOME STATISTICS
       if (mod(loop_count,op%stat_display_rate) .eq. 0) then
          write(*,*) " ---------- LOOP " ,  loop_count , "  ---------------"
          solution_theta = sum(sum(all_population_knl, 3),2)/real(N*L,wpfmpi)
          write(*,*) "CURRENT MEAN ESTIMATE"
          call printvec(solution_theta)

          est_max_loc = maxloc(all_population_posteriors_nl)
          write(*,*) "CURRENT MAX ESTIMATE: " , maxval(all_population_posteriors_nl)
          call printvec(all_population_knl(:,est_max_loc(1) , est_max_loc(2)))

          write(*,*) "last posteriors"
          call PrintVec(candidate_population_posteriors_n)

          write(*,*) "acceptance probability"
          call printVec(alpha)

          write(*,*) "acceptance outcome"
          call printvec(real(all_acceptance_results_nl(:,I),wpfmpi))

          write(*,*) "acceptance: strong accept rate: " , sum(merge(1.0_wpfmpi,0.0_wpfmpi,all_acceptance_results_nl == 3))/(N*L)
          write(*,*) "acceptance: equal  accept rate: " , sum(merge(1.0_wpfmpi,0.0_wpfmpi,all_acceptance_results_nl == 2))/(N*L)
          write(*,*) "acceptance: weak   accept rate: " , sum(merge(1.0_wpfmpi,0.0_wpfmpi,all_acceptance_results_nl == 1))/(N*L)
          write(*,*) "acceptance: weak   reject rate: " , sum(merge(1.0_wpfmpi,0.0_wpfmpi,all_acceptance_results_nl == 0))/(N*L)
          write(*,*) "not in prior reject rate      : " , sum(merge(1.0_wpfmpi,0.0_wpfmpi,all_acceptance_results_nl == -1))/(N*L)
          write(*,*) "outliers                      : " , sum(merge(1.0_wpfmpi,0.0_wpfmpi,all_outliner_results_nl(:,I)))
          write(*,*) "outliers rate                 : " , sum(merge(1.0_wpfmpi,0.0_wpfmpi,all_outliner_results_nl))/(N*L)

          !-----------------------------------------------------
          !			     DUMPING CHAINS TO FILE
          !-----------------------------------------------------
          if (op%dump_chain) then
             do C=1,K
                write(Lfilename,'("chain_param_" , i5 , "_", i5,".dat")'), 10000+C , 10000+buffer_loop
                write(*,*) "saving to ", Lfilename
                call writetocsv(Lfilename , all_population_knl(C,:,:))
             end do
             write(Lfilename,'("chain_post_" , i5,".dat")') , 10000+buffer_loop
             call writetocsv(Lfilename,all_population_posteriors_nl)
             buffer_loop=buffer_loop+1
          end if
       end if


       !-----------------------------------------------------
       !                 CEHCKING CONVERGENCE
       !-----------------------------------------------------
       ! TO BE DONE!!!

    END DO


    !SAVING THE CHAINS
    write(*,*) "Saving the chains and posteriors"
    ch%thetas_knl = all_population_knl
    ch%posts_nl = all_population_posteriors_nl

    solution_theta = sum(sum(all_population_knl, 3),2)/(N*L)
  end subroutine StartOptimizerMaster


  ! generates a random list of couple of random integers between 1 and tot.
  ! for each row, the two integers are different from each
  ! other and diffrent from avoid
  subroutine pickFirstDifferences(tot , avoid, vec1 , vec2)
    implicit none
    integer, intent(IN):: tot, avoid
    integer,intent(INOUT) :: vec1(:)
    integer,intent(INOUT) :: vec2(:)
    integer :: lcount,length
    real(wpfmpi) :: random_draw(2)
    integer :: val1, val2

    lcount = 1
    length = size(vec1)

    do while ( lcount <= length)
       call randvec(random_draw)
       val1  = floor(random_draw(1) * real(tot,wpfmpi)) + 1
       val2  = floor(random_draw(2) * real(tot,wpfmpi)) + 1

       if ((val1 .eq. val2) .or. (val1 .eq. avoid ) .or. (val2 .eq. avoid)) then
          cycle
       end if

       vec1(lcount) = val1
       vec2(lcount) = val2
       lcount = lcount +1
    end do

  end subroutine pickFirstDifferences


  subroutine ComputeParamCandidate(current_population_kn,param_candidate, C ,op,jump)
    implicit none
    type(MCMCParams), intent(IN) :: op
    integer :: K,L,N,P, J
    integer, intent(IN) :: C
    logical, allocatable :: dimension_mask(:)   ! randomly selecting the number of dimensions to update
    real(wpfmpi), allocatable :: rand_dimension_mask(:)   ! randomly selecting the number of dimensions to update
    real(wpfmpi), allocatable :: rand_epsilon(:)   !
    real(wpfmpi), allocatable :: rand_e(:)   !
    integer :: updated_dimension_count          ! randomly selecting the number of dimensions to update
    integer, allocatable :: index1(:) , index2(:)   ! index for the differences
    logical, intent(IN) :: jump

    real(wpfmpi),intent(INOUT) :: param_candidate(:)
    real(wpfmpi),intent(IN) :: current_population_kn(:,:)
    real(wpfmpi) :: gamma

    K = op%parameter_size
    L = op%chain_length
    N = op%chain_count
    P = nprocs

    allocate(dimension_mask(K))
    allocate(rand_dimension_mask(K))
    allocate(rand_epsilon(K))
    allocate(rand_e(K))

    allocate (index1(op%delta))
    allocate (index2(op%delta))
    ! COMPUTE THE NUMBER OF DIMENSIONS TO UPDATE
    call randvec(rand_dimension_mask)
    dimension_mask = merge(.true. , .false. , rand_dimension_mask < op%CR)
    updated_dimension_count = sum(merge(1 , 0 ,dimension_mask))

    !make sure we update at least one dimension!
    if (updated_dimension_count == 0) then
       dimension_mask(int(rand_dimension_mask(1) * K)+1) = .true.
       updated_dimension_count = 1
    end if

    call randnvec(rand_epsilon)
    call randvec(rand_e)

    !GENERATE A CANDIDATE
    if (jump) then
       ! allow jumps between modes
       gamma = 1.0_wpfmpi
    else
       gamma = 2.38_wpfmpi / sqrt(2.0_wpfmpi * real(op%delta * updated_dimension_count,wpfmpi))
    end if
    ! get random differences - TODO: change to random without replacement
    call pickFirstDifferences(N , C, index1 , index2)
    param_candidate = 0.0_wpfmpi
    ! the line below turns off differences between chains.
    ! gamma = 0.0_wpfmpi  !! Do not use differences between chains.

    do J = 1, op%delta
       param_candidate = param_candidate + current_population_kn(:,index1(J)) - current_population_kn(:,index2(J))
    end do
    param_candidate = current_population_kn(:,C) + (1.0_wpfmpi + op%shock_mult_std*rand_e) * gamma * param_candidate &
         + rand_epsilon*op%shock_add_std
    param_candidate = merge(param_candidate , current_population_kn(:,C) , dimension_mask)
    !TODO: the additive shock should be scaled to the width of the prior

    deallocate(dimension_mask)
    deallocate(rand_dimension_mask)
    deallocate(rand_epsilon)
    deallocate(rand_e)
    deallocate (index1)
    deallocate (index2)

  end subroutine ComputeParamCandidate

  subroutine MasterReleaseSlaves(op)
    implicit none
    type(MCMCParams), intent(IN) :: op
    real :: vals
    real,allocatable :: theta_tmp(:)
    integer :: ntasks,itask
    integer :: ierror

    ntasks = nprocs - 1
    allocate( theta_tmp(op%parameter_size))
    theta_tmp = 0.0

    do itask=1,ntasks
       call MPI_Send(theta_tmp, op%parameter_size, MPI_REAL, itask, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
    end do

  end subroutine MasterReleaseSlaves

  !  subroutine mirrorParamBackInPrior(param , op)

  !  end subroutine

  subroutine MpiMCMFinalize
    call MPI_FINALIZE(ierror)
  end subroutine MpiMCMFinalize

end module mpi_mcmc_mod
