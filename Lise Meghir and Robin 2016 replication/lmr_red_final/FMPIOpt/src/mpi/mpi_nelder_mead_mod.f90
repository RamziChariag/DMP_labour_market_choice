module mpi_nelder_mead_mod
  use glob
  use params
  use solver
  use output
  use Compute_Surplus_Mod
  use Simulate_Wages_Mod
  use compute_moments_mod
  include 'mpif.h'

	type NelderMeadParams
		integer :: P
		real(wp) :: alpha
		real(wp) :: gamma
		real(wp) :: beta
		real(wp) :: tau
		real(wp) :: epsilon_step
		real(wp) :: taul
	end type NelderMeadParams
  save

  integer :: procId !proc id on the mpi grid
  integer :: nprocs !number of procs
  integer :: theta_size

contains

  subroutine MpiNelderMeadInit(a_theta_size)
	implicit none
	integer :: ierror
	integer, intent(IN) :: a_theta_size

	theta_size = a_theta_size
    call MPI_INIT(ierror)
  	call MPI_COMM_RANK(MPI_COMM_WORLD, procId, ierror)
  	call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierror)
  end subroutine MpiNelderMeadInit

   ! This methods recieve a list of parameter values ni theta.
   ! For each of the values, it sends it to a processor, then wait
   ! for the results that are returned in the same order in vals
   ! IN    thetas(:,:)  (nparams x number_of_evaluations)
   ! INOUT vals(:)      (number_of_evaluations)
   !
   ! The idea here is to do blocking send/recieve everywhere
   ! but on the recieve side of the master
  subroutine Master_ComputeInParrallel(thetas , vals)
	implicit none
	real(wp), intent(IN) :: thetas(:,:)
	real(wp), intent(INOUT) :: vals(:)
	real :: tmp_val
	real,allocatable :: theta_tmp(:)

	integer,allocatable :: requests(:)
	integer :: ntasks,itask
	integer :: left_to_recieve
	integer :: ierror

	ntasks = size(thetas,2)
	allocate( requests(ntasks))
	allocate( theta_tmp(theta_size))

	!SENDING THE REQUESTS
	!eventually we should do this in a smarter way where we
	!we queue requests
	do itask=1,ntasks
		write(*,*) "sending theta to " , itask
		theta_tmp = real(thetas(:,itask))
		call printVec(real(theta_tmp,wp))
		call MPI_Send(theta_tmp, theta_size, MPI_REAL, itask, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
		write(*,*) "done sending"
	end do
    left_to_recieve = ntasks

	!FOR NOW BLOKING RECIEVE
	do itask=1,ntasks
	    write(*,*) "Master ready to receive value from : " , itask
		call MPI_Recv(tmp_val, 1, MPI_REAL, itask, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
		vals(itask) = real(tmp_val,wp)
		write(*,*) "[mpi_nelder_mead_mod] Master got value ", tmp_val , " from:" , itask
	end do

	!DONE - CLEANING
	deallocate(requests)

  end subroutine Master_ComputeInParrallel

  subroutine Slaves_ComputeValues_Loop
    use objective_function_mod
    implicit none
    real,allocatable :: theta(:)
	real :: value
	type (ExogenousParameters) :: p
	integer :: ierror

    ! INIT PARAMETERS
    call Init(p);
    allocate (theta(theta_size))

    !ADJUST THE DEFAULT TOLERANCE
    p%tol_h = 1E-6_wp
    p%tol_S = 1E-9_wp
    p%tol_u = 1E-7_wp
    p%tol_v = 1E-7_wp
    p%tol_W = 1E-5_wp

	do while (.TRUE.)
	  write(*,*) procid , ": ready to receive"
      call MPI_Recv(theta, theta_size, MPI_REAL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
      write(*,*) procid , ": Recieved a parameter"

      if (sum(abs(theta)) .eq. 0.0) then
         write(*,*) procid, " received null param, so exits"
         return
      end if

	  ! CALLING THE OBJECTIVE FUNCTION
      value = -1.0 * real(objective_function(real(theta,wp),p))

      write(*,*) "[mpi_nelder_mead_mod]" , procid , "value: " ,  value
	  call MPI_Send(value , 1, MPI_REAL, 0 , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
	  write(*,*) procid , ": value sent back"
	end do
  end subroutine Slaves_ComputeValues_Loop


!##########################################################################
! 				OPTIMIZATION PROCEDURE
!##########################################################################

  subroutine StartOptimizer(initial_theta ,solution_theta, op)
    implicit none
    type(NelderMeadParams), intent(IN) :: op
    real(wp),intent(IN) :: initial_theta(:)
    real(wp),intent(INOUT) :: solution_theta(:)
    real(wp),allocatable :: thetas(:,:)
	real(wp),allocatable :: new_simplex_vals(:), old_simplex_vals(:) , ranked_simplex_vals(:)
    real(wp),allocatable :: old_simplex(:,:), new_simplex(:,:), ranked_simplex(:,:)
    real(wp),allocatable :: candidates(:,:) , candidates_vals(:)
    integer, allocatable :: rank(:)
    real(wp),allocatable :: centroid(:)
    integer :: J,P
	integer :: i,k
	logical :: improvement, limprovement

	J = theta_size
	P = op%P

	allocate( old_simplex_vals(J + 1))
	allocate( old_simplex(J ,J + 1))
	allocate( new_simplex_vals(J + 1))
	allocate( new_simplex(J ,J + 1))
	allocate( ranked_simplex_vals(J + 1))
	allocate( ranked_simplex(J ,J + 1))
	allocate( rank(J+1) )
	allocate( centroid(J+1) )
	allocate( candidates(J , 3*P))
	allocate( candidates_vals(3*P))

    !CREATE THE CURRENT SIMPLEX
	old_simplex = spread(initial_theta , 2, J + 1)
    do i=1,theta_size
		old_simplex(i,i+1) = old_simplex(i,i+1) + op%epsilon_step
	end do

	!EVALUATE AT EACH POINT OF THE SIMPLEX
	call log4mat_debug_real("mpi_nelder_mead_mod", 1, "first evaluation of the simplex", 0.0_wp)
	call Master_ComputeInParrallel(old_simplex, old_simplex_vals)

    !-----------------------------------------------------
    !			PREPARE THE POINTS TO EVALUATE
    !-----------------------------------------------------
	!REORDER SIMPLEX BASED ON VALUES
	call computeRank(old_simplex_vals,rank)
	do i=1,J+1
	  	ranked_simplex(rank(i),:) = old_simplex(:,i)
	  	ranked_simplex_vals(rank(i)) = old_simplex_vals(i)
	end do

    !COMPUTE CENTROID SELECTING ONLY BESTS
    centroid = 1.0_wp/real(P,wp) * sum( ranked_simplex(:,1:P) , 2)

	!COMPUTE REFLEXIONS
	candidates(:,1:P)         = (1+op%alpha) * spread(centroid,2,P) + (1-op%alpha) * ranked_simplex(:,1:P)
	!COMPUTE EXPANSIONS
	candidates(:,P+1:2*P)	  = (1+op%gamma) * spread(centroid,2,P) + (1-op%gamma) * ranked_simplex(:,1:P)
	!COMPUTE CONTRACTIONS
	candidates(:,2*P+1:3*P)   = (1+op%beta) * spread(centroid,2,P) + (1-op%beta) * ranked_simplex(:,1:P)

    !-----------------------------------------------------
    !			   EVALUATES THE POINTS
    !-----------------------------------------------------
 	call log4mat_debug_real("mpi_nelder_mead_mod", 1, "evaluation of the candidates", 0.0_wp)
 	call Master_ComputeInParrallel(candidates, candidates_vals)

    !-----------------------------------------------------
    !		  SELECTS NEW POINTS FOR SIMPLEX
    !-----------------------------------------------------
	improvement = .FALSE.
	do k=P,1
		call pickPoint( ranked_simplex_vals(J+1) , ranked_simplex_vals(k+1) , &
						candidates(:, k      ) , candidates_vals(k      ), &
						candidates(:, P + k  ) , candidates_vals(P+k    ), &
						candidates(:, 2*P + k) , candidates_vals(2*P + k), &
						new_simplex(:,k)		, new_simplex_vals(k), &
						limprovement)
		improvement = improvement .OR. limprovement
	end do

	!IF NOT IMPROVEMENT, CONTRACT SIMPLEX
	if (improvement .eq. .FALSE.) then
		new_simplex = op%tau * spread(ranked_simplex(:,1), 2, J+1) + ( 1 - op%tau) * ranked_simplex
	end if

	!DONE, LET's

	solution_theta = initial_theta
  end subroutine StartOptimizer


  subroutine pickPoint(A0 , AP , reflexion_vec , reflexion_val , &
  								 expansion_vec , expansion_val , &
  								 contraction_vec, contraction_val, &
  								 best_vec , best_val, improvement)
  	implicit none
	real(wp), intent(IN) :: A0 , AP
	real(wp), intent(IN) :: reflexion_vec(:), reflexion_val
	real(wp), intent(IN) :: expansion_vec(:), expansion_val
	real(wp), intent(IN) :: contraction_vec(:), contraction_val
	real(wp), intent(INOUT) :: best_vec(:), best_val
	logical,intent(INOUT) :: improvement


	improvement  = .FALSE.
	!IF REFLECTION IS AN IMPROVEMENT ON BEST
	if ( reflexion_val > A0) then
		improvement  = .TRUE.
		if ( expansion_val > A0 ) then
			best_vec = expansion_vec
			best_val = expansion_val
		else
			best_vec = reflexion_vec
			best_val = reflexion_val
		end if
	elseif ( reflexion_val > AP) then
		improvement  = .TRUE.
		best_vec = reflexion_vec
		best_val = reflexion_val
	else
		if (reflexion_val > expansion_val) then
			best_vec = reflexion_vec
			best_val = reflexion_val
		else
			best_vec = expansion_vec
			best_val = expansion_val
		end if

		if (contraction_val > best_val) then
			improvement  = .TRUE.
			best_vec = contraction_vec
			best_val = contraction_val
		end if
	end if
  end subroutine


  subroutine MasterReleaseSlaves
  	implicit none
	real :: vals
	real,allocatable :: theta_tmp(:)
	integer :: ntasks,itask
	integer :: ierror

	ntasks = nprocs - 1
	allocate( theta_tmp(theta_size))
	theta_tmp = 0.0

	do itask=1,ntasks
		call MPI_Send(theta_tmp, theta_size, MPI_REAL, itask, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
	end do

  end subroutine


  subroutine FinalizeMPI
	  call MPI_FINALIZE(ierror)
  end subroutine FinalizeMPI

end module mpi_nelder_mead_mod
