module glob
!  use mkl95_precision, only : wp => dp
    implicit none
    save
!    !integer, parameter :: wp = selected_real_kind(8) !Statement to control the precision of real variables in the program.
    INTEGER, PARAMETER :: wp = kind(0.0D0) ! double precision on any machine
    
    ! random numbers for simulation
    real(wp),allocatable :: global_rand_x_r(:)
    real(wp),allocatable :: global_rand_delta_rt(:,:) 
    real(wp),allocatable :: global_rand_zeta_rt(:,:) 
    real(wp),allocatable :: global_rand_y_rt(:,:)
    real(wp),allocatable :: global_rand_lambda_rt(:,:)

    ! random numbers for measurement error in wage moments
    real(wp),allocatable :: global_merror(:,:)
    

end module glob
