module data_statistics_mod
  use glob
  use optimization_mod
  use Compute_Moments_Mod
  
  real(wp),allocatable :: GLOBAL_TRUE_MOMENT_MATRIX(:)
  real(wp),allocatable :: GLOBAL_TRUE_SD_MATRIX(:)
  real(wp),allocatable :: GLOBAL_TRUE_N_MATRIX(:)
  real(wp),allocatable :: GLOBAL_TRUE_COV_MATRIX(:,:)
  real(wp),allocatable :: GLOBAL_TRUE_CHOLESKY_MATRIX(:,:)
  
  logical :: GLOBAL_DISPLAY_SWITCH
  
  
end module data_statistics_mod
