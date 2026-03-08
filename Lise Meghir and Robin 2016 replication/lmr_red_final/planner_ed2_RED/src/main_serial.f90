program Msw
  use params
  use output
  use glob
  use objective_function_mod
  use optimization_mod
  use data_statistics_mod

  ! use planner_objective
  ! use tax_objective
  ! use minw_objective
  ! use sev_objective

  implicit none
  real(wp) :: theta(nparams) , param_start(nparams), best_theta(nparams) ,value
  type (ExogenousParameters) :: p
  logical :: base_switch 
  integer i
  
  ! INIT PARAMETERS
  call Init(p)
  ! turn ON printing and saving of simulated moments etc
  GLOBAL_DISPLAY_SWITCH = .TRUE.

  base_switch = .TRUE.
  
  ! read in starting values for parameters from disc
  call readFromCSVvec('data/planner_starting_val.raw', param_start )
  write(*,*) "Starting Values: "
  call printVec(param_start)

  theta = param_start
 
if (base_switch) then

  write(*,*) "Planner Version"
  write(*,*) "starting the program"
  
 write(*,*) "calling objective function"
 value = real(objective_function(real(theta,wp),p))
  
 write(*,*) "Value= " , value


end if

write(*,*) "Finished"

end program Msw




