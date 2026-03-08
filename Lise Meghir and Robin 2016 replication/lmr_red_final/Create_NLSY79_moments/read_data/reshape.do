clear 
set more off
set maxvar 32000

log using reshape_errors, replace
use id STATUS_WK* HRS_WORKED_WK* JOB_WK_NUM* using nlsy79_raw

reshape long STATUS_WK_NUM@_XRND HRS_WORKED_WK_NUM@_XRND JOB_WK_NUM@_DUALJOB_NUM1_XRND JOB_WK_NUM@_DUALJOB_NUM2_XRND JOB_WK_NUM@_DUALJOB_NUM3_XRND JOB_WK_NUM@_DUALJOB_NUM4_XRND JOB_WK_NUM@_DUALJOB_NUM5_XRND, i(id) j(week)

save nlsy79_long, replace
log close



