// Read in, rename and transform data for weekly panel file for Lise & Postel-Vinay
// Date 12 Sepat 2014


clear 
set more off
set maxvar 32000

// Read in raw NLSY79 data (this only needs to be 
infile using "./lpv_weekly_panel/lpv_weekly_panel.dct"

// label values and rename variables using question and survey year
do "./lpv_weekly_panel/lpv_weekly_panel-value-labels"

do lpv_weekly_panel_rename

compress
save nlsy79_raw, replace

aorder
save nlsy79_raw, replace

