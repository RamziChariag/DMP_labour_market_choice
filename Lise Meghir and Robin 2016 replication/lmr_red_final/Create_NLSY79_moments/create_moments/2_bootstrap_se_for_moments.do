clear
set more off
* SELECTION 1: INITIAL CROSS-SECTION

* load NLSY cross-section panel and select relevant population
use lpv_data_cross_section.dta

* drop people in over-sample
keep if SAMPLE_ID<=2

* keep white males not in the military
keep if sex == 1
keep if race == 3

* keep initial list of relevant ids and merge with panel
keep id highest_grade
merge 1:m id using lpv_data_weekly_panel.dta
keep if _merge==3
gen cal_year 	= year
gen cal_month 	= month

* SELECTION 2: PANEL

* drop people seen in the military
drop if ever_military>0

* drop the one week from 1977
drop if cal_year == 1977

rename E employed

* select starting year
drop if cal_year<=finished_ed

* reset starting week, month and year to 1 for everyone
tempvar minweek minyear
egen `minweek' = min(week), by(id)
egen `minyear' = min(year), by(id)
replace week = week - `minweek' + 1
replace year = year - `minyear' + 1
replace month = (year-1)*12 + cal_month

* iron out short nonemployment spells
gen transition = (E2U==1) + (J2J==1) +  (U2E==1)
bys id (week): gen spell = sum(transition)
egen u_dur = count(week) if employed==0, by(id spell)
local ee_cutoff = 2
replace employed = 1 if u_dur<=`ee_cutoff'
bys id (week): replace J2J = 1 if u_dur==. & u_dur[_n-1]<=`ee_cutoff'
replace J2J = 0 if employed==1 & J2J==.
bys id (week): replace U2E = . if u_dur==. & u_dur[_n-1]<=`ee_cutoff'
replace U2E=. if u_dur<=`ee_cutoff'
bys id (week): replace E2U = 0 if u_dur==. & u_dur[_n-1]<=`ee_cutoff'
replace E2U = 0 if u_dur<=`ee_cutoff'
tempvar transition
replace transition = (E2U==1) + (J2J==1) +  (U2E==1)
tab transition
bys id (week): replace spell = sum(transition)
replace wage = . if employed==. | employed==0

* tidy up and save clean panel
drop _merge sex race SAMPLE_ID ever_military *_v* finished_ed u_dur transition
cap drop __00*

* WARNING: DO NOT divide wages by hours - the data already has hourly wages
* consgruct weekly earnings
replace wage = wage*hours
* trim wages
su wage, det
replace wage = . if wage>r(p99) | wage<r(p1)

gen log_w = log(wage)

* tidy up and save
drop if month>363
drop if month<=3
cap drop __00*

gen educ = 0
*replace educ = 1 if highest_grade  < 12 
*replace educ = 2 if highest_grade == 12 
*replace educ = 3 if highest_grade > 12 &  highest_grade < 16
*replace educ = 4 if highest_grade >= 16 

replace educ = 1 if highest_grade <  16 
replace educ = 2 if highest_grade >= 16 

save clean_panel_no_growth, replace

*remove aggregate growth based on the growth in after 10 years in the labor force
*this is effectively assuming it takes 10 years for a cohort to reach stationarity
regress log_w week if (year >10 & year <= 20 & educ == 1)
replace log_w = log_w - _b[week]*week if educ == 1
regress log_w week if (year >10 & year <= 20 & educ == 2)
replace log_w = log_w - _b[week]*week if educ == 2

replace wage = exp(log_w)

save clean_panel_no_growth, replace

foreach replication of numlist 1/100 {
set more off
di `replication'

clear

* set up matrix to hold moments

		
	forval ed = 1/2{
	matrix mom_`ed' = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

	matrix se_`ed'  = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0\  ///                        0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\ 0\  ///
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
	}


use clean_panel_no_growth
bsample, cluster(id)
save bs_sample, replace

su week
local minweek=r(min)


/*
* selection: drop people who are unemployed more than x percent of the time
egen m_u = mean(1-employed), by(id)
su m_u if week==`minweek', det
drop if m_u>.25
*/

bys id spell: gen spdur = _N*12/52*(employed==0)
gen sel = (spdur>=36)

bys id (week): replace sel = sum(sel)
drop if sel>0
count if week==`minweek'

* MOMENTS: sample size, unemployment and transition rates
* ... aggregate cross-sections
* preserve
collapse (mean) J2J U2E E2U employed month year (count) ind_cnt = id, by(week educ) fast
* ... then collapse to monthly
collapse (sum) J2J U2E E2U (mean) employed ind_cnt year (count) weeks_this_month = week, by(month educ) fast
foreach var of varlist E2U U2E J2J	{
	replace `var' = `var'*4/weeks_this_month
	* rename `var' m_`var'_mom
}

collapse (mean) J2J U2E E2U employed  ind_cnt, by(year educ)fast 
* we only want the first 20 years of data.
keep if year <= 20

gen u_rate = 1-employed
sort year
quietly tab year, gen(y)

	foreach ed of numlist 1/2 {
		// employment
		quietly regress employed y1-y20 if educ == `ed', noc
		foreach yyy of numlist 1/20 {
			matrix mom_`ed'[1,`yyy'] = _b[y`yyy']				
			*matrix  se_`ed'[1,`yyy'] = _se[y`yyy']				
	 	}
		// U2E
		quietly regress U2E y1-y20 if educ == `ed', noc
		foreach yyy of numlist 1/20 {
			matrix mom_`ed'[2,`yyy'] = _b[y`yyy']				
			*matrix  se_`ed'[2,`yyy'] = _se[y`yyy']				
	 	}
	 	// E2U
		quietly regress E2U y1-y20 if educ == `ed', noc
		foreach yyy of numlist 1/20 {
			matrix mom_`ed'[3,`yyy'] = _b[y`yyy']				
			*matrix  se_`ed'[3,`yyy'] = _se[y`yyy']				
	 	}
	 	// J2J
		quietly regress J2J y1-y20 if educ == `ed', noc
		foreach yyy of numlist 1/20 {
			matrix mom_`ed'[4,`yyy'] = _b[y`yyy']				
			*matrix  se_`ed'[4,`yyy'] = _se[y`yyy']				
	 	}
	}

*Wage moments

drop _all
use bs_sample

su week
local minweek=r(min)

bys id spell: gen spdur = _N*12/52*(employed==0)
gen sel = (spdur>=36)
bys id (week): replace sel = sum(sel)
drop if sel>0

collapse (sum) wage employed (mean) educ, by(id year)
generate log_w = log( wage / employed )  
collapse (mean) log_w (sd) var_log_w = log_w, by(year educ)
replace var_log_w = var_log_w^2

sort year
quietly tab year, gen(y)

foreach ed of numlist 1/2 {
	// Mean log wage
	quietly regress log_w y1-y20 if educ == `ed', noc
	foreach yyy of numlist 1/20 {
		matrix mom_`ed'[5,`yyy'] = _b[y`yyy']				
		*matrix  se_`ed'[5,`yyy'] = _se[y`yyy']				
 	}
	// variance of log wage
	quietly regress var_log_w y1-y20 if educ == `ed', noc
	foreach yyy of numlist 1/20 {
		matrix mom_`ed'[9,`yyy'] = _b[y`yyy']				
		*matrix  se_`ed'[9,`yyy'] = _se[y`yyy']				
	}
}


*Wage growth moments

* year over year (regardless of number of weeks employed)
drop _all
use bs_sample

su week
local minweek=r(min)

bys id spell: gen spdur = _N*12/52*(employed==0)
gen sel = (spdur>=36)
bys id (week): replace sel = sum(sel)
drop if sel>0

collapse (sum) wage employed (mean) educ, by(id year)
generate log_w = log( wage / employed )  

sort id year
xtset id year
gen D_log_w = F.log_w - log_w
gen sel =  ( D_log_w > log(0.5) & D_log_w < log(5) ) 
keep if sel == 1

collapse (mean) D_log_w (sd) var_D_log_w = D_log_w, by(year educ)
replace var_D_log_w = var_D_log_w^2

bys educ:su if year <= 20
sort year
quietly tab year, gen(y)

foreach ed of numlist 1/2 {
	// Mean Wage growth
	quietly regress D_log_w y1-y20 if educ == `ed', noc
	foreach yyy of numlist 1/20 {
		matrix mom_`ed'[6,`yyy'] = _b[y`yyy']				
		*matrix  se_`ed'[6,`yyy'] = _se[y`yyy']				
 	}
	//  Variance of Wage growth
	quietly regress var_D_log_w y1-y20 if educ == `ed', noc
	foreach yyy of numlist 1/20 {
		matrix mom_`ed'[10,`yyy'] = _b[y`yyy']				
		*matrix  se_`ed'[10,`yyy'] = _se[y`yyy']				
 	}
}


drop _all
use bs_sample

su week
local minweek=r(min)

bys id spell: gen spdur = _N*12/52*(employed==0)
gen sel = (spdur>=36)
bys id (week): replace sel = sum(sel)
drop if sel>0

collapse (sum) wage employed (mean) J2J educ, by(id year)
generate log_w = log( wage / employed )  

sort id year
xtset id year
gen D_log_w = F.log_w - log_w
* in the first year 
gen sel = ( (( F.employed == 52 & employed == 52 & F.J2J == 0 & J2J == 0 ) | ( year == 1 & (F.employed == 52 & employed > 26 & F.J2J == 0 & J2J == 0) ) ) & ( D_log_w > log(0.5) & D_log_w < log(5) ) )
keep if sel == 1

collapse (mean) D_log_w (sd) var_D_log_w = D_log_w, by(year educ)
replace var_D_log_w = var_D_log_w^2


bys educ:su if year <= 20
sort year
quietly tab year, gen(y)
foreach ed of numlist 1/2 {
	// Mean Wage growth conditional on continuous employment for 2 years at same job
	quietly regress D_log_w y1-y20 if educ == `ed', noc
	foreach yyy of numlist 1/20 {
		matrix mom_`ed'[7,`yyy'] = _b[y`yyy']				
		*matrix  se_`ed'[7,`yyy'] = _se[y`yyy']				
 	}
	// Variance of Wage growth conditional on continuous employment for 2 years at same job
	quietly regress var_D_log_w y1-y20 if educ == `ed', noc
	foreach yyy of numlist 1/20 {
		matrix mom_`ed'[11,`yyy'] = _b[y`yyy']				
		*matrix  se_`ed'[11,`yyy'] = _se[y`yyy']				
 	}
}


drop _all
use bs_sample

su week
local minweek=r(min)

bys id spell: gen spdur = _N*12/52*(employed==0)
gen sel = (spdur>=36)
bys id (week): replace sel = sum(sel)
drop if sel>0

collapse (sum) wage employed (mean) J2J educ, by(id year)
generate log_w = log( wage / employed )  

sort id year
xtset id year
gen D_log_w = F.log_w - log_w
gen sel = ( (( F.employed == 52 & employed == 52 & (F.J2J > 0 | J2J > 0) ) | ( year == 1 & (F.employed == 52 & employed > 26 & (F.J2J > 0 | J2J > 0) ) ) ) & ( D_log_w > log(0.5) & D_log_w < log(5) ) )
keep if sel == 1

collapse (mean) D_log_w (sd) var_D_log_w = D_log_w, by(year educ)
replace var_D_log_w = var_D_log_w^2

bys educ:su if year <= 20
sort year
quietly tab year, gen(y)
foreach ed of numlist 1/2 {
	// Mean Wage growth conditional on continuous employment for 2 years and job change
	quietly regress D_log_w y1-y20 if educ == `ed', noc
	foreach yyy of numlist 1/20 {
		matrix mom_`ed'[8,`yyy'] = _b[y`yyy']				
		*matrix  se_`ed'[8,`yyy'] = _se[y`yyy']				
 	}
	// Variance of Wage growth conditional on continuous employment for 2 years and job change
	quietly regress var_D_log_w y1-y20 if educ == `ed', noc
	foreach yyy of numlist 1/20 {
		matrix mom_`ed'[12,`yyy'] = _b[y`yyy']				
		*matrix  se_`ed'[12,`yyy'] = _se[y`yyy']				
 	}
}

foreach ed of numlist 1/2 {
	matrix mom_`ed'_ = mom_`ed''
	*matrix se_`ed'_ = se_`ed''
}

	drop _all
	
	generate year = _n

  	foreach ed of numlist 1/2 {
		svmat double mom_`ed'_
		*svmat double se_`ed'_
	//	outsheet mom_`ed'_* using "data/mom_ed`ed'.raw", delimiter(" ") nonames replace
	//	outsheet se_`ed'_* using "data/se_ed`ed'.raw", delimiter(" ") nonames replace
		
  	}

  	foreach ed of numlist 1/2 {
		foreach mom of numlist 1/12 {
			egen se_`ed'_`mom' = mean(mom_`ed'_`mom')
		}
	}
	
	// format properly for reading into my fortran programme
	foreach ed of numlist 1/2 {
		file open myfile_`ed' using "moments/bs_mom_ed`ed'_`replication'.raw", write replace
		foreach moment of numlist 1/12 {
			foreach year of numlist 1/20 {
				file write myfile_`ed' %24.16f (mom_`ed'_`moment'[`year']) %3s " "
			}
		}
		file write myfile_`ed' %24.16f (0.54)  // 0.54 is V/U ratio from Hall (2005)
		file close myfile_`ed'
	}
	
	// the addition of 0.54 and 0.054		 are just place holders for the mean and standard deviaiton of the vacancy unemployment ratio


}



