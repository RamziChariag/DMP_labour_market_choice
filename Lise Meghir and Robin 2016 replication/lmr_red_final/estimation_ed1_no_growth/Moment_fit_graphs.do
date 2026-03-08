clear 

// set graphic scheme
set scheme sj /* A plain monochrome scheme for papers */

// Read in moments from Fortran output
infix model 1-18 data 19-37 se 38-56 tstat 57-75 using "Moment_Fit.dat"

// create indicator for moment number (20 years for each moment)
generate moment = ceil(_n / 20)

// Moments 
// 1 E_t    
// 2 U2E_t
// 3 E2U_t
// 4 J2J_t
// 5 w_t
// 6 Dw_t
// 7 DwEE_t
// 8 Dw_DJ_t
// 9 var w_t
// 10 var Dw_t
// 11 var DwEE_t
// 12 var DwDJ_t

// create year within moment
bys moment : generate year = _n

// create standard error bands
drop tstat
generate data_low = data - 2*se
generate data_high = data + 2*se

label variable year "Years in labour force"

// drop if year == 1

foreach m in 1 2 3 4 5 6 7 8 9 10 11 12 {
    twoway rarea data_low data_high year if moment == `m', ///
           bcolor(gs14) || ///
           line data year if moment == `m', legend(off) || ///
	   line model year if moment == `m', lpattern(solid) lwidth(thick)  legend(off) 
    graph export ./figures/model_fit_moment_`m'.eps, fontface(Times) replace
  }
