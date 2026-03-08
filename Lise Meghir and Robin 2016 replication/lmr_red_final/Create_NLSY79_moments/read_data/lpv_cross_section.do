// Read in, rename and transform data for cross-sectional file for Lise & Postel-Vinay
// date written      11 Sept 2014
// date edited last  12 Sept 2014

clear 
set more off

// Read in raw NLSY79 data
infile using "./lpv_cross_section/lpv_cross_section.dct"

// label values and rename variables using question and survey year
do "./lpv_cross_section/lpv_cross_section-value-labels"


/* Recode Missing Values to Stata Missing Values */
	
	/* 	Noninterview     -5   .e */
	/* 	Valid Skip       -4   .d */
	/* 	Invalid Skip     -3   .c */
	/* 	Don't Know       -2   .b */
	/* 	Refusal          -1   .a */
	
	mvdecode _all , mv(-1 = .a)
	mvdecode _all , mv(-2 = .b)
	mvdecode _all , mv(-3 = .c)
	mvdecode _all , mv(-4 = .d)
	mvdecode _all , mv(-5 = .e)

// Rename using cross-walk from NLSY79 label file
  rename R0000100 CASEID_1979 
  rename R0021600 Q3_23_1_1979   // Q3-23_1
  rename R0173600 SAMPLE_ID_1979 
  rename R0214700 SAMPLE_RACE_1979 
  rename R0214800 SAMPLE_SEX_1979 
  rename R0216700 HGC_1979 
  rename R0230900 Q3_23_1_1980   // Q3-23_1
  rename R0406400 HGC_1980 
  rename R0419100 Q3_23_1_1981   // Q3-23_1
  rename R0614700 SAMPWEIGHT_ASVAB_1981 
  rename R0616000 ASVAB_13_1981   // ASVAB-13
  rename R0616200 ASVAB_15_1981   // ASVAB-15
  rename R0616400 ASVAB_17_1981   // ASVAB-17
  rename R0616600 ASVAB_19_1981   // ASVAB-19
  rename R0616800 ASVAB_21_1981   // ASVAB-21
  rename R0617000 ASVAB_23_1981   // ASVAB-23
  rename R0617200 ASVAB_25_1981   // ASVAB-25
  rename R0617400 ASVAB_27_1981   // ASVAB-27
  rename R0617600 ASVAB_29_1981   // ASVAB-29
  rename R0617800 ASVAB_31_1981   // ASVAB-31
  rename R0618200 AFQT_1_1981   // AFQT-1
  rename R0618300 AFQT_2_1981   // AFQT-2
  rename R0618301 AFQT_3_1981   // AFQT-3
  rename R0618900 HGC_1981 
  rename R0666200 Q3_23_1_1982   // Q3-23_1
  rename R0898200 HGC_1982 
  rename R0907500 Q3_23_1_1983   // Q3-23_1
  rename R1145000 HGC_1983 
  rename R1207800 Q3_23_1_1984   // Q3-23_1
  rename R1520200 HGC_1984 
  rename R1607000 Q3_23_1_1985   // Q3-23_1
  rename R1890900 HGC_1985 
  rename R1907400 Q3_23_01_1986   // Q3-23.01
  rename R2258000 HGC_1986 
  rename R2445400 HGC_1987 
  rename R2511300 Q3_23_1_1988   // Q3-23_1
  rename R2871100 HGC_1988 
  rename R2910200 Q3_23_1_1989   // Q3-23_1
  rename R3074800 HGC_1989 
  rename R3112200 Q3_23_1_1990   // Q3-23_1
  rename R3401500 HGC_1990 
  rename R3656900 HGC_1991 
  rename R3712700 Q3_23_1_1992   // Q3-23_1
  rename R4007400 HGC_1992 
  rename R4140300 Q3_23_01_1993   // Q3-23.01
  rename R4418500 HGC_1993 
  rename R4528700 Q3_23_1_1994   // Q3-23.1
  rename R5081500 HGC_1994 
  rename R5166800 HGC_1996 
  rename R5224000 Q3_23_01_1996   // Q3-23.01
  rename R6467800 Q3_23_CODE_01_1998   // Q3-23_CODE.01
  rename R6479400 HGC_1998 
  rename R6543300 Q3_23_CODE_01_2000   // Q3-23_CODE.01
  rename R7007100 HGC_2000 
  rename R7106500 Q3_23_CODE_01_2002   // Q3-23_CODE.01
  rename R7704400 HGC_2002 
  rename R8496800 HGC_2004 
  rename T0017300 Q3_23_CODE_01_2006   // Q3-23_CODE.01
  rename T0988600 HGC_2006 
  rename T1217700 Q3_23_CODE_01_2008   // Q3-23_CODE.01
  rename T2210600 HGC_2008 
  rename T2275700 Q3_23_CODE_01_2010   // Q3-23_CODE.01
  rename T3108500 HGC_2010 
	
	
// Rename some common variables
	rename CASEID_1979 id
	rename SAMPLE_ID_1979 SAMPLE_ID
	rename SAMPLE_RACE_1979 race
	rename SAMPLE_SEX_1979 sex
	
// Rename some variables to make reshape easier
  rename Q3_23_1_1979 field1979 
  rename Q3_23_1_1980 field1980 
  rename Q3_23_1_1981 field1981 
  rename Q3_23_1_1982 field1982 
  rename Q3_23_1_1983   field1983
  rename Q3_23_1_1984  field1984
  rename Q3_23_1_1985   field1985
  rename Q3_23_01_1986   field1986
  rename Q3_23_1_1988   field1988
  rename Q3_23_1_1989   field1989
  rename Q3_23_1_1990   field1990
  rename Q3_23_1_1992   field1992
  rename Q3_23_01_1993   field1993
  rename Q3_23_1_1994   field1994
  rename Q3_23_01_1996   field1996
  rename Q3_23_CODE_01_1998   field1998
  rename Q3_23_CODE_01_2000   field2000
  rename Q3_23_CODE_01_2002   field2002
  rename Q3_23_CODE_01_2006   field2006
  rename Q3_23_CODE_01_2008   field2008
  rename Q3_23_CODE_01_2010   field2010

rename ASVAB_13_1981  ASVAB_SCIENCE
rename ASVAB_15_1981  ASVAB_ARITHMETIC
rename ASVAB_17_1981  ASVAB_WORD_KNLDG
rename ASVAB_19_1981  ASVAB_PARAGRAPH_COMP
rename ASVAB_21_1981  ASVAB_NUMERIC_OPERS
rename ASVAB_23_1981  ASVAB_CODING_SPEED
rename ASVAB_25_1981  ASVAB_AUTO_SHOP_INFO
rename ASVAB_27_1981  ASVAB_MATH_KNLDG
rename ASVAB_29_1981  ASVAB_MECH_COMP
rename ASVAB_31_1981  ASVAB_ELCTRNIC_INFO

aorder

// Code field of study as field of study in last eduction attended
generate field = field1979
replace  field = field1980 if (field1980 < .)
replace  field = field1981 if (field1981 < .)
replace  field = field1982 if (field1982 < .)
replace  field = field1983 if (field1983 < .)
replace  field = field1984 if (field1984 < .)
replace  field = field1985 if (field1985 < .)
replace  field = field1986 if (field1986 < .)
replace  field = field1988 if (field1988 < .)
replace  field = field1989 if (field1989 < .)
replace  field = field1990 if (field1990 < .)
replace  field = field1992 if (field1992 < .)
replace  field = field1993 if (field1993 < .)
replace  field = field1994 if (field1994 < .)
replace  field = field1996 if (field1996 < .)
replace  field = field1998 if (field1998 < .)
replace  field = field2000 if (field2000 < .)
replace  field = field2002 if (field2002 < .)
replace  field = field2006 if (field2006 < .)
replace  field = field2008 if (field2008 < .)
replace  field = field2010 if (field2010 < .)
//cleanup
drop field1979-field2010

generate highest_grade = HGC_1979
replace  highest_grade = HGC_1980 if ( (HGC_1980 < .) )
replace  highest_grade = HGC_1981 if ( (HGC_1981 < .) )
replace  highest_grade = HGC_1982 if ( (HGC_1982 < .) )
replace  highest_grade = HGC_1983 if ( (HGC_1983 < .) )
replace  highest_grade = HGC_1984 if ( (HGC_1984 < .) )
replace  highest_grade = HGC_1985 if ( (HGC_1985 < .) )
replace  highest_grade = HGC_1986 if ( (HGC_1986 < .) )
replace  highest_grade = HGC_1987 if ( (HGC_1987 < .) )
replace  highest_grade = HGC_1988 if ( (HGC_1988 < .) )
replace  highest_grade = HGC_1989 if ( (HGC_1989 < .) )
replace  highest_grade = HGC_1990 if ( (HGC_1990 < .) )
replace  highest_grade = HGC_1991 if ( (HGC_1991 < .) )
replace  highest_grade = HGC_1992 if ( (HGC_1992 < .) )
replace  highest_grade = HGC_1993 if ( (HGC_1993 < .) )
replace  highest_grade = HGC_1994 if ( (HGC_1994 < .) )
replace  highest_grade = HGC_1996 if ( (HGC_1996 < .) )
replace  highest_grade = HGC_1998 if ( (HGC_1998 < .) )
replace  highest_grade = HGC_2000 if ( (HGC_2000 < .) )
replace  highest_grade = HGC_2002 if ( (HGC_2002 < .) )
replace  highest_grade = HGC_2004 if ( (HGC_2004 < .) )
replace  highest_grade = HGC_2006 if ( (HGC_2006 < .) )
replace  highest_grade = HGC_2008 if ( (HGC_2008 < .) )
replace  highest_grade = HGC_2010 if ( (HGC_2010 < .) )

//cleanup
drop HGC_1979-HGC_2010

order id SAMPLE_ID race sex AFQT* highest_grade SAMPW* ASVAB* field

// compress the data and save.
compress
save lpv_data_cross_section, replace

