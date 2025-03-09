clear all
use "C:\Users\eduar\Downloads\T1.dta"

*****************************************************************************************
//EXPLORATORY ANALYSIS

tab year, gen(y)

global yxvars lcrmrte lprbconv lprbarr lavgsen lpolpc ldensity taxpc west central urban
global continous lprbconv lprbarr lavgsen lpolpc ldensity taxpc
global dummies west central urban y2-y7
global regressors prbconv lprbarr lavgsen lpolpc ldensity taxpc west central urban y2-y7

//UNIVARIATE ANALYSIS
describe
sum  lcrmrte $continous, detail
tab west
tab central
tab urban

//BIVARIATE ANALYSIS
pwcorr lcrmrte $continous west urban central, star(0.05)
foreach var of varlist $dummies{
	bysort `var': sum lcrmrte
}

//UNIVARIATE GRAPHS
twoway histogram lcrmrte || kdensity lcrmrte
twoway histogram lprbconv || kdensity lprbconv
twoway histogram lprbarr || kdensity lprbarr
twoway histogram lpolpc || kdensity lpolpc
twoway histogram ldensity || kdensity ldensity
twoway histogram taxpc || kdensity taxpc

//BIVARIATE GRAPHS
twoway (scatter lcrmrte lprbconv) (lfit lcrmrte lprbconv)
twoway (scatter lcrmrte lprbarr) (lfit lcrmrte lprbarr)
twoway (scatter lcrmrte lavgsen) (lfit lcrmrte lavgsen)
twoway (scatter lcrmrte lpolpc) (lfit lcrmrte lpolpc)
twoway (scatter lcrmrte ldensity) (lfit lcrmrte ldensity)
twoway (scatter lcrmrte taxpc) (lfit lcrmrte taxpc)

graph box lcrmrte, over(west)
graph box lcrmrte, over(central)
graph box lcrmrte, over(urban)
graph box lcrmrte, over(y2)
graph box lcrmrte, over(y3)
graph box lcrmrte, over(y4)
graph box lcrmrte, over(y5)
graph box lcrmrte, over(y6)
graph box lcrmrte, over(y7)

//PANEL DATA ANALYSIS
xtset county year, yearly

xtdescribe
xtsum $yxvars
xttab west
xttab central
xttab urban 

*****************************************************************************************
//POOLES OLS 
reg $yxvars y2-y7
    
	//heteroscedasticity test
	xtgls $yxvars y2-y7, igls panels(h)
	estimates store hetero
	xtgls $yxvars y2-y7, igls 
	local df = e(N_g) - 1
	lrtest hetero . , df(`df') //reject the null so the heteroscedasticity is present

xtserial lcrmrte lprbconv lprbarr lavgsen lpolpc ldensity taxpc west central urban y2-y7
reg $yxvars y2-y7, vce(cluster county)
estimates store beta_POLS
test (y2 y3 y4 y5 y6 y7)

*****************************************************************************************
//RANDOM EFFECTS
xtreg $yxvars y2-y7, re

//heteroscedasticity test
predict uhar, ue
predict xb, xb
gen uhatsq = uhar^2
reg uhatsq c.xb##c.xb, vce(cl county)
testparm c.xb##c.xb

xttest0
xttest1
predict res, e
xtcdf res
xtreg $yxvars y2-y7, re vce(cluster county)
estimates store beta_rec

*****************************************************************************************
//FIXED EFFECTS
xtreg lcrmrte lprbconv lprbarr lavgsen lpolpc ldensity taxpc y2-y7 , fe
xttest3
xtserial lcrmrte lprbconv lprbarr lavgsen lpolpc ldensity taxpc y2-y7
xtreg lcrmrte lprbconv lprbarr lavgsen lpolpc ldensity taxpc y2-y7, fe vce(cluster county)
estimates store beta_fec

*****************************************************************************************
//Comparison of the models
estimates table beta_POLS beta_rec beta_fec, star(0.01 0.05 0.1)

*****************************************************************************************
//Choice of adequadate model 
** Robust Hausman Specification Test
set seed 18031997
xtreg $yxvars y2-y7, re vce(cluster county)
estimates store b_r_re
xtreg $yxvars y2-y7, fe vce(cluster county)
estimates store b_r_fe

rhausman b_r_fe b_r_re, reps(200) cluster //choose fixed effects

** Wooldrige robust Hausman test
foreach var of varlist lprbconv lprbarr lavgsen lpolpc ldensity taxpc  {
	by county: egen m`var' = mean(`var')
}
reg $yxvars mlprbconv mlprbarr mlavgsen mlpolpc mldensity mtaxpc y2-y7,vce(cluster county)
test mlprbconv mlprbarr mlavgsen mlpolpc mldensity mtaxpc //choose fixed effects