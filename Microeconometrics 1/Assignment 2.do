clear all
cd "C:\Users\eduar\OneDrive\Escritorio\Masterado\Segundo Semestre\Microeconometrics\Assignment 2"
use "T2.dta"

global regressors l.log_loans int_rate_dep l.int_rate_dep int_rate_loan l.int_rate_loan log_disp_inc l.log_disp_inc

xtset id year, yearly

//POOLED OLS 
reg log_loans $regressors td3-td6, vce(cluster id)
predict resid, res
estimates store POLS
//(.9379136) for the lag dep coef, also upward bias, upper limit

//FIXED EFFECTS
xtreg log_loans $regressors td3-td6, fe vce(cluster id)
estimates store FE
// (.4038686) downward bias, lower limit

//FIRST DIFFERENCE
reg D.(log_loans $regressors td3-td6), nocons vce(cluster id)
estimates store FD

estimates table POLS FE FD, star(.1 .05 .01)

*****************************************************************************************
//one step GMM, robust s.e., xb are strictly exogenous variables

xtabond2 log_loans $regressors td3-td6, iv(int_rate_dep int_rate_loan log_disp_inc td3-td6) gmm(l.log_loans) noleveleq robust
//AR1 and AR2 test sugests serial correlation

//one step differenced GMM, robust s.e., xb are predetermined
xtabond2 log_loans $regressors td3-td6, iv(td3-td6) gmm(l.log_loans) gmm(int_rate_dep int_rate_loan log_disp_inc) noleveleq robust
//(.408497) close to the lower limit so a system gmm should be used
//AR1 test suggests serial correlation of errors in first degree

//Two-step differneced GMM, endogenous variables, windmeijer correction
xtabond2 log_loans $regressors td3-td6, iv(td3-td6) gmm(l.log_loans) gmm(int_rate_dep int_rate_loan log_disp_inc) noleveleq twostep robust

//one step system GMM, robust s.e., xb are predetermined
xtabond2 log_loans $regressors td3-td6, iv(td3-td6) gmm(l.log_loans) gmm(int_rate_dep int_rate_loan log_disp_inc) arlevels robust 


//Two-step GMM, endogenous variables, windmeijer correction
xtabond2 log_loans $regressors td3-td6, iv(td3-td6) gmm(l.log_loans) gmm(int_rate_dep int_rate_loan log_disp_inc) arlevels twostep robust
