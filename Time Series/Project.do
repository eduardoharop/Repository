clear all
//DATA PREPARATION
import excel "C:\Users\eduar\OneDrive\Escritorio\Masterado\Primer Semestre\Time Series Analysis\Project\Sweden's air temperature\stockholm_daily_mean_temperature.xlsx", sheet("stockholm_daily_mean_temperatur") firstrow
keep in 52596/97481
gen date2 = date(date,"MDY")
format date2 %td
drop date
rename date2 date
gen date2 = mofd(date)
format date2 %tm
collapse (mean) raw homo adjust, by(date2)
rename date2 date
tsset date

//EXPLORATORY ANALYSIS
tsline adjust
varsoc adjust, exog(date)
corrgram adjust, lags(100)
ac adjust, lags(100)
pac adjust , lags(100)
dfuller adjust, lags(4)

// SEASONAL DIFFERENCING TIME SERIES
gen diff = adjust - L12.adjust
tsline diff
varsoc diff, exog(date)
corrgram diff, lags(100)
ac diff, lags(100) // q = 3 and Q = 1
pac diff, lags(100)	// p = 1 and P = 0
dfuller diff, lags(4)

//NORMAL DIFFERENCING TIME SERIES
gen ndiff = diff - L.diff
tsline ndiff
varsoc ndiff
corrgram ndiff, lags(100)
ac ndiff, lags(100)
pac ndiff, lags(100)

//MODEL1
arima adjust, arima(1,0,3) sarima(0,1,1,12) noconst //modelo 1
predict res1, res
tsline res1
ac res1
pac res1
wntestq res1 
drop res1
estat ic // AIC:5991.013 BIC:6022.742

//MODEL2
arima adjust, arima(1,0,2) sarima(0,1,1,12) noconst //modelo 1
predict res2, res 
tsline res2
ac res2
pac res2
wntestq res2, lags(3)
drop res2
estat ic // AIC:6000.81  BIC:6027.251


//MODEL3
arima adjust, arima(1,0,1) sarima(0,1,1,12) noconst //modelo 1
predict res3, res
tsline res3
ac res3, lags(100)
pac res3, lags(100)
wntestq res3, lags(10)
drop res3
estat ic // AIC:6004.801  BIC:6025.954


//MODEL4
arima adjust, arima(1,0,0) sarima(0,1,1,12) noconst //modelo 1
predict res4, res
tsline res4
ac res4
pac res4
wntestq res4, lags(3)
drop res4
estat ic // AIC:6006.735  BIC:6022.6


//MODEL5
arima adjust, arima(0,0,1) sarima(0,1,1,12) noconst //modelo 1
predict res5, res
tsline res5
ac res5
pac res5
wntestq res5, lags(3)
drop res5
estat ic // AIC:6046.98  BIC:6062.845
