options(digits = 6, width = 60)
rm(list=ls())

library(MTS)
library(tidyverse)
library(tidyquant)
library(tsibble)
library(fpp3)
library(sandwich)
library(lmtest)
library(GGally)
library(kableExtra)
library(timeSeries)
library(ggplot2)
library(aTSA)
library(forecast)
library(rugarch)
library(sandwich)
library(lmtest)
library(Metrics)
library(stargazer)

##################################################################################################################
##PART 1

##Dataset creation
df <- tidyquant::tq_get(c("MSFT","WMT"),get = "stock.prices" ,from="2018-03-01", to="2023-02-28") %>%
        as_tsibble(index = date,key = symbol) %>% 
        group_by_key() %>%
        mutate(rtn = difference(log(close))) %>% 
        drop_na() %>% 
        group_by_key() %>% 
        mutate(trading_day = row_number()) 
#2b)        
##Plots on price 
df %>% 
  autoplot(close)+facet_grid(vars(symbol), scale = "free_y")+
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

##Plots on logreturns
df %>% 
  autoplot(rtn)+facet_grid(vars(symbol), scale = "free_y" )+
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

##descriptve statistics on prices
df %>%
  features(close,list(n = ~length(.), min = ~min(.), "1stQ"= ~quantile(.)[2],mean = ~mean(.),median = ~median(.),"3rdQ" = ~quantile(.)[4],max = ~max(.),skew =timeSeries::colSkewness, kurt =  timeSeries::colKurtosis 
  )) 
df %>% dplyr::filter(symbol == "MSFT") %>% dplyr::select(close) %>% dplyr::filter(close > quantile(close,probs = 0.75)+1.5*(quantile(close,probs = 0.75)-quantile(close,probs = 0.25)) | close < quantile(close,probs = 0.25)-1.5*(quantile(close,probs = 0.75)-quantile(close,probs = 0.25))) #no presenta outliers 
df %>% dplyr::filter(symbol == "WMT") %>% dplyr::select(close) %>% dplyr::filter(close > quantile(close,probs = 0.75)+1.5*(quantile(close,probs = 0.75)-quantile(close,probs = 0.25)) | close < quantile(close,probs = 0.25)-1.5*(quantile(close,probs = 0.75)-quantile(close,probs = 0.25)))  #no presenta outliers


##descriptve statistics on logreturns
df %>%
  features(rtn,list(n = ~length(.), min = ~min(.), "1stQ"= ~quantile(.)[2],mean = ~mean(.),median = ~median(.),"3rdQ" = ~quantile(.)[4],max = ~max(.),skew =timeSeries::colSkewness, kurt =  timeSeries::colKurtosis 
  )) 
df %>% dplyr::filter(symbol == "MSFT") %>% dplyr::select(rtn) %>% dplyr::filter(rtn > quantile(rtn,probs = 0.75)+1.5*(quantile(rtn,probs = 0.75)-quantile(rtn,probs = 0.25)) | rtn < quantile(rtn,probs = 0.25)-1.5*(quantile(rtn,probs = 0.75)-quantile(rtn,probs = 0.25)))  #presenta outliers
df %>% dplyr::filter(symbol == "WMT") %>% dplyr::select(rtn) %>% dplyr::filter(rtn > quantile(rtn,probs = 0.75)+1.5*(quantile(rtn,probs = 0.75)-quantile(rtn,probs = 0.25)) | rtn < quantile(rtn,probs = 0.25)-1.5*(quantile(rtn,probs = 0.75)-quantile(rtn,probs = 0.25)))  #presenta outliers


##graphs for prices
ggplot(df, aes(x=close,y = ..density..))+
  geom_histogram(bins = 100)+ geom_density() +
  facet_grid(vars(symbol), scale = "free_y" ) +  
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

ggplot(df, aes(x=symbol, y = close)) +
  geom_boxplot() 
  theme(legend.position="none", axis.text=element_text(size=6),
         axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

##graphs for logreturns
ggplot(df, aes(x=rtn,y = ..density..))+
    geom_histogram(bins = 100)+ geom_density() +
    facet_grid(vars(symbol), scale = "free_y" ) +  
    theme(legend.position="none", axis.text=element_text(size=6),
          axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

ggplot(df, aes(x=symbol, y = rtn)) +
  geom_boxplot() 
theme(legend.position="none", axis.text=element_text(size=6),
      axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

#2c)
##ADF for logprices of Microsoft
ADFtestMSFT <- df %>% 
  as_tibble() %>% 
  dplyr::filter(symbol=="MSFT") %>%
  mutate(logprices = log(close)) %>%
  dplyr::select(logprices) %>%
  ts() %>% 
  adf.test()
stargazer(ADFtestMSFT[["type1"]],ADFtestMSFT[["type2"]],ADFtestMSFT[["type3"]], type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 2)

help(stargazer)
##ADF for logprices of Wallmart
ADFtestWMT <- df %>% 
  as_tibble() %>% 
  dplyr::filter(symbol=="WMT") %>%
  mutate(logprices = log(close)) %>%
  dplyr::select(logprices) %>%
  ts() %>% 
  adf.test()
stargazer(ADFtestWMT[["type1"]],ADFtestWMT[["type2"]],ADFtestWMT[["type3"]], type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 2)


#2d)
##Arima Model for Microsoft
##ACF and PACF of Microsoft
df %>% 
  as_tibble() %>% 
  dplyr::filter(symbol=="MSFT") %>% 
  dplyr::select(rtn) %>%
  ts() %>% 
  adf.test()

df %>%
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="MSFT") %>% 
  dplyr::select(rtn) %>% 
  ACF(rtn,lag_max = 100) %>% 
  autoplot()+
  ggtitle("Microsoft ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))
df %>%
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="MSFT") %>% 
  dplyr::select(rtn) %>% 
  PACF(rtn,lag_max = 100) %>% 
  autoplot()+
  ggtitle("Microsoft PACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

arimaMSF <- df %>%
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="MSFT") %>% 
  dplyr::select(rtn) %>% 
  auto.arima(max.p = 18,max.q = 24, start.p = 0, start.q = 0, seasonal = FALSE,stepwise = FALSE)
arimaMSF

checkresiduals(arimaMSF, lag= 7)

df <- df %>% 
  mutate(resMSFT = arimaMSF[["residuals"]], resMSFT2 = (arimaMSF[["residuals"]])^2)


df %>% 
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="MSFT") %>% 
  dplyr::select(resMSFT) %>% 
  PACF(resMSFT,lag_max = 30) %>% 
  autoplot()+
  ggtitle("Miscrosoft residuals PACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))



##Arima Model for Wallmart
##ACF and PACF of Wallmart
df %>% 
  as_tibble() %>% 
  dplyr::filter(symbol=="WMT") %>% 
  dplyr::select(rtn) %>%
  ts() %>% 
  adf.test()

df %>% 
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="WMT") %>% 
  dplyr::select(rtn) %>% 
  ACF(rtn,lag_max = 100) %>% 
  autoplot()+
  ggtitle("Wallmart ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

df %>% 
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="WMT") %>% 
  dplyr::select(rtn) %>% 
  PACF(rtn,lag_max = 100) %>% 
  autoplot()+
  ggtitle("Wallmart PACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

arimaWMT <- df %>%
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="WMT") %>% 
  dplyr::select(rtn) %>% 
  auto.arima(max.p = 9,max.q = 9, start.p = 1, start.q = 1, seasonal = FALSE, stepwise = FALSE)
arimaWMT

checkresiduals(arimaWMT, lag= 7)

df <- df %>% 
  mutate(resWMT = arimaWMT[["residuals"]], resWMT2 = (arimaWMT[["residuals"]])^2)

df %>% 
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="WMT") %>% 
  dplyr::select(resWMT) %>% 
  PACF(resWMT,lag_max = 30) %>% 
  autoplot()+
  ggtitle("Wallmart residuals PACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))



#2e)
##GARCH-Microsoft
archTest(arimaMSF[["residuals"]], lag = 1) ##residual have prescence of conditional heteroschedasticity


df %>%
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="MSFT") %>% 
  dplyr::select(resMSFT2) %>% 
  ACF(resMSFT2,lag_max = 100) %>% 
  autoplot()+
  ggtitle("MSFT res2 ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

df %>%
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="MSFT") %>% 
  dplyr::select(resMSFT2) %>% 
  PACF(resMSFT2,lag_max = 100) %>% 
  autoplot()+
  ggtitle("MSFT res2 PACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

dataMSFT <- df %>% 
  dplyr::filter(symbol == "MSFT") %>% 
  dplyr::filter(!is.na(rtn)) %>% 
  pull(rtn)
  
  
auto.garch <- function(gp,gq,p,q,vmodel,df,distribution.model = "norm") {
if (gp == 0){
  result <- matrix(nrow = 4,ncol = gq)
  modelo <- c()
  for(i in 1:gq){
      spec <- rugarch::ugarchspec(variance.model=list(model=vmodel, garchOrder=c(0,i)),
                                  mean.model=list(armaOrder=c(p,q), 
                                                  include.mean=TRUE), distribution.model = distribution.model)
      fit <- rugarch::ugarchfit(data = df, spec = spec)
      result[,i] <- infocriteria(fit)[,1]
      modelo[i]<- c(print(paste0("GARCH ",0,",",i)))
  }
  criterios <- c("Akaike","Bayes","Shibata", "HQ")
  modelo
  rownames(result) <- criterios
  colnames(result) <- modelo
  t.result <- t(result)
  
  for(i in 1:4) {
    min <- min(t.result[,i])
    modelof <- rownames(t.result)[which(t.result[,i] == min(t.result[,i]))]
    print(paste0("El modelo con mejor criterio ", criterios[i] ," es del modelo: ", modelof))  
  }  
return(t.result)  
}
  
else if (gq  == 0){
  result <- matrix(nrow = 4,ncol = gp)
  modelo <- c()
  for(i in 1:gp){
    tryCatch({spec <- rugarch::ugarchspec(variance.model=list(model=vmodel, garchOrder=c(i,0)),
                                mean.model=list(armaOrder=c(p,q), 
                                                include.mean=TRUE), distribution.model = distribution.model)
    fit <- rugarch::ugarchfit(data = df, spec = spec)
    result[,i] <- infocriteria(fit)[,1]
    modelo[i]<- c(print(paste0("GARCH ",i,",",0))) }, error = function(e) NA)
  }
  criterios <- c("Akaike","Bayes","Shibata", "HQ")
  modelo
  rownames(result) <- criterios
  colnames(result) <- modelo
  t.result <- t(result)
  
  for(i in 1:4) {
    min <- min(t.result[,i])
    modelof <- rownames(t.result)[which(t.result[,i] == min(t.result[,i],na.rm=TRUE))]
    print(paste0("El modelo con mejor criterio ", criterios[i] ," es del modelo: ", modelof))  
  }  
  return(t.result)  
}
  
else {
  result <- matrix(nrow = 4,ncol = gp*gq)
  modelo <- c()
  for(i in 1:gp){
    for (j in 1:gq) {
      spec <- rugarch::ugarchspec(variance.model=list(model=vmodel, garchOrder=c(i,j)),
                                  mean.model=list(armaOrder=c(p,q), 
                                                  include.mean=TRUE), distribution.model = distribution.model)
      fit <- rugarch::ugarchfit(data = df, spec = spec)
      result[,i*j +(i-1)*(gq-j)] <- infocriteria(fit)[,1]
      modelo[i*j +(i-1)*(gq-j)]<- c(print(paste0("GARCH ",i,",",j)))
    }
  }
  criterios <- c("Akaike","Bayes","Shibata", "HQ")
  modelo
  rownames(result) <- criterios
  colnames(result) <- modelo
  t.result <- t(result)
  
  for(i in 1:4) {
    min <- min(t.result[,i])
    modelof <- rownames(t.result)[which(t.result[,i] == min(t.result[,i]))]
    print(paste0("El modelo con mejor criterio ", criterios[i] ," es del modelo: ", modelof))  
  }  
  return(t.result)    
  } 
}


#auto.garch(gp=10,gq =0,p =1,q = 4,vmodel="sGARCH",df=dataMSFT) ##AIC GARC(10,0) = -5.299
#auto.garch(gp=10,gq =10,p =1,q = 4,vmodel="sGARCH",df=dataMSFT) ##AIC GARCH(1,1) = -5.318

specMSFT <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                            mean.model=list(armaOrder=c(1,4), 
                                            include.mean=TRUE))
fitMSFT <- rugarch::ugarchfit(data = dataMSFT, spec = specMSFT)
options(digits = 3, width = 60)
show(fitMSFT)

##Garch Wallmart
archTest(arimaWMT[["residuals"]], lag = 1) ##residual have prescence of conditional heteroschedasticity


df %>%
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="WMT") %>% 
  dplyr::select(resWMT2) %>% 
  ACF(resWMT2,lag_max = 100) %>% 
  autoplot()+
  ggtitle("WMT res2 ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

df %>%
  update_tsibble(index = trading_day) %>%
  dplyr::filter(symbol=="WMT") %>% 
  dplyr::select(resWMT2) %>% 
    PACF(resWMT2,lag_max = 100) %>% 
  autoplot()+
  ggtitle("WMT res2 PACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

dataWMT <- df %>% 
  dplyr::filter(symbol == "WMT") %>% 
  dplyr::filter(!is.na(rtn)) %>% 
  pull(rtn)

#auto.garch(gp=9,gq =12,p =0,q = 4,vmodel="sGARCH",df=dataWMT)

specWMT <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                                mean.model=list(armaOrder=c(0,4), 
                                                include.mean=TRUE))
fitWMT <- rugarch::ugarchfit(data = dataWMT, spec = specWMT)
show(fitWMT)



##2f)Diagnostic checks
##Microsoft
  fitMSFT@fit$z %>% ts() %>% as_tsibble() %>%
  ACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Residuals ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

fitMSFT@fit$z^2 %>% ts() %>% as_tsibble() %>%
  ACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Squared Residuals ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

fitMSFT@fit$z %>% ts() %>% as_tsibble() %>%
  PACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Residuals PACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

fitMSFT@fit$z^2 %>% ts() %>% as_tsibble() %>%
  PACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Squared Residuals PACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))



qqnorm(fitMSFT@fit$z,main="",col="red")
qqline(fitMSFT@fit$z,col="blue")
archTest(fitMSFT@fit$z) ##Conditonal Heteroschedastic test on GARCH residuals show no heteroschedsticity
Box.test(fitMSFT@fit$z, lag = 1, type = c("Box-Pierce", "Ljung-Box"), fitdf = 0) ##autocorrelation test in R

##Wallmart
fitWMT@fit$z %>% ts() %>% as_tsibble() %>%
  ACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Residuals ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

fitWMT@fit$z^2 %>% ts() %>% as_tsibble() %>%
  ACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Squared Residuals ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

qqnorm(fitWMT@fit$z,main="",col="red")
qqline(fitWMT@fit$z,col="blue")
archTest(fitWMT@fit$z) ##Conditonal Heteroschedastic test on GARCH residuals show no heteroschedsticity



## 2h) t-distribution error term
##Microsoft
##auto.garch(gp=10,gq =0,p =1,q = 4,vmodel="sGARCH",df=dataMSFT,distribution.model = "std") #GARCH (10,0) = -5.336
##auto.garch(gp=10,gq =10,p =1,q = 4,vmodel="sGARCH",df=dataMSFT,distribution.model = "std") #GARCH (1,1) = -5.352

specMSFTstd <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                               mean.model=list(armaOrder=c(1,4), 
                                               include.mean=TRUE),distribution.model = "std")
fitMSFTstd <- rugarch::ugarchfit(data = dataMSFT, spec = specMSFTstd)
show(fitMSFTstd)

fitMSFTstd@fit$z %>% ts() %>% as_tsibble() %>%
  ACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Residuals t-student ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

fitMSFTstd@fit$z^2 %>% ts() %>% as_tsibble() %>%
  ACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Residuals t-student ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

qqnorm(fitMSFTstd@fit$z,main="",col="red")
qqline(fitMSFTstd@fit$z,col="blue")
show(fitMSFTstd)


##Wallmart
##auto.garch(gp=9,gq =12,p =0,q = 4,vmodel="sGARCH",df=dataMSFT,distribution.model = "std") #GARCH (1,1)

specWMTstd <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                                   mean.model=list(armaOrder=c(0,4), 
                                                   include.mean=TRUE),distribution.model = "std")
fitWMTstd <- rugarch::ugarchfit(data = dataMSFT, spec = specWMTstd)
show(fitWMTstd)

fitWMTstd@fit$z %>% ts() %>% as_tsibble() %>%
  ACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Residuals t-student ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

fitWMTstd@fit$z^2 %>% ts() %>% as_tsibble() %>%
  ACF(lag_max = 20) %>% autoplot()+
  ggtitle("GARCH(1,1) Standarized Residuals t-student ACF")+
  theme(legend.position="none", axis.text=element_text(size=10),
        axis.title.x = element_text(size = 10), axis.title.y = element_text(size = 10))

qqnorm(fitWMTstd@fit$z,main="",col="red")
qqline(fitWMTstd@fit$z,col="blue")
show(fitWMTstd)

##2i) Forecasts
##For Microsoft
trainMSFT <- dataMSFT[1:1193]
testMSFT <- dataMSFT[1194:1256]
spec1MSFT <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                                  mean.model=list(armaOrder=c(1,4), 
                                                  include.mean=TRUE),distribution.model = "std")
spec2MSFT <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(0,1)),
                                 mean.model=list(armaOrder=c(1,4), 
                                                 include.mean=TRUE),distribution.model = "std")
fit1MSFT <- rugarch::ugarchfit(data = trainMSFT, spec = spec1MSFT)
fit2MSFT <- rugarch::ugarchfit(data = trainMSFT, spec = spec2MSFT)
forecast1MSFT <- rugarch::ugarchforecast(fit1MSFT,n.ahead = 63)
forecast2MSFT <- rugarch::ugarchforecast(fit2MSFT,n.ahead = 63)

RMSEfitMSFT <- c(rmse(testMSFT,forecast1MSFT@forecast[["seriesFor"]]),rmse(testMSFT,forecast2MSFT@forecast[["seriesFor"]]))
MAEfitMSFT <- c(mae(testMSFT,forecast1MSFT@forecast[["seriesFor"]]),mae(testMSFT,forecast2MSFT@forecast[["seriesFor"]]))
MSEfitMSFT <- c(mse(testMSFT,forecast1MSFT@forecast[["seriesFor"]]),mse(testMSFT,forecast2MSFT@forecast[["seriesFor"]]))

metricsMSFT <- matrix(c(RMSEfitMSFT,MAEfitMSFT,MSEfitMSFT),byrow = FALSE,ncol=3)
colnames(metricsMSFT) <- c("RMSE","MAE","MSE")
rownames(metricsMSFT) <- c("Model1","Model2")
metricsMSFT

fitfinal1MSFT <- rugarch::ugarchfit(data = dataMSFT, spec = spec1MSFT)
fitfinal2MSFT <- rugarch::ugarchfit(data = dataMSFT, spec = spec2MSFT,solver = "hybrid")

rugarch::ugarchforecast(fitfinal1MSFT,n.ahead = 65)

##For Wallmart
trainWMT <- dataWMT[1:1193]
testWMT <- dataWMT[1194:1256]
spec1WMT <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                                 mean.model=list(armaOrder=c(1,4), 
                                                 include.mean=TRUE),distribution.model = "std")
spec2WMT <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,0)),
                                 mean.model=list(armaOrder=c(1,4), 
                                                 include.mean=TRUE),distribution.model = "std")
fit1WMT <- rugarch::ugarchfit(data = trainWMT, spec = spec1WMT)
fit2WMT <- rugarch::ugarchfit(data = trainWMT, spec = spec2WMT)
forecast1WMT <- rugarch::ugarchforecast(fit1WMT,n.ahead = 63)
forecast2WMT <- rugarch::ugarchforecast(fit2WMT,n.ahead = 63)

RMSEfitWMT <- c(rmse(testWMT,forecast1WMT@forecast[["seriesFor"]]),rmse(testWMT,forecast2WMT@forecast[["seriesFor"]]))
MAEfitWMT <- c(mae(testWMT,forecast1WMT@forecast[["seriesFor"]]),mae(testWMT,forecast2WMT@forecast[["seriesFor"]]))
MSEfitWMT <- c(mse(testWMT,forecast1WMT@forecast[["seriesFor"]]),mse(testWMT,forecast2WMT@forecast[["seriesFor"]]))

metricsWMT <- matrix(c(RMSEfitWMT,MAEfitWMT,MSEfitWMT),byrow = FALSE,ncol=3)
colnames(metricsWMT) <- c("RMSE","MAE","MSE")
rownames(metricsWMT) <- c("Model1","Model2")
metricsWMT

fitfinal1WMT <- rugarch::ugarchfit(data = dataWMT, spec = spec1WMT)
fitfinal2WMT <- rugarch::ugarchfit(data = dataWMT, spec = spec2WMT)
rugarch::ugarchforecast(fitfinal1WMT,n.ahead = 57)


##2j) Only for Microsoft  
MSFTforecast <- rugarch::ugarchforecast(fitfinal1MSFT,n.ahead = 2)
MSFTforecaststd <- rugarch::ugarchforecast(fitfinal2MSFT,n.ahead = 2)

MSFTforecast <- matrix(data = c(MSFTforecast@forecast$seriesFor,MSFTforecast@forecast$sigmaFor,MSFTforecaststd@forecast$seriesFor,MSFTforecaststd@forecast$sigmaFor),nrow=2,ncol=4,byrow = FALSE)
colnames(MSFTforecast) <- c("seriesModel1","sigmaModel1","seriesModel2","sigmaModel2")
MSFTforecast



##2k) Stylized facts of Series
##Microsoft
#Mean = 0
ttestMSFT <- lm(rtn ~ 1, data = df %>% dplyr::filter(symbol == "MSFT") %>% dplyr::select(rtn))
coeftest(ttestMSFT, vcov = vcovHC(ttestMSFT, type = "HC0"))

#Leverage effects, T-GARCH
specMSFTtgarch <- rugarch::ugarchspec(variance.model=list(model="gjrGARCH", garchOrder=c(1,1)),
                                 mean.model=list(armaOrder=c(1,4), 
                                                 include.mean=TRUE))
fitMSFTtgarch <- rugarch::ugarchfit(data = dataMSFT, spec = specMSFTtgarch)
show(fitMSFTtgarch)
#Risk premia, - M-GARCH
specMSFTmgarch <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                                      mean.model=list(armaOrder=c(1,4), 
                                                      include.mean=TRUE, archm = TRUE))
fitMSFTmgarch <- rugarch::ugarchfit(data = dataMSFT, spec = specMSFTmgarch)
show(fitMSFTmgarch)

##Wallmart
#Mean = 0

ttestWMT <- lm(rtn ~ 1, data = df %>% dplyr::filter(symbol == "WMT") %>% dplyr::select(rtn))
coeftest(ttestWMT, vcov = vcovHC(ttestWMT, type = "HC0"))

#Leverage effects, T-GARCH
specWMTtgarch <- rugarch::ugarchspec(variance.model=list(model="gjrGARCH", garchOrder=c(1,1)),
                                      mean.model=list(armaOrder=c(1,4), 
                                                      include.mean=TRUE))
fitWMTtgarch <- rugarch::ugarchfit(data = dataWMT, spec = specWMTtgarch)
show(fitWMTtgarch)
#Risk premia, - M-GARCH
specWMTmgarch <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                                      mean.model=list(armaOrder=c(1,4), 
                                                      include.mean=TRUE, archm = TRUE))
fitWMTmgarch <- rugarch::ugarchfit(data = dataWMT, spec = specWMTmgarch)
show(fitWMTmgarch)

#########################################################################################################
#PART 2
options(digits = 1, width = 60)
capital <- 10000000
##2ai)
##Microsoft
results2aMSFT <- matrix(nrow =5,ncol=6)
partial <- matrix(ncol=3,nrow=5)
for(i in 1:5){
partial[i,] <- PerformanceAnalytics::VaR(dataMSFT,p =c(0.90,0.95,0.99) ,method = "historical")
    if (i == 1){
    VaR_1 <- partial[i,]
    results1 <- matrix(c(VaR_1,rep(c(NA,NA,NA),4)),byrow = TRUE,nrow=5,ncol=3)
    }
    if (i == 5){
    results2aMSFT <- cbind(results1,partial) 
    colnames(results2aMSFT)<- c("VaR_10%","VaR_5%","VaR_1%","VaR_10%","VaR_5%","VaR_1%")
    rownames(results2aMSFT)<- c("T+1","T+2","T+3","T+4","T+5")
    }
}
capital*results2aMSFT
stargazer(capital*results2aMSFT, type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits=0)

##Wallmart
results2aWMT <- matrix(nrow =5,ncol=6)
partial <- matrix(ncol=3,nrow=5)
for(i in 1:5){
  partial[i,] <- PerformanceAnalytics::VaR(dataWMT,p =c(0.90,0.95,0.99) ,method = "historical")
  if (i == 1){
    VaR_1 <- partial[i,]
    results1 <- matrix(c(VaR_1,rep(c(NA,NA,NA),4)),byrow = TRUE,nrow=5,ncol=3)
  }
  if (i == 5){
    results2aWMT <- cbind(results1,partial) 
    colnames(results2aWMT)<- c("VaR_10%","VaR_5%","VaR_1%","VaR_10%","VaR_5%","VaR_1%")
    rownames(results2aWMT)<- c("T+1","T+2","T+3","T+4","T+5")  
    }
}
capital*results2aWMT
stargazer(capital*results2aWMT, type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 0)

##2aii)
#Microsoft
VAR <- function(gp,gq,p,q,data,variance.model="sGARCH",distribution.model = "norm"){

    specVAR <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(gp,gq)),
                                 mean.model=list(armaOrder=c(p,q), 
                                                 include.mean=TRUE),distribution.model =distribution.model)
  fitVAR <- rugarch::ugarchfit(data = data, spec = specVAR)
  forecastVAR <- rugarch::ugarchforecast(fitVAR,n.ahead = 5)
  results <- matrix(nrow =5,ncol=6)  
  partial <- matrix(ncol=3,nrow=5)
  if (distribution.model == "norm"){
  for(i in 1:5){
  partial[i,] <- -(forecastVAR@forecast$seriesFor[i] + qnorm(p = c(0.9,0.95,0.99),mean =0,sd=1)*forecastVAR@forecast$sigmaFor[i])
  if (i == 1){
    VaR_1 <- -(forecastVAR@forecast$seriesFor[i] + qnorm(p = c(0.9,0.95,0.99),mean =0,sd=1)*forecastVAR@forecast$sigmaFor[i])
    results1 <- matrix(c(VaR_1,rep(c(NA,NA,NA),4)),byrow = TRUE,nrow=5,ncol=3)
  }
  if (i == 5){
    results <- cbind(results1,partial) 
    colnames(results)<- c("VaR_10%","VaR_5%","VaR_1%","VaR_10%","VaR_5%","VaR_1%")
    rownames(results)<- c("T+1","T+2","T+3","T+4","T+5")
  }
}
show(fitVAR)
capital*results
rugarch::ugarchforecast(fitVAR,n.ahead = 1)
} 
  else if (distribution.model == "std"){
    for(i in 1:5){
      partial[i,] <- -(forecastVAR@forecast$seriesFor[i] + forecastVAR@forecast$sigmaFor[i]*qdist(distribution='std', shape=fitVAR@fit[["coef"]][["shape"]], p=c(0.90,0.95,0.99)))
      if (i == 1){
        VaR_1 <- partial[i,]
        results1 <- matrix(c(VaR_1,rep(c(NA,NA,NA),4)),byrow = TRUE,nrow=5,ncol=3)
      }
      if (i == 5){
        results <- cbind(results1,partial) 
        colnames(results)<- c("VaR_10%","VaR_5%","VaR_1%","VaR_10%","VaR_5%","VaR_1%")
        rownames(results)<- c("T+1","T+2","T+3","T+4","T+5")
      }
}
    show(fitVAR)
    capital*results
    rugarch::ugarchforecast(fitVAR,n.ahead = 1)
} 
  else {
  }
}
VAR(gp=1,gq=1,p=0,q=0,data = dataMSFT,distribution.model = "norm")
stargazer(VAR(gp=1,gq=1,p=0,q=0,data = dataMSFT,distribution.model = "norm"), type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 0)
##Wallmart
VAR(gp=1,gq=1,p=0,q=0,data = dataWMT,distribution.model = "norm")
stargazer(VAR(gp=1,gq=1,p=0,q=0,data = dataWMT,distribution.model = "norm"), type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 0)


##2aiii)
##Microsoft
VAR(gp=1,gq=1,p=1,q=0,data = dataMSFT,distribution.model = "norm")
stargazer(VAR(gp=1,gq=1,p=1,q=0,data = dataMSFT,distribution.model = "norm"), type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 0)

##Wallmart
VAR(gp=1,gq=1,p=1,q=0,data = dataWMT,distribution.model = "norm")
stargazer(VAR(gp=1,gq=1,p=1,q=0,data = dataWMT,distribution.model = "norm"), type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 0)

##2aiv)
##Microsoft
VAR(gp=1,gq=1,p=1,q=0,data = dataMSFT,distribution.model = "std")
stargazer(VAR(gp=1,gq=1,p=1,q=0,data = dataMSFT,distribution.model = "std"), type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 0)

##Wallmart
VAR(gp=1,gq=1,p=1,q=0,data = dataWMT,distribution.model = "std")
stargazer(VAR(gp=1,gq=1,p=1,q=0,data = dataWMT,distribution.model = "std"), type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 0)

##2av)
##Microsoft
VAR(gp=1,gq=1,p=0,q=0,data = dataMSFT,distribution.model = "std")
stargazer(VAR(gp=1,gq=1,p=0,q=0,data = dataMSFT,distribution.model = "std"), type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 0)

##Wallmart
VAR(gp=1,gq=1,p=0,q=0,data = dataWMT,distribution.model = "std")
stargazer(VAR(gp=1,gq=1,p=0,q=0,data = dataWMT,distribution.model = "std"), type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 0)

##2b)
#Diagonal Vectorization (VEC) model 
options(digits = 6, width = 60)


df_wider <- df %>% 
  dplyr::select(symbol,date,rtn)%>% 
  pivot_wider(names_from = "symbol",values_from = "rtn")

df_wider[,2:3] %>% MTS::MarchTest()

##i)
  results2b <- matrix(nrow =5,ncol=6)
  partialb <- matrix(ncol=3,nrow=5)
  for(i in 1:5){
  partialMSFT <- PerformanceAnalytics::VaR(dataMSFT,p =c(0.90,0.95,0.99) ,method = "historical")
  partialWMT <- PerformanceAnalytics::VaR(dataWMT,p =c(0.90,0.95,0.99) ,method = "historical")
  cor <- cor(dataMSFT,dataWMT)
  partialb[i,] <- -sqrt(partialMSFT^2 + partialWMT^2 + cor*partialMSFT*partialWMT)
  if (i == 1){
    VaR_1 <- partialb[i,] 
    results1 <- matrix(c(VaR_1,rep(c(NA,NA,NA),4)),byrow = TRUE,nrow=5,ncol=3)
  }
  if (i == 5){
    results2b <- cbind(results1,partialb) 
    colnames(results2aMSFT)<- c("VaR_10%_1","VaR_5%_1","VaR_1%_1","VaR_10%_5","VaR_5%_5","VaR_1%_5")
    rownames(results2aMSFT)<- c("T+1","T+2","T+3","T+4","T+5")
  }
}
results2b*capital
##ii)
#using the Diagonal Vectorization (VEC) model different models to choose from
MVAR <- function(gp,gq,p,q,data1,data2,variance.model="sGARCH",distribution.model = "norm"){
  
  specVAR <- rugarch::ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(gp,gq)),
                                 mean.model=list(armaOrder=c(p,q), 
                                                 include.mean=TRUE),distribution.model =distribution.model)
  fitVAR11 <- rugarch::ugarchfit(data = data1, spec = specVAR)
  fitVAR22 <- rugarch::ugarchfit(data = data2, spec = specVAR)
  fitVAR12 <- rugarch::ugarchfit(data = data1*data2, spec = specVAR)
  forecastVAR11 <- rugarch::ugarchforecast(fitVAR11,n.ahead = 5)
  forecastVAR22 <- rugarch::ugarchforecast(fitVAR22,n.ahead = 5)
  forecastVAR12 <- rugarch::ugarchforecast(fitVAR12,n.ahead = 5)
  results <- matrix(nrow =5,ncol=6)  
  partial <- matrix(ncol=3,nrow=5)
  if (distribution.model == "norm"){
    for(i in 1:5){
      VAR11 <- -(forecastVAR11@forecast$seriesFor[i] + qnorm(p = c(0.9,0.95,0.99),mean =0,sd=1)*forecastVAR11@forecast$sigmaFor[i])
      VAR22 <- -(forecastVAR22@forecast$seriesFor[i] + qnorm(p = c(0.9,0.95,0.99),mean =0,sd=1)*forecastVAR22@forecast$sigmaFor[i])
      cor <- forecastVAR12@forecast$sigmaFor[i]/(sqrt(forecastVAR11@forecast$sigmaFor[i])*sqrt(forecastVAR12@forecast$sigmaFor[i]))
      partial[i,] <- -sqrt(VAR11^2 + VAR22^2 + cor*VAR11*VAR22)
      if (i == 1){
        VaR_1 <- partial[i,]
        results1 <- matrix(c(VaR_1,rep(c(NA,NA,NA),4)),byrow = TRUE,nrow=5,ncol=3)
      }
      if (i == 5){
        results <- cbind(results1,partial) 
        colnames(results)<- c("VaR_10%_1","VaR_5%_1","VaR_1%_1","VaR_10%_5","VaR_5%_5","VaR_1%_5")
        rownames(results)<- c("T+1","T+2","T+3","T+4","T+5")
      }
    }
  capital*results
  } 
  else if (distribution.model == "std"){
    for(i in 1:5){
      VAR11 <- -(forecastVAR11@forecast$seriesFor[i] + forecastVAR11@forecast$sigmaFor[i]*qdist(distribution='std', shape=fitVAR11@fit[["coef"]][["shape"]], p=c(0.90,0.95,0.99)))
      VAR22 <- -(forecastVAR22@forecast$seriesFor[i] + forecastVAR22@forecast$sigmaFor[i]*qdist(distribution='std', shape=fitVAR22@fit[["coef"]][["shape"]], p=c(0.90,0.95,0.99)))
      cor <- forecastVAR12@forecast$sigmaFor[i]/(sqrt(forecastVAR11@forecast$sigmaFor[i])*sqrt(forecastVAR12@forecast$sigmaFor[i]))
      partial[i,] <- -sqrt(VAR11^2 + VAR22^2 + cor*VAR11*VAR22)
      if (i == 1){
        VaR_1 <- partial[i,]
        results1 <- matrix(c(VaR_1,rep(c(NA,NA,NA),4)),byrow = TRUE,nrow=5,ncol=3)
      }
      if (i == 5){
        results <- cbind(results1,partial) 
        colnames(results)<- c("VaR_10%_1","VaR_5%_1","VaR_1%_1","VaR_10%_5","VaR_5%_5","VaR_1%_5")
        rownames(results)<- c("T+1","T+2","T+3","T+4","T+5")
      }
    }
    return(capital*results)
    return(show(fitVAR11))
    return(show(fitVAR22))
    
  } 
  else {
  }
}

MVAR(gp=1,gq=1,p=0,q=0,data1 = dataMSFT,data2 = dataWMT,distribution.model = "norm")
MVAR(gp=1,gq=1,p=1,q=0,data1 = dataMSFT,data2 = dataWMT,distribution.model = "norm")
results <- MVAR(gp=1,gq=1,p=1,q=0,data1 = dataMSFT,data2 = dataWMT,distribution.model = "norm")
stargazer(results, type = "html",out = "C:/Users/eduar/OneDrive/Documents/tables",digits = 1)




