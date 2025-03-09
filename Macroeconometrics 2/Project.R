options(digits = 6, width = 120)
rm(list=ls())
dev.off()

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
library(vars)
library(bruceR)
library(rempsyc)

##################################################################################################################
##PART 1

##Dataset creation
  df_init <- tidyquant::tq_get(c("BTC-USD","^IXIC","^DJI","^SPX"),get = "stock.prices" ,from="2014-09-18", to="2023-12-15") %>%
  as_tsibble(index = date,key = symbol) %>% 
  group_by_key() %>%
  mutate(rtn = difference(log(close))) %>% 
  drop_na() %>% 
  group_by_key() %>% 
  mutate(trading_day = row_number()) 

dates <- df_init %>% dplyr::filter(symbol == "^DJI")  %>% as.data.frame() %>% dplyr::select(date)

BTC <- df_init %>% 
      dplyr::filter(symbol == "BTC-USD") %>% 
      dplyr::filter(date %in% dates$date) %>% 
      mutate(trading_day = row_number())

df <- df_init %>% dplyr::filter(symbol == "^IXIC" | symbol == "^DJI" |symbol == "^SPX") %>% rbind(BTC)

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
##descriptve statistics on logreturns
descriptive <- df %>%
  features(rtn,list(n = ~length(.), min = ~min(.), "1stQ"= ~quantile(.)[2],mean = ~mean(.),median = ~median(.),"3rdQ" = ~quantile(.)[4],max = ~max(.),skew =timeSeries::colSkewness, kurt =  timeSeries::colKurtosis 
  )) 
nice_descriptive <- nice_table(descriptive)
print(nice_descriptive,preview = "docx")


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
histograms <- ggplot(df, aes(x=rtn,y = ..density..))+
  geom_histogram(bins = 100)+ geom_density() +
  facet_grid(vars(symbol), scale = "free_y" ) +  
  theme(legend.position="none", axis.text=element_text(size=6),
        axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

ggplot(df, aes(x=symbol, y = rtn)) +
  geom_boxplot() 
theme(legend.position="none", axis.text=element_text(size=6),
      axis.title.x = element_text(size = 6), axis.title.y = element_text(size = 6))

#################################


df_var <- df %>% dplyr::select(rtn) %>% pivot_wider(names_from=symbol,values_from=rtn) %>% as.data.frame()
df_var <- df_var %>% dplyr::select(-date)
df_var <- df_var %>% ts(frequency = 365)

##descriptve statistics on logreturns
corr <- df_var %>% cor()
nice_corr <- nice_table(corr)
print(nice_corr,preview = "docx")


ADFtestBTC <- df_var %>% 
  as_tibble() %>% 
  dplyr::select("BTC-USD") %>%
  ts() %>% 
  adf.test()


ADFtestDJI <- df_var %>% 
  as_tibble() %>% 
  dplyr::select("^DJI") %>%
  ts() %>% 
  adf.test()

ADFtestIXIC <- df_var %>% 
  as_tibble() %>% 
  dplyr::select("^IXIC") %>%
  ts() %>% 
  adf.test()

ADFtestSPX <- df_var %>% 
  as_tibble() %>% 
  dplyr::select("^SPX") %>%
  ts() %>% 
  adf.test()


VARselect(df_var,lag.max=20,type="cons")[["selection"]]
varselect<-VARselect(df_var,lag.max=10,type="cons")[["criteria"]]
print(nice_table(varselect),preview = "docx")

VARest <- VAR(df_var, p = 4, type = "cons", season = NULL)
summary(VARest)
print(nice_table(VARest[["varresult"]][["BTC.USD"]][["coefficients"]]),preview = "docx")
print(nice_table(VARest[["varresult"]][["X.DJI"]][["coefficients"]]),preview = "docx")
print(nice_table(VARest[["varresult"]][["X.IXIC"]][["coefficients"]]),preview = "docx")
print(nice_table(VARest[["varresult"]][["X.SPX"]][["coefficients"]]),preview = "docx")

granger <- granger_causality(VARest)
print(nice_table(granger[["result"]]),preview = "docx")

ts(resid(VARest), frequency = 365) %>% forecast::ggAcf(lag.max = 10)

  serial.test(VARest, lags.bg = 1, type = "BG")

vars::normality.test(VARest, multivariate.only = TRUE)

for (j in 1:4){
  show(vars::arch.test(VARest, lags.multi = j, multivariate.only = TRUE))
}

plot(stability(VARest, type = "OLS-CUSUM"))

##amat
a.mat <- diag(4)
diag(a.mat) <- NA
a.mat[2, 1] <- NA
a.mat[3, 1] <- NA
a.mat[4, 1] <- NA
a.mat[3, 2] <- NA
a.mat[4, 2] <- NA
a.mat[4, 3] <- NA
print(a.mat)

##bmat
b.mat <- diag(4)
diag(b.mat) <- NA
print(b.mat)

##SVAR
SVAR <- SVAR(VARest, Amat = a.mat, Bmat = b.mat, max.iter = 100000, hessian = TRUE,lrtest = FALSE)
SVAR

##IRF
BTC_BTC <- irf(SVAR, response = "BTC.USD", impulse = "BTC.USD", 
               n.ahead = 40, ortho = TRUE, boot = TRUE)
plot(BTC_BTC)

BTC_DJI <- irf(SVAR, response = "X.DJI", impulse = "BTC.USD", 
               n.ahead = 40, ortho = TRUE, boot = TRUE)
plot(BTC_DJI)

BTC_IXIC <- irf(SVAR, response = "X.IXIC", impulse = "BTC.USD", 
               n.ahead = 40, ortho = TRUE, boot = TRUE)
plot(BTC_IXIC)

BTC_SPX <- irf(SVAR, response = "X.SPX", impulse = "BTC.USD", 
                n.ahead = 40, ortho = TRUE, boot = TRUE)
plot(BTC_SPX)

