install.packages("spdep")
install.packages("rgdal")
install.packages("rgeos")
install.packages("spatialreg")
install.packages("sphet")
install.packages("FactoMineR")
install.packages("corrplot")


rm(list = ls())
getwd()
setwd("C:/Users/eduar/OneDrive/Escritorio/Masterado/Segundo Semestre/Spatial Econometrics/Project/R files")
library("rgdal")
library("corrplot")
spatialdata = readOGR(dsn = ".",layer = 'realestate') #add the shapefile in R
summary(spatialdata) #statistical summary of variables
spplot(spatialdata, "medvalsqm2") #distributional plot of median house prices in Portugal 
cor(x = scale(spatialdata@data[-(1:5)]))
corrplot(cor(x = scale(spatialdata@data[-(1:5)])),method = "shade", type = "full", diag = TRUE,
         tl.col = "black")
hist(spatialdata@data$medvalsqm2, breaks = 20, main = "Histogram of house price value per square meter", xlab = "", ylab =)
help(hist)

spatialdata@data$area <- spatialdata@data$pop / spatialdata@data$popdens #km2
spatialdata@data$newhouseparea <- spatialdata@data$newhouse / spatialdata@data$area #km2
spatialdata@data$classicalHarea <- spatialdata@data$classicalH / spatialdata@data$area 
spatialdata@data$touristparea <- spatialdata@data$touracc / spatialdata@data$area 
spatialdata@data$clasicalAparea <- spatialdata@data$classacc / spatialdata@data$area 
spatialdata@data$unemploymentrate <- spatialdata@data$unemployed / spatialdata@data$active 
spatialdata@data$lnclassicalhh <- log(spatialdata@data$classicalH, base = exp(1))

cor(x = scale(spatialdata@data[-(1:5)]))
?cor



#pca
#library(FactoMineR)
#data <- spatialdata@data[-(1:6)]
#pca <- PCA(data, scale.unit = TRUE)
#pca$eig
#fviz_eig(pca)
#pca$var$cor
#pca$var$cos2
#pca$var$contrib
#spatialdata@data$dim1 <- pca$ind$coord[,1]
#spatialdata@data$dim2 <- pca$ind$coord[,2]

#make a weight matrix
library(spdep)
queen.nb = poly2nb(spatialdata)
queen.listw = nb2listw(queen.nb) #convert nb to listw type so regression commands work 
listw1=queen.listw                

#define regression
reg = medvalsqm2 ~ imi + classicalH + owcharges + charges400 + newhouse + pp + netpop + netmig + pop15 + popeduc15 + popdens + pop + pop65 + active + unemployed + touracc + classacc+ area + newhouseparea + classicalHarea + touristparea + clasicalAparea + unemploymentrate 
options(scipen = 7)
#reg = medvalsqm2 ~ dim1 +dim2


#OLS regression
library(spatialreg)
library(MASS)
reg1 = lm(reg,data = spatialdata) #OLS model
summary(reg1)
reg1aic <- stepAIC(reg1,direction = "backward", trace = TRUE)
summary(reg1aic)

lm.morantest(reg1,listw1) #OLS errors have spatial dependency
moran(spatialdata$medvalsqm2,listw1, n=278, Szero(listw1)) #the output shows spatial dependency in the dependant variabe
lm.LMtests(reg1,listw1,test = "all")
#Both the lm test for lag and error the dependency are statistically significant. 
#However, the robust lm error dependence test is not statistically significant and the robust lm lag dependence test is. this suggest a Lag dependence model
help(lm.LMtests)


#SEM Spatial Error Model y = XB + e + We + u
reg4 = errorsarlm(reg1, spatialdata,listw1)
summary(reg4)

#Spatial Hausman Test
#model1: OLS
#model2: SEM
#The motivation for this type of comparison is that theory indicates OLS and SEM estimates should be the same if the true DGP is either OLS, SEM, or any

Hausman.test(reg4) #rejects the null hypothesis which suggests that neither OLS and SEM are probably the best models

#We do not know if either the dependent variable or regressors should have the dependency or both

#SLX Spatially Lagged X y = XB + WXT + e
reg2 = lmSLX(reg1, spatialdata, listw1)
summary(reg2)

#SAR Spatial Autoregressive model y = pWy + XB + e
reg3 = lagsarlm(reg1,spatialdata,listw1)
summary(reg3)

#SDM Spatial Durbin model  y = pWy + XB + WXT + e
reg5 = lagsarlm(reg1,spatialdata,listw1, type ="mixed")
summary(reg5)

#LR Test : SDM vs SAR
LR.Sarlm(reg5,reg3) #do not restrict the model so model SDM is better
#LR Test : SDM vs SLX
LR.Sarlm(reg5,reg2) #do not restrict the model so model SDM is better

#So our preferred model is SDM which is model 5

#Heterosckedasticity
bptest.Sarlm(reg5,studentize=TRUE) #Heterosckedasticity is present

summary(reg5)

#impacts
impacts(reg5,listw = listw1)
?cor

#save a new shapefile for stata
writeOGR(spatialdata, ".","realestate2",driver = "ESRI Shapefile")