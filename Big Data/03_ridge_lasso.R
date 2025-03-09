rm(list = ls())
setwd("C:/Users/jbastos/Desktop/BIGD/2_class_exercises")
library(glmnet)
library(lmtest)
library(sandwich)
library(ISLR)

# +------------------------------------------------------------------------------------------------
#     LINEAR REGRESSION HAS HIGH VARIANCE WHEN p ~ n
# +------------------------------------------------------------------------------------------------

set.seed(1)
npoints = 4
ntrials = 10
x = 1:npoints + rnorm(npoints, 0, 0.25)
Sys.sleep(2)
for(i in 1:ntrials) {
  y = x + 1 + rnorm(npoints, 0, 1.5)
  plot(x, y, pch = 16, col="darkorange", xlim=c(0,5), ylim=c(0,10))
  curve(x + 1, lwd = 3, col = "grey", add = TRUE)
  lm = lm(y ~ x)
  curve(lm$coefficients[1] + lm$coefficients[2] * x, lwd = 3, col = "black", add = TRUE)
  Sys.sleep(2)
}

# -------------------------------------------------------------------------------------------------
#    REGULARIZED LINEAR MODELS (RIDGE REGRESSION AND LASSO)
# -------------------------------------------------------------------------------------------------

#   BIKE SHARING DATASET
#   Two-year (2011-2012) historical log from a bikeshare system, Washington D.C.
#
# - dteday: date
# - season: season (1:winter, 2:spring, 3:summer, 4:fall)
# - yr: year (0: 2011, 1:2012)
# - mnth: month (1 to 12)
# - holiday: whether day is holiday or not
# - weekday: day of the week
# - workingday: if day is neither weekend nor holiday is 1, otherwise is 0.
# - weathersit: 1 - Clear, Few clouds, Partly cloudy; 2 - Mist + Cloudy, Mist + Broken clouds,
#               3 - Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain.
# - temp: Normalized temperature in Celsius.
# - atemp: Normalized feeling temperature in Celsius.
# - hum: Normalized humidity.
# - windspeed: Normalized wind speed.
# - casual: count of casual users.
# - registered: count of registered users.
# - cnt: count of total rental bikes including both casual and registered.

data = read.csv("data/bike_sharing_day.csv", header = TRUE, row.names = 1, sep = ",")
fix(data)
formula = as.formula(cnt ~ season + mnth + holiday + weekday + weathersit + temp + hum + windspeed + yr)

data$season     = factor(data$season)
data$mnth       = factor(data$mnth)
data$holiday    = factor(data$holiday)
data$weekday    = factor(data$weekday)
data$workingday = factor(data$workingday)
data$weathersit = factor(data$weathersit)

plot(data$cnt, pch = 19, cex = 0.5, col = "darkorange")
plot(data$cnt ~ data$season)
plot(data$cnt ~ data$mnth)
plot(data$cnt ~ data$holiday)
plot(data$cnt ~ data$weekday)
plot(data$cnt ~ data$weathersit)
plot(data$temp, data$cnt, pch = 19, cex = 0.5, col = "darkorange")
plot(data$windspeed, data$cnt, pch = 19, cex = 0.5, col = "darkorange")
plot(data$hum, data$cnt, pch = 19, cex = 0.5, col = "darkorange")

linear_model = lm(formula, data = data)
summary(linear_model)

linear_model_pred = predict(linear_model) 
plot(data$cnt - linear_model_pred, pch = 19, col = "darkorange") # Check residuals


# === CREATE TRAIN AND TEST SETS ==================================================================
set.seed(1)
train_index = sample(1:nrow(data), size = floor(nrow(data) * 2 / 3), replace = FALSE)
data_train  = data[ train_index, ]
data_test   = data[-train_index, ]
X_train = model.matrix(formula, data_train)[,-1]
X_test  = model.matrix(formula, data_test)[,-1]
y_train = data_train$cnt
y_test  = data_test$cnt

# === LINEAR MODEL ================================================================================
linear_model = lm(formula, data = data_train)
summary(linear_model)

# === REGULARIZED MODEL ===========================================================================
lambda = 100
regularized_reg = glmnet(X_train, y_train, alpha = 1, lambda = lambda) # alpha = 0: ridge; alpha = 1: lasso
regularized_coef = predict(regularized_reg, type = "coefficients", s = lambda)
regularized_coef

# === OUT-OF-SAMPLE PREDICTION ACCURACY: MSE ======================================================
linear_model_P = predict(linear_model, newdata = data_test)
cat("RMSE for OLS = ", sqrt(mean((linear_model_P - y_test)^2)), "\n")
cv_lambda = cv.glmnet(X_train, y_train, alpha = 0)$lambda.min
ridge_reg = glmnet(X_train, y_train, alpha = 0, lambda = cv_lambda)
ridge_reg_P = predict(ridge_reg, newx = X_test, s = cv_lambda)
cat("RMSE for Ridge = ", sqrt(mean((ridge_reg_P - y_test)^2)), "\n")
cv_lambda = cv.glmnet(X_train, y_train, alpha = 1)$lambda.min
lasso_reg = glmnet(X_train, y_train, alpha = 1, lambda = cv_lambda)
lasso_reg_P = predict(lasso_reg, newx = X_test, s = cv_lambda)
cat("RMSE for Lasso = ", sqrt(mean((lasso_reg_P - y_test)^2)), "\n")

# =================================================================================================
#     RIDGE REGRESSION AND THE LASSO: HITTERS DATA SET
# =================================================================================================

sum(is.na(Hitters$Salary))
Hitters = na.omit(Hitters) # remove obs. without salary info
nrow(Hitters)
fix(Hitters)

X = model.matrix(Salary ~ ., Hitters)[,-1]
y = Hitters$Salary

set.seed(1)
train_index = sample(1:nrow(X), size = nrow(X) * 2 / 3, replace = FALSE)
X_train  = X[ train_index, ]
X_test   = X[-train_index, ]
y_train  = y[ train_index]
y_test   = y[-train_index]

linear_reg   = lm(y_train ~ ., data = data.frame(cbind(y_train, X_train)))
linear_reg_P = predict(linear_reg, newdata = data.frame(X_test))
cat("RMSE for OLS = ", sqrt(mean((linear_reg_P - y_test)^2)), "\n")
cv_lambda = cv.glmnet(X_train, y_train, alpha = 0)$lambda.min
ridge_reg = glmnet(X_train, y_train, alpha = 0, lambda = cv_lambda)
ridge_reg_P = predict(ridge_reg, newx = X_test, s = cv_lambda)
cat("RMSE for Ridge = ", sqrt(mean((ridge_reg_P - y_test)^2)), "\n")
cv_lambda = cv.glmnet(X_train, y_train, alpha = 1)$lambda.min
lasso_reg = glmnet(X_train, y_train, alpha = 1, lambda = cv_lambda)
lasso_reg_P = predict(lasso_reg, newx = X_test, s = cv_lambda)
cat("RMSE for Lasso = ", sqrt(mean((lasso_reg_P - y_test)^2)), "\n")

