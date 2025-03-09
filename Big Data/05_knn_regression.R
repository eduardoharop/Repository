library(FNN)
rm(list = ls())
setwd("C:/Users/jbastos/Desktop/BIGD/2_class_exercises")

# ==============================================================================
#    READ ADVERTISING DATASET AND SPLIT DATA TRAIN+TEST
# ==============================================================================

df = read.csv("data/Advertising.csv", header = TRUE, row.names = 1, sep = ",")

set.seed(1)
train_index = sample(1:nrow(df), size = floor(nrow(df) * 0.7), replace = FALSE)
df_train  = df[ train_index, ]
df_test   = df[-train_index, ]

# ==============================================================================
#    TRAIN LINEAR MODEL
# ==============================================================================

linear_model = lm(Sales ~ ., data = df_train)
linear_model_pred = predict(linear_model, newdata = df_test)
RMSE_lm = sqrt(mean((linear_model_pred - df_test$Sales)^2))
cat("RMSE for linear model = ", RMSE_lm)

# ==============================================================================
#    TRAIN KNN REGRESSORS
# ==============================================================================

X_train = as.matrix(df_train[,c("TV","Radio","Newspaper")])
X_test  = as.matrix(df_test[,c("TV","Radio","Newspaper")])

RMSE_knn = array(0, c(20))
for(i in 1:20) {
  knn = knn.reg(train = X_train, test = X_test, y = df_train$Sales, k = i)
  RMSE_knn[i] = sqrt(mean((knn$pred - df_test$Sales)^2))
  cat("k =", i, "\t", rmse[i],"\n")
}
plot(RMSE_knn, xlab = "Number of nearest neighbors", ylab = "RMSE", pch = 20, col = "red")
abline(RMSE_lm, 0, lty = 2)

# ==============================================================================
#    READ BIKE SHARING DATASET
# ==============================================================================

df = read.csv("data/bike_sharing_day.csv", header = TRUE, row.names = 1, sep = ",")
df = df[,c("cnt","season","mnth","holiday","weekday","workingday","weathersit","temp","hum","windspeed","yr")]
formula = as.formula(cnt ~ season + mnth + holiday + weekday + weathersit + temp + hum + windspeed + yr)

df$season     = factor(df$season)
df$mnth       = factor(df$mnth)
df$holiday    = factor(df$holiday)
df$weekday    = factor(df$weekday)
df$workingday = factor(df$workingday)
df$weathersit = factor(df$weathersit)

# ==============================================================================
#    TRAIN-TEST SPLIT
# ==============================================================================

set.seed(1)
train_index = sample(1:nrow(df), size = floor(nrow(df) * 0.7), replace = FALSE)
df_train  = df[ train_index, ]
df_test   = df[-train_index, ]

# ==============================================================================
#    TRAIN LINEAR MODEL
# ==============================================================================

linear_model = lm(formula, data = df_train)
linear_model_pred = predict(linear_model, newdata = df_test)
RMSE_lm = sqrt(mean((linear_model_pred - df_test$cnt)^2))
cat("RMSE for linear model = ", RMSE_lm)

# ==============================================================================
#    TRAIN KNN REGRESSORS
# ==============================================================================

X_train = model.matrix(formula, df_train)[,-1]
X_test  = model.matrix(formula, df_test)[,-1]

RMSE_knn = array(0, c(30))
for(i in 1:30) {
  knn = knn.reg(train = X_train, test = X_test, y = df_train$cnt, k = i)
  RMSE_knn[i] = sqrt(mean((knn$pred - df_test$cnt)^2))
  cat("k =", i, "\t", RMSE_knn[i],"\n")
}
plot(RMSE_knn, xlab = "Number of nearest neighbors", ylab = "RMSE", pch = 19, col = "red", ylim=c(600, 1400))
abline(rmse_lm, 0, lty = 2)

