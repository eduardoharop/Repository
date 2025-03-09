library(rpart)
library(rattle)
setwd("C:/Users/jbastos/Desktop/BIGD/2_class_exercises/")

# +------------------------------------------------------------------------------------------------+
#     BIKE SHARING DATASET
# +------------------------------------------------------------------------------------------------+

data = read.csv("data/bike_sharing_day.csv", header = TRUE, row.names = 1, sep = ",")
formula = as.formula(cnt ~ season + mnth + holiday + weekday + weathersit + temp + hum + yr + windspeed)

set.seed(1)
train_index = sample(1:nrow(data), size = floor(nrow(data) * 0.7), replace = FALSE)
data_train  = data[ train_index, ]
data_test   = data[-train_index, ]

tree = rpart(formula, data = data_train, method = "anova")
fancyRpartPlot(tree)
predicted_count_tree = predict(tree, newdata = data_test)

linear_model = lm(formula, data = data_train)
predicted_count_lm = predict(linear_model, newdata = data_test)
cat("RMSE for Tree = ", sqrt(mean((predicted_count_tree - data_test$cnt)^2)), "\n")
cat("RMSE for OLS = ", RMSE_lm = sqrt(mean((predicted_count_lm - data_test$cnt)^2)), "\n")

# +------------------------------------------------------------------------------------------------+
#     WINE QUALITY DATASET
# +------------------------------------------------------------------------------------------------+

data = read.csv("data/winequality-white.csv", header = TRUE, sep = ";")

set.seed(1)
train_index = sample(1:nrow(data), size = floor(nrow(data) * 0.7), replace = FALSE)
data_train  = data[ train_index, ]
data_test   = data[-train_index, ]

tree = rpart(quality ~ ., data = data_train, method = "anova")
fancyRpartPlot(tree)

predicted_quality = predict(tree, newdata = data_test)
linear_model = lm(quality ~ ., data = data_train)
summary(linear_model)
predicted_quality_ols = predict(linear_model, newdata = data_test)

cat("RMSE for Tree = ", sqrt(mean((predicted_quality - data_test$quality)^2)), "\n")
cat("RMSE for OLS = ", sqrt(mean((predicted_quality_ols - data_test$quality)^2)), "\n")



