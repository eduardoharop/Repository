library(rpart)
library(randomForest)

# ----------------------------------------------------------------------------------------------------
# Letter Recognition Data Set
# The objective is to identify each of a large number of black-and-white rectangular pixel displays
# as one of the 26 capital letters in the English alphabet. The character images were based on 20
# different fonts and each letter within these 20 fonts was randomly distorted to produce a file
# of 20,000 unique stimuli. Each stimulus was converted into 16 primitive numerical attributes
# (statistical moments and edge counts) which were then scaled to fit into a range of integer
# values from 0 through 15.
# ----------------------------------------------------------------------------------------------------

setwd("C:/Users/jbastos/Desktop/BIGD/2_class_exercises")
data = read.csv("data/letterdata.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)

data_train = data[1:16000, ]
data_test  = data[16001:20000, ]

# === Classification tree ===

tree = rpart(letter ~ ., data = data_train, method = "class")
yhat_tree = predict(tree, newdata = data_test,type = "class")
table(data_test$letter, yhat_tree)

correct = (data_test$letter == yhat_tree)
table(correct)
prop.table(table(correct))

# === Random Forest ===

rf <- randomForest(letter ~ ., data = data_train, ntree = 500)
yhat_rf = predict(rf, newdata = data_test)
table(data_test$letter, yhat_rf)

correct = (data_test$letter == yhat_rf)
table(correct)
prop.table(table(correct))

