library(rpart)
library(rattle)

setwd("C:/Users/jbastos/Desktop/BIGD/2_class_exercises/")
data = read.csv("data/titanic.csv", header = TRUE, sep = ";")

set.seed(1)
train_index = sample(1:nrow(data), size = 1000, replace = FALSE)
data_train  = data[ train_index, ]
data_test   = data[-train_index, ]

tree = rpart(survived ~ pclass + sex + age, data = data_train, method = "class")

printcp(tree)
summary(tree)
fancyRpartPlot(tree)

classes = predict(tree, newdata = data_test, type = "class")
confusion_matrix = table(true = data_test$survived, pred = classes)
confusion_matrix
cat("Error rate =", 1 - sum(diag(confusion_matrix)) / sum(confusion_matrix), "\n")

logit = glm(survived ~ pclass + sex + age, data = data_train, family = binomial)
classes = ifelse(predict(logit, newdata = data_test, type = "response") < 0.5, 0, 1)
confusion_matrix = table(true = data_test$survived, pred = classes)
confusion_matrix
cat("Error rate =", 1 - sum(diag(confusion_matrix)) / sum(confusion_matrix), "\n")
