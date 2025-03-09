library(ggplot2)

setwd("C:/Users/jbastos/Desktop/BIGD/2_class_exercises")
data = read.csv("data/Default.csv", header = TRUE, row.names = 1, stringsAsFactors = TRUE)
summary(data)

logistic_reg = glm(default ~ balance + income + student, data = data, family = binomial)
summary(logistic_reg)

# +--- create training and test sets -------------------------------------------------------------+
set.seed(1)
train_index = sample(1:nrow(data), size = 0.7 * nrow(data), replace = FALSE)
data_train  = data[ train_index, ]
data_test   = data[-train_index, ]

# +--- error rate for baseline -------------------------------------------------------------------+
baseline_pred = rep("No", nrow(data_test))
confusion_matrix = table(true = data_test$default, pred = baseline_pred)
confusion_matrix
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Error rate for baseline = ", 1 - accuracy, "\n")

# +--- error rate for logistic regression --------------------------------------------------------+
logistic_reg = glm(default ~ balance + income + student, data = data_train, family = binomial)

logistic_prob = predict(logistic_reg, newdata = data_test, type = "response")
logistic_class = ifelse(logistic_prob < 0.5, "No", "Yes")
confusion_matrix = table(true = data_test$default, pred = logistic_class)
confusion_matrix
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Error rate for logistic regression = ", 1 - accuracy, "\n")

# +-----------------------------------------------------------------------------------------------+
#     TITANIC DATA SET
# +-----------------------------------------------------------------------------------------------+

data = read.csv("data/titanic.csv", header = TRUE, sep = ";")
fix(data)

ggplot(data,aes(x=pclass,group=survived,fill=survived))+
  geom_histogram(position="dodge",binwidth=0.5)+theme_bw()

ggplot(data,aes(x=age,group=survived,fill=survived))+
  geom_histogram(position="dodge",binwidth=5)+theme_bw()

ggplot(data, aes(x=sex,group=survived,fill=survived)) + geom_bar()

# +--- specific examples -------------------------------------------------------------------------+
logit = glm(survived ~ pclass + sex + age, data = data, family = binomial)
summary(logit)

# +--- predictions for specific examples ---------------------------------------------------------+
new_data = data.frame(pclass = 1, sex = "female", age = 15)
predict(logit, newdata = new_data, type = "response")

# +--- create training and test sets -------------------------------------------------------------+
set.seed(1)
train_index = sample(1:nrow(data), size = 0.7 * nrow(data), replace = FALSE)
data_train  = data[ train_index, ]
data_test   = data[-train_index, ]

# +--- error rate for baseline -------------------------------------------------------------------+
classes = rep(0, nrow(data_test))
confusion_matrix = table(true = data_test$survived, pred = classes)
confusion_matrix
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Error rate for baseline = ", 1 - accuracy)

# +--- error rate for logistic regression --------------------------------------------------------+
logit = glm(survived ~ pclass + sex + age, data = data_train, family = binomial)
probabilities = predict(logit, newdata = data_test, type = "response")
classes = ifelse(probabilities < 0.5, 0, 1)
confusion_matrix = table(true = data_test$survived, pred = classes)
confusion_matrix
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Error rate for logistic regression = ", 1 -accuracy)



