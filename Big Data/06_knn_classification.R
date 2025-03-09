library(FNN)

# ==============================================================================
#    READ TITANIC DATASET
# ==============================================================================

setwd("C:/Users/jbastos/Desktop/BIGD/2_class_exercises")
df = read.csv("data/titanic.csv", header = TRUE, sep = ";", stringsAsFactors = TRUE)
df = df[,c("survived","pclass","sex","age")]
df = df[complete.cases(df), ]
df$sex = as.numeric(df$sex)

# ==============================================================================
#    TRAIN-TEST SPLIT
# ==============================================================================

set.seed(1)
train_index = sample(1:nrow(df), size = nrow(df) * 0.7, replace = FALSE)
df_train = df[ train_index, ]
df_test  = df[-train_index, ]

# ==============================================================================
#    TRAIN LOGISTIC REGRESSION
# ==============================================================================

log_reg = glm(survived ~ pclass + sex + age, data = df_train, family = binomial)
classes = ifelse(predict(log_reg, newdata = df_test, type = "response") < 0.5, 0, 1)
confusion_matrix = table(true = df_test$survived, pred = classes)
confusion_matrix
error_rate_log = 1 - sum(diag(confusion_matrix)) / sum(confusion_matrix)

# ==============================================================================
#    TRAIN KNN CLASSIFIER
# ==============================================================================

X_train = df_train[ ,-1]
X_test = df_test[ ,-1]
X_train = scale(X_train)
X_test = scale(X_test)

error_rate = array(0, c(30))
for(i in 1:30) {
  classes = knn(train = X_train, test = X_test, cl = df_train$survived, k = i)
  classes = knn(train = X_train, test = X_test, cl = df_train$survived, k = i)
  confusion_matrix = table(true = df_test$survived, pred = classes)
  error_rate[i] = 1 - sum(diag(confusion_matrix)) / sum(confusion_matrix)
  cat("k =", i, "\t", error_rate[i],"\n")
}

plot(error_rate, xlab = "Number of nearest neighbors", ylab = "Error rate", pch = 19, col = "red", ylim=c(0, 0.3))
abline(error_rate_log, 0, lty = 2)


