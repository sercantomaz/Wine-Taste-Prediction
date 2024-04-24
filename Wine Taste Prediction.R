## Wine Quality Prediction using Random Forest, XGBoost, Decision Trees & KNN
# load the necessary libraries
library(caTools)
library(mice)
library(ggplot2)
library(dplyr)
library(corrgram)
library(corrplot)
library(randomForest)
library(caret)
library(vip)
library(class)
library(rpart)
library(rpart.plot)
library(xgboost)

# load the data set
df <- read.csv("wine.csv")

# str of the df & correctting the classes
str(df)
df$taste <- as.factor(df$taste)

# removing the quality column due to its strong correlation with taste
df <- df [, -12]

# check the missing values
md.pattern(df) # completely observed

# check the taste col
table(df$taste) # upon examining the confusion matrices in my dataset, it appears that sensitivity and specificity values may be unavailable (NA) due to the dataset's lack of a balanced distribution.

# correlation graph
corr.matrix <- cor(df [,-12])
corrplot(corr.matrix, method = "number")

# test & train split
split <- sample.split(df$taste, SplitRatio = 0.8)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)

# dim of the test set
dim(test)

# dim of the train set
dim(train)

## RANDOM FOREST
# creating parameter grid for random forest model
param_grid <- expand.grid(mtry = c(1, 2, 3, 4)) # values to try for mtry

# setting cross-validation parameters with the control function
control <- trainControl(method = "cv", number = 5) # 5 fold cross-validation

# select the best parameters for random forest
rf.model <- train(taste~.,
                  data = train,
                  method = "rf",
                  trControl = control,
                  tuneGrid = param_grid)

# best tune
rf.model$bestTune

# building a rf with best parameters
rf.model.param <- randomForest(taste~., train, mtry = rf.model$bestTune$mtry, ntree = 30)
rf.model.param

# predictions of the RF model
rf.preds <- predict(rf.model.param, test)

# confusion matrix of the RF model
cm.rf <- confusionMatrix(test$taste, rf.preds)
cm.rf
cm.rf$byClass

# variable importance
vip(rf.model.param)

# optimum number of tree
error.rate <- data.frame(
  Trees = rep(1:nrow(rf.model.param$err.rate)),
  Error = rf.model.param$err.rate [, "OOB"]
)

# min error
minerrorpts <- error.rate$Trees [error.rate$Error == min(error.rate$Error)]

# plotting to determine optimal number of trees
ggplot(error.rate, aes(x = Trees, y = Error)) +
  geom_line() +
  geom_vline(xintercept = minerrorpts [1])

## KNN
# test & train set without labels
test.knn <- test [1:11] # test set without label to be used in knn classification
train.knn <- train [1:11] # train set without label to be used in knn classification

# building a knn model with k = 1
knn.classifier <- knn(train = train.knn, # train set without label used
                      test = test.knn, # test set without label used
                      cl = train$taste, # train set with label used
                      k = 1 )

# choosing the k value
classifier.knn = NULL
error.rate.knn = NULL
for (i in 1:10) {
  classifier.knn <- knn(train = train.knn,
                        test = test.knn,
                        cl = train$taste,
                        k = i)
  error.rate.knn [i] <- mean(test$taste != classifier.knn) 
}
print(error.rate.knn)

# error rate knn vs values of K
k.values <- 1:10
error.rate.knn.k <- data.frame(error.rate.knn, k.values)

# elbow method to determine optimum k value
ggplot(error.rate.knn.k, aes(x = k.values, y = error.rate.knn)) +
  geom_point() +
  geom_line(lty = "dotted", color = "red") +
  scale_x_continuous(breaks = seq(1, 10, by = 1))

# confusion matrix of the KNN model
table(knn.classifier, test$taste)

# error rate of the best KNN model
misclasserror.knn <- mean(test$taste != classifier.knn)
accuracy.knn <- 1 - misclasserror.knn
accuracy.knn 

## CLASSIFICATION TREE
# define the parameter grid
param_grid.ct <- expand.grid(
  cp = seq(0.01, 1, by = 0.01))

# control parameters
control.ct <- trainControl(method = "cv", number = 5)

# best parameters via cross validation
ct.model <- train(taste~., data = train,
                  method = "rpart", 
                  trControl = control.ct,
                  tuneGrid = param_grid.ct)

# best parameters
ct.model$bestTune

# building the final model
ct.final.model <- rpart(
  taste~.,
  data = train,
  cp = ct.model$bestTune$cp
)

# Cross-validation Error vs. Complexity Parameter (CP)
plotcp(ct.final.model)

# classification tree
print(prp(ct.final.model))

# predictions of the CT model
ct.preds <- predict(ct.final.model, newdata = test, type = "class")

# confusion matrix of the CT model
confusionMatrix(test$taste, ct.preds)

## XGBOOST
# grid search
param_grid_xgb <- expand.grid(
  nrounds = c(50, 100, 200, 500),
  max_depth = c(2, 4, 6),
  eta = seq(0.1, 0.3, by = 0.1),
  gamma = c(1, 3, 5),
  min_child_weight = c(2, 5, 7),
  subsample = 0.5,
  colsample_bytree = 0.5
)

# train control
control.xgb <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# xgb tune
xgb.tune <- train(
  x = train [, -12],
  y = train [, 12],
  trControl = control.xgb,
  tuneGrid = param_grid_xgb,
  verbose = TRUE,
  method = "xgbTree"
)

# best tune
xgb.tune$bestTune

# writing out the best model
control.xgb <- trainControl(method = "none",
                            verboseIter = TRUE,
                            allowParallel = TRUE
)

# final grid with the best values
final.grid <- expand.grid(nrounds = xgb.tune$bestTune$nrounds,
                          max_depth = xgb.tune$bestTune$max_depth,
                          gamma = xgb.tune$bestTune$gamma,
                          colsample_bytree = xgb.tune$bestTune$colsample_bytree,
                          min_child_weight = xgb.tune$bestTune$min_child_weight,
                          subsample = xgb.tune$bestTune$subsample,
                          eta = xgb.tune$bestTune$eta) 

# building the model with the best parameters
xgb.model <- train(
  x = train [, -12],
  y = train [, 12],
  trControl = control.xgb,
  tuneGrid = final.grid,
  method = "xgbTree",
  verbose = TRUE
)

# predictions of the XGB model
xgb.preds <- predict(xgb.model, test)

# confusion matrix of the XGB model
confusionMatrix(xgb.preds, test$taste)
