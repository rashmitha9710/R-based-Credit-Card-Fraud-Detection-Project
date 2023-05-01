library(ranger)
library(caret)
library(data.table)
install.packages("pROC")
library(pROC)
install.packages("rpart")
library(rpart)
library(rpart.plot)
library(neuralnet)


creditcard_df <- read.csv("creditcard.csv")

dim(creditcard_df)
head(creditcard_df,10)

tail(creditcard_df,10)

table(creditcard_df$Class)
summary(creditcard_df)
names(creditcard_df)
var(creditcard_df$Amount)
sd(creditcard_df$Amount)
head(creditcard_df)
creditcard_df$Amount=scale(creditcard_df$Amount)
newcredit_df=creditcard_df[,-c(1)]
head(newcredit_df)

#Data Modelling
set.seed(1)
## partitioning into training (60%) and validation (40%)
# randomly sample 60% of the row IDs for training; the remaining 40% serve as
# validation
train.rows <- sample(rownames(newcredit_df), dim(newcredit_df)[1]*0.6)
# collect all the columns with training row ID into training set:
train.data <- newcredit_df[train.rows, ]
dim(train.data)
# collect all the columns without training row ID into validation set
test.rows <- setdiff(rownames(newcredit_df), train.rows)
test.data <- newcredit_df[test.rows, ]
dim(test.data)

Logistic_Model=glm(Class~.,test.data,family=binomial())
summary(Logistic_Model) 
par(mar=c(1,1,1,1))
plot(Logistic_Model)

lr.predict <- predict(Logistic_Model,train.data, probability = TRUE)
auc.gbm = roc(train.data$Class, lr.predict, plot = TRUE, col = "blue")



decisionTree_Model <- rpart(Class ~ . , creditcard_df, method = 'class')
predicted_val <- predict(decisionTree_Model, creditcard_df, type = 'class')
probability <- predict(decisionTree_Model, creditcard_df, type = 'prob')
rpart.plot(decisionTree_Model)


NN.model =neuralnet (Class~.,train.data,linear.output=FALSE)
plot(NN.model)

pred.NN=compute(NN.model,test.data)
result.NN=pred.NN$net.result
result.NN=ifelse(result.NN>0.5,1,0)

plot(result.NN)
library(gbm, quietly=TRUE)

# Get the time to train the GBM model
system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train.data, test.data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train.data) / (nrow(train.data) + nrow(test.data))
  )
)
# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")

model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)
#Plot the gbm model

plot(model_gbm)

# Plot and calculate AUC on test data
gbm_test = predict(model_gbm, newdata = test.data, n.trees = gbm.iter)
gbm_auc = roc(test.data$Class, gbm_test, plot = TRUE, col = "red")
print(gbm_auc)
