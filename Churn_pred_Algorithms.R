
# Importing the dataset
df <- read.csv("Telco-Customer-Churn.csv")
head(df)
table(df$Churn)

# Data preprocessing
## Deleting customerID as it will be unique for every customer
df$customerID <- NULL

## Factorising the attributes
df$Churn <- as.factor(df$Churn)
df$gender <- as.factor(df$gender)
df$SeniorCitizen <- as.factor(df$SeniorCitizen)
df$Partner <- as.factor(df$Partner)
df$Dependents <- as.factor(df$Dependents)
df$PhoneService <- as.factor(df$PhoneService)
df$MultipleLines <- as.factor(df$MultipleLines)
df$InternetService <- as.factor(df$InternetService)
df$Contract <- as.factor(df$Contract)
df$PaymentMethod <- as.factor(df$PaymentMethod)

## Removing rows with tenure less than 1 month
df <- subset(df,tenure>0)

## Ordinal discretisation for field SeniorCitizen
df$SeniorCitizen <- as.factor(plyr::mapvalues(df$SeniorCitizen,
                    from = c("0", "1"), to = c("No", "Yes")))

# Dividing the dataset into training and test dataset
library(caret)
set.seed(1234) #To get reproducible result
trainIndex <- createDataPartition(df$Churn, p = 0.7, list = FALSE)
train_dataset <- df[trainIndex, ]
test_dataset <- df[-trainIndex, ]

########################## ALGORITHMS ############################

############## 1. Decision Tree

# Loading required libraries
#library(party)      # has a method ctree() which is used to create and analyze decision tree
library(rpart)       # has rpart() as a method to create decision tree
library(rpart.plot)  # for plotting the decision tree

# Building the decision tree
tree <- rpart(Churn~., data = train_dataset)

# Predict test data
test_predict <- predict(tree,test_dataset)
head(test_predict)
yes_no <- function(no, yes) {
  if (yes>no) {
    return("Yes")
  }
  else {
    return("No")
  }
}
test_dataset$Pred = test_dataset$Churn
## Prediction for test data
for (i in 1:nrow(test_predict)) {
  msg = yes_no(test_predict[i,1], test_predict[i,2])
  #test_predict[i] = msg
  test_dataset$Pred[i]=msg
}
## Creating confusion matrix and misclassification errors
xtab <- table(test_dataset$Churn,test_dataset$Pred)
## Plotting Decision Tree
rpart.plot(tree)
## Accuracy
library("caret")
caret::confusionMatrix(xtab)

############## 2. SVM

# Load required packages
library(e1071)  # For SVM
library(caret)  # For confusion matrix

# Train the SVM model
svm_model <- svm(Churn ~ ., data = train_dataset, kernel = "linear", cost = 10)

# Make predictions on the test set
predictions <- predict(svm_model, newdata = test_dataset)

# Evaluate the model's performance
confusionMatrix(predictions, test_dataset$Churn)


############## 3. Logistic Regression

# Load required packages
library(caTools)  # For Logistic regression
library(ROCR)     # For ROC curve to evaluate model
library(caret)    # For confusion matrix

# Train the logistic model
logistic_model <- glm(Churn ~ .,
                      data = train_dataset,
                      family = "binomial")
logistic_model

# Summary
summary(logistic_model)

# Predict test data based on logistic model
predict_reg <- predict(logistic_model,
                       test_dataset, type = "response")
predict_reg

# Changing probabilities
predict_reg <- ifelse(predict_reg >0.5, 1, 0)
predict_reg<-as.factor(predict_reg)
# Evaluating model accuracy using confusion matrix
table(test_dataset$Churn, predict_reg)
confusionMatrix(predict_reg,as.factor(test_dataset$Churn))

############## 4. Random Forest

# Load required packages
library(caTools)
library(ROCR)
library(randomForest)  # For implementing random forest algorithm

# Fitting Random Forest to the train dataset
set.seed(120)  # Setting seed
classifier_RF = randomForest(Churn~.,
                             data = train_dataset,
                             ntree = 500)

classifier_RF

# Predicting the Test set results
y_pred = predict(classifier_RF, newdata = test_dataset)
y_pred
confusionMatrix(y_pred,as.factor(test_dataset$Churn))

# Plotting model
plot(classifier_RF)

# Importance plot
importance(classifier_RF)

# Variable importance plot
varImpPlot(classifier_RF)


############## 5. Under sampling on decision tree

# Required library
library(ROSE)

original = df
table(original$Churn)

# Define the under-sampling method
sampled_data <- ovun.sample(Churn ~ ., data = df, method = "under")$data
df <- as.data.frame(sampled_data)

table(df$Churn)

library(rpart)
treeimb <- rpart(Churn ~ ., data = df)
pred.treeimb <- predict(treeimb, newdata = df)
predict_reg <- ifelse(pred.treeimb >0.5, 1, 0)
predict_reg<-as.factor(predict_reg)
confusionMatrix(predict_reg,as.factor(df$Churn))

table(original$Churn)
table(df$Churn)

############## 6. Over sampling on decision tree

# Required library
library(ROSE)

original = df
table(original$Churn)

# Define the over-sampling method
sampled_data <- ovun.sample(Churn ~ ., data = df, method = "over")$data
df <- as.data.frame(sampled_data)

table(df$Churn)

library(rpart)
treeimb <- rpart(Churn ~ ., data = df)
pred.treeimb <- predict(treeimb, newdata = df)
predict_reg <- ifelse(pred.treeimb >0.5, 1, 0)
predict_reg<-as.factor(predict_reg)
confusionMatrix(predict_reg,as.factor(df$Churn))

table(original$Churn)
table(df$Churn)

############## 7. SMOTTEEN method on decision tree

# Required library
library(ROSE)

original = df
table(original$Churn)

# Define the SMOTTEEN method, that is both under and over sampling
df <- ovun.sample(Churn ~ ., data = df, method = "both", p=0.5, N=1000, seed = 1)$data
df <- as.data.frame(df)

table(df$Churn)

library(rpart)
treeimb <- rpart(Churn ~ ., data = df)
pred.treeimb <- predict(treeimb, newdata = df)
predict_reg <- ifelse(pred.treeimb >0.5, 1, 0)
predict_reg<-as.factor(predict_reg)
confusionMatrix(predict_reg,as.factor(df$Churn))

table(original$Churn)
table(df$Churn)

############## 8. Tuning
suppressMessages(library(gridExtra))
suppressMessages(library(tidyverse))
suppressMessages(library(ggplot2))
suppressMessages(library(GGally))
suppressMessages(library(MASS))
suppressMessages(library(smotefamily))
suppressMessages(library(randomForest))
suppressMessages(library(rpart))
suppressMessages(library(rpart.plot))
suppressMessages(library(e1071))

df = read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", sep=",", na.strings="?")
glimpse(df)

colnames(df)<-c("customerID","gender","seniorCitizen","partner","dependents","tenure","phoneService","multipleLines","internetService","onlineSecurity","onlineBackup","deviceProtection","techSupport","streamingTV","streamingMovies","contract","paperlessBilling","paymentMethod","monthlyCharges","totalCharges","churn")
summary(df)

df <- df %>% mutate(seniorCitizen=as.factor(seniorCitizen)) %>% na.omit()
summary(df)

df <- df %>% dplyr::select(-customerID) %>%
  mutate_at(7,~as.factor(case_when(. =="No phone service"~"No",.=="No"~"No",.=="Yes"~"Yes"))) %>%
  mutate_at(c(9:14),~as.factor(case_when(.=="No internet service"~"No", .=="No"~"No", .=="Yes"~"Yes")))
summary(df)

df %>% group_by(gender) %>% dplyr::summarize ("Number of observations"=n(),"Average Tenure in Months"=round(mean(tenure),0),"Monthly Charges"=round(mean(monthlyCharges),2),"Average Total Charges"=round(mean(totalCharges),2))

df <-df %>% mutate(churn=as.factor(churn))
g1 <- df %>% ggplot(aes(x=churn, y=tenure, fill=fct_rev(churn))) + geom_bar(stat="summary", fun="mean", alpha=0.6, show.legend=F) + stat_summary(aes(label=paste(round(..y.., 0), "months")), fun=mean, geom="text", size=3.5, vjust = -0.5) + labs(title = "Average Tenure")
g2 <- df %>% ggplot(aes(x=churn, y=monthlyCharges, fill=fct_rev(churn))) + geom_bar(stat="summary", fun="mean", alpha=0.6, show.legend=F) + stat_summary(aes(label = paste(round(..y.., 0), "months")), fun=mean, geom="text", size=3.5, vjust = -0.5) + labs(title = "Average Monthly Charges")

grid.arrange(g1, g2, ncol = 2, nrow = 1)

g3 <- df %>% ggplot(aes(x=contract, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn by Contract Type", y="Count of Contract Type")
g4 <- df %>% ggplot(aes(x=paymentMethod, fill=fct_rev(churn)))+ geom_bar(alpha=0.6) + labs(title="Customer Churn by Contract Type", y="Count of Payment Method")
g5 <- df %>% ggplot(aes(x=internetService, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn by Contract Type", y="Count of Internet Service") 

grid.arrange(g3, g4, g5)

g6 <- df %>% ggplot(aes(x=ifelse(seniorCitizen==1, "Senior", "Not Senior"), fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn on Senior Citizens", y="Count of Senior Citizen")
g7 <- df %>% ggplot(aes(x=gender, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn on Gender", y="Count of Gender")
g8 <- df %>% ggplot(aes(x=partner, fill=fct_rev(churn))) + geom_bar(alpha=0.6) + labs(title="Customer Churn on Partner", y="Count of Partner") 
g9 <- df %>% ggplot(aes(x=dependents, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn on Dependents", y="Count of Dependents") 

grid.arrange(g6, g7, g8, g9)

g10 <- df %>% ggplot(aes(x=phoneService, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn on Phone Service", y="Count of Phone Service")
g11 <- df %>% ggplot(aes(x=multipleLines, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn on Multiple Lines", y="Count of Mulitple Lines")
g12 <- df %>% ggplot(aes(x=onlineSecurity, fill=fct_rev(churn))) + geom_bar(alpha=0.6) + labs(title="Customer Churn on Online Security", y="Count of Online Security") 
g13 <- df %>% ggplot(aes(x=onlineBackup, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn on Online Backup", y="Count of Online Backup") 

grid.arrange(g10, g11, g12, g13, ncol=2)

g14 <- df %>% ggplot(aes(x=deviceProtection, fill=fct_rev(churn))) +  geom_bar(alpha=0.6) + labs(title="Customer Churn on Device Protection", y="Count of Device Protection")
g15 <- df %>% ggplot(aes(x=techSupport, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn on Tech Support", y="Count of Tech Support")
g16 <- df %>% ggplot(aes(x=streamingTV, fill=fct_rev(churn))) + geom_bar(alpha=0.6) + labs(title="Customer Churn on Streaming TV", y="Count of Streaming TV") 
g17 <- df %>% ggplot(aes(x=streamingMovies, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn on Streaming Movies", y="Count of Streaming Movies")
g18 <- df %>% ggplot(aes(x=paperlessBilling, fill=fct_rev(churn)))+  geom_bar(alpha=0.6) + labs(title="Customer Churn on Paperless Billing", y="Count of Paperless Billing")

grid.arrange(g14, g15, g16, g17, g18, ncol=2)

grid.arrange(g10, g11, g12, g13, g14, g15, g16, g17, g18, ncol=3)

df %>% dplyr::select(tenure, monthlyCharges, totalCharges, churn) %>% ggpairs(aes(color=fct_rev(churn)),diag = list(continuous = wrap("densityDiag", alpha = 0.6), discrete = wrap("barDiag", alpha = 0.7, color="grey30")))




df1 <- df %>% dplyr::select(-totalCharges)
glimpse(df1)

min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

norm <- as.data.frame(lapply(df1[,c(5,18)], min_max_norm))
summary(norm)

df_normed <- df1 %>% dplyr::select(-c(5,18)) %>% cbind(norm)
glimpse(df_normed)

set.seed(1)
train = sample(nrow(df_normed),nrow(df_normed)*0.7,replace=FALSE)
df_train = df_normed[train,]
df_test = df_normed[-train,]
dim(df_train)
dim(df_test)

#Logit Regression
model_AIC_1 <- glm(churn~.,df_train, family=binomial(link = "logit"))
model_AIC_2 <- stepAIC(model_AIC_1,direction = "both")
summary(model_AIC_2)

model_logit_tuned <- glm(formula = churn ~ seniorCitizen + phoneService + multipleLines +internetService + onlineBackup + deviceProtection + streamingTV + streamingMovies + contract + paperlessBilling + paymentMethod +  tenure + monthlyCharges, family = binomial(link = "logit"), data = df_train)

df_test$logit_pred_prob_tuned <- predict(model_logit_tuned, df_test, type="response") 
df_test$logit_pred_class_tuned <- ifelse(df_test$logit_pred_prob_tuned>0.5,"Yes","No")

mean(df_test$logit_pred_class_tuned==df_test$churn)

logit_tuned_ct <- table(df_test$logit_pred_class, df_test$churn) 
logit_tuned_ct
logit_tuned_recall <- (logit_tuned_ct[2,2]+logit_tuned_ct[1,1])/(logit_tuned_ct[2,2]+logit_tuned_ct[1,1]+logit_tuned_ct[1,2]+logit_tuned_ct[2,1]) 
logit_tuned_recall
logit_tuned_ct[1,2]
#Random Forest
set.seed(1)
res <- tuneRF(x = df_train %>% dplyr::select(-churn), y = df_train$churn, mtryStart=2, ntreeTry = 500)

set.seed(1) 
model_rf_tuning <- randomForest(churn~., df_train, ntree=50, mtry=2)
varImpPlot(model_rf_tuning)

model_rf_tuned <- randomForest(churn~tenure+monthlyCharges+contract+paymentMethod+internetService+onlineSecurity+paperlessBilling+techSupport, df_train, ntree=500, mtry=2)
df_test$rf_vote_tuned <- predict(model_rf_tuned, df_test, type="class")
mean(df_test$rf_vote_tuned==df_test$churn)

rf_tuned_ct <- table(df_test$rf_vote_tuned, df_test$churn)
rf_tuned_ct
rf_tuned_recall <- rf_tuned_ct[2,2]/(rf_tuned_ct[2,2]+rf_tuned_ct[1,2])
rf_tuned_recall

