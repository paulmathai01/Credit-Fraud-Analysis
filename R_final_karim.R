library(MASS)
library(pROC)
library(ROCR)
library(caret)
library(rattle)
## Data set up
# Set up work directory 
setwd("C:\\Users\\Kareem\\Documents\\Wayne State University\\Fall 2019\\DSA 6000\\Final Project\\R Code")

# Uploading data
default_data = read.csv(file="FINAL_DATA.CSV", header = T)

# Removing ID column
default_data <- default_data[,2:22]

# Removing NULL cases
default_data <- default_data[complete.cases(default_data),]

# Changing TARGET column to factor
default_data$TARGET <- as.factor(default_data$TARGET)

# Creating subsets of training and test data
train_size <- round(0.60 * nrow(default_data))
test_size <- nrow(default_data) - train_size

train <- sample(nrow(default_data), size = train_size, replace = FALSE)
train_data <- default_data[train,]
summary(train_data$TARGET)
test_data <- default_data[-train,]
summary(test_data$TARGET)

# Fixing the datasets
set.seed(2019)

## Correlation Matrix
cor(train_data[,2:21])

### Logsitic Regression
# Model with all variables
glm_fit_all <- glm(TARGET~., data=train_data, family="binomial")
summary(glm_fit_all)

# Stepwise selection of variables
#step(glm_fit_all, direction = c("both"))

# Assessing the Predictive ability of the model
glm_probs_all <- predict(glm_fit_all, test_data, type="response")
roc_step <- roc(test_data$TARGET, glm_probs_all)
roc_step$auc
#glm_pred_all <- rep("0", length(glm_probs_all))
#glm_pred_all[glm_probs_all >.5] = "1"
#table(glm_probs_all,train_data$TARGET)
#(glm_probs_all == train_data$TARGET)

# After selecting the best variables, creating a new model
glm.final = glm(formula = TARGET ~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
      AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + FLAG_EDU_HIGHER + 
      FLAG_EDU_SEC + FLAG_MS_SEP + FLAG_MS_CM + FLAG_MS_SINGLE + 
      DAYS_AGE + DAYS_EMPLOYED + CNT_PREV_APPS + DAYS_PREV_DPD, 
    family = "binomial", data = train_data)

summary(glm.final)


# Predicting the test TARGET
glm.pred <- predict(glm.final, test_data, type="response")
hist(glm.pred)
glm_pred_all <- rep("0", length(glm.pred))
glm_pred_all[glm.pred >.5] = "1"
table(glm_pred_all, test_data$TARGET)
mean(glm_pred_all == test_data$TARGET)


# Creating the ROC Curve 
df <- data.frame(score = glm.pred, true.class = test_data$TARGET)
df <- df[order(-df$score),]
df
#Area under the ROC curve
roc_step <- roc(test_data$TARGET, glm.pred)
roc_step$auc
plot(roc(test_data$TARGET, qda.pred$posterior[,2]),legacy.axes = TRUE
     , xlab="Specificity", main=paste0("Log Reg ROC curve \n"," AUC = ", round(roc_step$auc, digits = 4)), col="red")


#___________________________________________
# LDA Model


lda.final <- lda(TARGET~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                 AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + FLAG_EDU_HIGHER + 
                 FLAG_EDU_SEC + FLAG_MS_SEP + FLAG_MS_CM + FLAG_MS_SINGLE + 
                 DAYS_AGE + DAYS_EMPLOYED + CNT_PREV_APPS + DAYS_PREV_DPD, data=train_data)

lda.pred <- predict(lda.final, test_data)
lda.class <- lda.pred$class
table(lda.class, test_data$TARGET)
mean(lda.class == test_data$TARGET)
hist(lda.pred$posterior)

df <- data.frame(score = lda.pred$posterior[,2], true.class = test_data$TARGET)

roc_step <- roc(test_data$TARGET, lda.pred$posterior[,2])
roc_step$auc
plot(roc(test_data$TARGET, lda.pred$posterior[,2]),legacy.axes = TRUE
     , xlab="Specificity", main=paste0("LDA Model ROC curve \n"," AUC = ", round(roc_step$auc, digits = 4)), col="red")

#___________________________________________
# QDA Model


qda.final <- qda(TARGET~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                   AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + FLAG_EDU_HIGHER + 
                   FLAG_EDU_SEC + FLAG_MS_SEP + FLAG_MS_CM + FLAG_MS_SINGLE + 
                   DAYS_AGE + DAYS_EMPLOYED + CNT_PREV_APPS + DAYS_PREV_DPD, data=train_data)

qda.pred <- predict(qda.final, test_data)
qda.class <- qda.pred$class
table(qda.class, test_data$TARGET)
mean(qda.class == test_data$TARGET)
hist(qda.pred$posterior)

df <- data.frame(score = qda.pred$posterior[,2], true.class = test_data$TARGET)

roc_step <- roc(test_data$TARGET, qda.pred$posterior[,2])
roc_step$auc
plot(roc(test_data$TARGET, qda.pred$posterior[,2]),legacy.axes = TRUE
     , xlab="Specificity", main=paste0("QDA Model ROC curve \n"," AUC = ", round(roc_step$auc, digits = 4)), col="red")

#___________________________________________
# Random Forest Model

library(randomForest)
mod.rf <- randomForest(TARGET~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                         AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + FLAG_EDU_HIGHER + 
                         FLAG_EDU_SEC + FLAG_MS_SEP + FLAG_MS_CM + FLAG_MS_SINGLE + 
                         DAYS_AGE + DAYS_EMPLOYED + CNT_PREV_APPS + DAYS_PREV_DPD, data=train_data, prob=TRUE)
plot(mod.rf, main="Random Forest Model", xlab="# Trees")
importance(mod.rf)

pred.rf <- predict(mod.rf, newdata = test_data, type = "prob")
mean(pred.rf == test_data$TARGET)

roc_rf <- roc(test_data$TARGET, pred.rf[,1])
plot(roc_rf,legacy.axes = TRUE
     , xlab="Specificity", main=paste0("RF Model ROC curve \n"," AUC = ", round(roc_rf$auc, digits = 4)), col="red")
roc_rf$auc
