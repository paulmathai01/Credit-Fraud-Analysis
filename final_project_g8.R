# In this analysis, we'll try to find the model that offers the best predictive ability 

# Call necessary libraries
library(MASS)
library(pROC)
library(ROCR)
library(caret)
library(rattle)
library(class)
library(randomForest)
library(reshape2)
library(ggplot2)
library(gbm)
library(tidyverse)


# Create Normalize function, to be used for kNN
normalize <- function(x) {
  normalized_x <- (x - min(x)) / (max(x) - min(x))
  return(normalized_x)
}

## Data set up
# Set up work directory 
setwd("C:\\Users\\Kareem\\Documents\\Wayne State University\\Fall 2019\\DSA 6000\\Final Project\\R Code")

# Uploading data
default_data = read.csv(file="app_final.csv", header = T)

# Removing ID column
default_data <- default_data[,2:23]

# Removing NULL cases
default_data <- default_data[complete.cases(default_data),]

# Changing TARGET column from numeric to factor
default_data$TARGET <- as.factor(default_data$TARGET)

# Adding a column of payment as a perc of income
default_data$PERC_PAYMENT_INC <- round(default_data$AMT_PAYMENT / default_data$AMT_INCOME_TOTAL, digits = 4)

# Adding a column of credit term
default_data$NUM_CREDIT_TERM <- round(default_data$AMT_CREDIT / default_data$AMT_PAYMENT, digits = 2)

# Creating subsets of training and test data
# We're using 70% as training and 30% as test (validation) data
train_size <- round(0.70 * nrow(default_data))
test_size <- nrow(default_data) - train_size

train <- sample(nrow(default_data), size = train_size, replace = FALSE)
train_data <- default_data[train,]
summary(train_data$TARGET)
test_data <- default_data[-train,]
summary(test_data$TARGET)

# Fixing the datasets
set.seed(123)

## Correlation Matrix
cor(train_data[,2:24])

## Correlation Heatmap
cormat <- cor(train_data[,2:24])

# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
reorder_cormat <- function(cormat){

  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}
cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)

# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)

# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
# Print the heatmap
print(ggheatmap)


### Logsitic Regression

# Model with all features
glm.fit.all <- glm(TARGET~., data=train_data, family="binomial")
summary(glm.fit.all)

# Assessing the model, running it on the test data
glm.pred.all <- predict(glm.fit.all, test_data, type="response")

# Probability distribution
hist(glm.pred.all, breaks=20)

# Using 10% threshold based on the probability distribution
glm.pred.class <- factor(ifelse(glm.pred.all >=0.10, 1, 0))
mean(glm.pred.class == test_data$TARGET)
confusionMatrix(glm.pred.class, test_data$TARGET)
# Using a threshold of more than 10% will increase the Sensitivity and decrease the Specificity

# ROC Curve
glm.pred.roc <- predict(glm.fit.all, test_data, type="response")
roc_step <- roc(test_data$TARGET, glm.pred.roc)
roc_step$auc
plot(roc(test_data$TARGET, glm.pred.roc),legacy.axes = TRUE
     , xlab="Specificity", main=paste0("Log Reg ROC curve \n"," AUC = ", round(roc_step$auc, digits = 4)), col="red")


# Stepwise selection of variables
step(glm_fit_all, direction = c("both"))


# After selecting the best variables, creating a new model
glm.final = glm(formula = TARGET ~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                  AMT_INCOME_TOTAL + AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + 
                  FLAG_EDU_HIGHER + FLAG_EDU_SEC + FLAG_MS_SEP + FLAG_MS_CM + 
                  FLAG_MS_SINGLE + DAYS_AGE + FLAG_RETIRED + DAYS_EMPLOYED + 
                  CNT_PREV_APPS + DAYS_PREV_DPD + PERC_PAYMENT_INC + NUM_CREDIT_TERM, 
                family = "binomial", data = train_data)

summary(glm.final)


# Predicting the test TARGET
glm.pred <- predict(glm.final, test_data, type="response")
hist(glm.pred, breaks=50)
glm.pred.class <- factor(ifelse(glm.pred >=0.10, 1, 0))
#table(glm.pred.class, test_data$TARGET)
mean(glm.pred.class == test_data$TARGET)
confusionMatrix(glm.pred.class, test_data$TARGET)

# Creating the ROC Curve 
df <- data.frame(score = glm.pred, true.class = test_data$TARGET)
head(df)
tail(df)
# ______________________________________________________
n <- length(test_data$TARGET) # Change the sample size from the ROC file.
n

# Sort by score (high to low)
df <- df[order(-df$score),]
rownames(df) <- NULL  # Reset the row number to 1,2,3,...

# Total # of positive and negative cases in the data set
P = sum(df$true.class == 1)
N = sum(df$true.class == 0)

# Vectors to hold the coordinates of points on the ROC curve
TPR = c(0,vector(mode="numeric", length=n))
FPR = c(0,vector(mode="numeric", length=n))

# Calculate the coordinates from one point to the next
for(k in 1:n){
  if(df[k,"true.class"] == 1){
    TPR[k+1] = TPR[k] + 1/P
    FPR[k+1] = FPR[k]
  } else{
    TPR[k+1] = TPR[k]
    FPR[k+1] = FPR[k] + 1/N
  }
}

# Color scheme: Positive case in red, negative case in blue
color = c("black","blue","red")

# Plot the ROC curve
plot(FPR, TPR, main=paste0("Log Reg Final ROC curve"," (n = ", n, ")"), pch=16, col=color[(2+c(-1,df$true.class))],
     cex.lab = 1.5, cex.axis = 1.5, cex.main = 1.5)
lines(FPR,TPR)



#Area under the ROC curve
roc_step <- roc(test_data$TARGET, glm.pred)
roc_step$auc
plot(roc(test_data$TARGET, glm.pred),legacy.axes = TRUE
     , xlab="Specificity", main=paste0("Log Reg ROC curve \n"," AUC = ", round(roc_step$auc, digits = 4)), col="red")


#___________________________________________
# LDA Model


lda.final <- lda(TARGET ~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                   AMT_INCOME_TOTAL + AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + 
                   FLAG_EDU_HIGHER + FLAG_EDU_SEC + FLAG_MS_SEP + FLAG_MS_CM + 
                   FLAG_MS_SINGLE + DAYS_AGE + FLAG_RETIRED + DAYS_EMPLOYED + 
                   CNT_PREV_APPS + DAYS_PREV_DPD + PERC_PAYMENT_INC + NUM_CREDIT_TERM, data=train_data)

lda.pred <- predict(lda.final, test_data)
lda.class <- lda.pred$class
table(lda.class, test_data$TARGET)
mean(lda.class == test_data$TARGET)
hist(lda.pred$posterior)

confusionMatrix(lda.class, test_data$TARGET)

df <- data.frame(score = lda.pred$posterior[,2], true.class = test_data$TARGET)

roc_step <- roc(test_data$TARGET, lda.pred$posterior[,2])
roc_step$auc
plot(roc(test_data$TARGET, lda.pred$posterior[,2]),legacy.axes = TRUE
     , xlab="Specificity", main=paste0("LDA Model ROC curve \n"," AUC = ", round(roc_step$auc, digits = 4)), col="red")

#___________________________________________
# QDA Model


qda.final <- qda(TARGET ~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                   AMT_INCOME_TOTAL + AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + 
                   FLAG_EDU_HIGHER + FLAG_EDU_SEC + FLAG_MS_SEP + FLAG_MS_CM + 
                   FLAG_MS_SINGLE + DAYS_AGE + FLAG_RETIRED + DAYS_EMPLOYED + 
                   CNT_PREV_APPS + DAYS_PREV_DPD + PERC_PAYMENT_INC + NUM_CREDIT_TERM, data=train_data)

qda.pred <- predict(qda.final, test_data)
qda.class <- qda.pred$class
table(qda.class, test_data$TARGET)
mean(qda.class == test_data$TARGET)
hist(qda.pred$posterior)

confusionMatrix(qda.class, test_data$TARGET)

df <- data.frame(score = qda.pred$posterior[,2], true.class = test_data$TARGET)

roc_step <- roc(test_data$TARGET, qda.pred$posterior[,2])
roc_step$auc
plot(roc(test_data$TARGET, qda.pred$posterior[,2]),legacy.axes = TRUE
     , xlab="Specificity", main=paste0("QDA Model ROC curve \n"," AUC = ", round(roc_step$auc, digits = 4)), col="red")

#___________________________________________
# Random Forest Model

mod.rf <- randomForest(TARGET ~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                         AMT_INCOME_TOTAL + AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + 
                         FLAG_EDU_HIGHER + FLAG_EDU_SEC + FLAG_MS_SEP + FLAG_MS_CM + 
                         FLAG_MS_SINGLE + DAYS_AGE + FLAG_RETIRED + DAYS_EMPLOYED + 
                         CNT_PREV_APPS + DAYS_PREV_DPD + PERC_PAYMENT_INC + NUM_CREDIT_TERM, data=train_data, prob=TRUE)
plot(mod.rf, main="Random Forest Model", xlab="# Trees")
importance(mod.rf)
pred.rf <- predict(mod.rf, newdata = test_data, type = "prob")
pred.rf[,2]
rf.pred.class <- factor(ifelse(pred.rf[,2] >=0.10, 1, 0))
confusionMatrix(rf.pred.class, test_data$TARGET)

roc_rf <- roc(test_data$TARGET, pred.rf[,1])
plot(roc_rf,legacy.axes = TRUE
     , xlab="Specificity", main=paste0("RF Model ROC curve \n"," AUC = ", round(roc_rf$auc, digits = 4)), col="red")
roc_rf$auc


#___________________________________________
# KNN Model


train.X<- as.data.frame(lapply(train_data[, c('FLAG_CASH_LOAN', 'CODE_FEMALE', 'FLAG_OWN_CAR', 
                                              'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_PAYMENT', 'AMT_ITEM_PRICE', 
                                              'FLAG_EDU_HIGHER', 'FLAG_EDU_SEC', 'FLAG_MS_SEP', 'FLAG_MS_CM', 
                                              'FLAG_MS_SINGLE', 'DAYS_AGE', 'FLAG_RETIRED', 'DAYS_EMPLOYED', 
                                              'CNT_PREV_APPS', 'DAYS_PREV_DPD', 'PERC_PAYMENT_INC', 'NUM_CREDIT_TERM')], normalize))
test.X <- as.data.frame(lapply(test_data[, c('FLAG_CASH_LOAN', 'CODE_FEMALE', 'FLAG_OWN_CAR', 
                                             'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_PAYMENT', 'AMT_ITEM_PRICE', 
                                             'FLAG_EDU_HIGHER', 'FLAG_EDU_SEC', 'FLAG_MS_SEP', 'FLAG_MS_CM', 
                                             'FLAG_MS_SINGLE', 'DAYS_AGE', 'FLAG_RETIRED', 'DAYS_EMPLOYED', 
                                             'CNT_PREV_APPS', 'DAYS_PREV_DPD', 'PERC_PAYMENT_INC', 'NUM_CREDIT_TERM')], normalize))
train.TARGET = train_data$TARGET
set.seed(1)

## Creating the KNN prediction

knn.pred <- knn(train.X, test.X, train.TARGET, k=5)
                                                                                                                                                                                                  
## Finally, the confusion matrix 
knn.pred
  
confusionMatrix(knn.pred, test_data$TARGET)

## The % of correct predictions


roc_step <- roc(test_data$TARGET, knn.pred)
roc_step$auc
plot(roc(test_data$TARGET, knn.pred),legacy.axes = TRUE
     , xlab="Specificity", main=paste0("KNN Model ROC curve \n"," AUC = ", round(roc_step$auc, digits = 4)), col="red")



#___________________________________________
# GBM Model

gbm.final <- gbm (TARGET ~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                    AMT_INCOME_TOTAL + AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + 
                    FLAG_EDU_HIGHER + FLAG_EDU_SEC + FLAG_MS_SEP + FLAG_MS_CM + 
                    FLAG_MS_SINGLE + DAYS_AGE + FLAG_RETIRED + DAYS_EMPLOYED + 
                    CNT_PREV_APPS + DAYS_PREV_DPD + PERC_PAYMENT_INC + NUM_CREDIT_TERM,
                  distribution = "gaussian",
                  data=train_data,
                  n.trees = 1000,
                  interaction.depth = 4,
                  shrinkage = 0.01,
                  cv.folds = 3)

summary(gbm.final)

n.trees.cv <- gbm.perf(gbm.final, method = "cv")
n.trees.cv
n.trees.oob <- gbm.perf(gbm.final, method = "OOB")
n.trees.oob
mean(pred.gbm-1)

pred.gbm <- predict(gbm.final, newdata = test_data, n.trees = 1000, type = "response")
pred.gbm.class <- factor(ifelse(pred.gbm-1 >=0.10, 1, 0))
confusionMatrix(pred.gbm.class, test_data$TARGET)


roc_step <- roc(test_data$TARGET, pred.gbm)
roc_step$auc
plot(roc(test_data$TARGET, pred.gbm),legacy.axes = TRUE
     , xlab="Specificity", main=paste0("GBM Model ROC curve \n"," AUC = ", round(roc_step$auc, digits = 4)), col="red")

