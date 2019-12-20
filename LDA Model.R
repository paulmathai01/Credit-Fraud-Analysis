library(MASS)

data <- read.csv("C:\\Users\\eaugu\\Google Drive\\Classes\\DSA 6000\\home_default_risk_train.csv")
data <- na.omit(data)
data <- data[,c(2:19, 22:30)]
data <- data[,-c(4,20)]

trainSize <- round(0.6*11351)
testSize <- 11351-trainSize

set.seed(2019)

train <- sample(nrow(data), size = trainSize, replace = FALSE)
trainData <- data[train,]
testData <- data[-train,]

data.frame(colnames(trainData))


str(trainData)

trainData$FLAG_CONT_MOBILE <- as.factor(trainData$FLAG_CONT_MOBILE)
trainData$TARGET <- as.factor(trainData$TARGET)
trainData$FLAG_MOBIL <- as.factor(trainData$FLAG_MOBIL)
trainData$FLAG_EMP_PHONE <- as.factor(trainData$FLAG_EMP_PHONE)
trainData$FLAG_WORK_PHONE <- as.factor(trainData$FLAG_WORK_PHONE)
trainData$FLAG_PHONE <- as.factor(trainData$FLAG_PHONE)
trainData$FLAG_EMAIL <- as.factor(trainData$FLAG_EMAIL)

testData$FLAG_CONT_MOBILE <- as.factor(testData$FLAG_CONT_MOBILE)
testData$FLAG_MOBIL <- as.factor(testData$FLAG_MOBIL)
testData$FLAG_EMP_PHONE <- as.factor(testData$FLAG_EMP_PHONE)
testData$FLAG_WORK_PHONE <- as.factor(testData$FLAG_WORK_PHONE)
testData$FLAG_PHONE <- as.factor(testData$FLAG_PHONE)
testData$FLAG_EMAIL <- as.factor(testData$FLAG_EMAIL)

#Create logistic regression model to determine significant variables for LDA model
fullModel <- glm(formula = TARGET ~ ., family = binomial, data = trainData)

step(fullModel, direction = c("both"))

#Removed DAYS_EMPLOYED because it was producing probabilities of 0 or 1
reducedModel <- glm(formula = TARGET ~ CODE_GENDER + CNT_CHILDREN + 
                      AMT_INCOME_TOTAL + AMT_CREDIT + AMT_ANNUITY + AMT_GOODS_PRICE + 
                      DAYS_BIRTH, family = binomial, data = trainData)

summary(reducedModel)

#Make male dummy variable
trainData$MALE <- as.numeric(trainData$CODE_GENDER == "M")
trainData$MALE <- as.factor(trainData$MALE)

testData$MALE <- as.numeric(testData$CODE_GENDER == "M")
testData$MALE <- as.factor(testData$MALE)

#Final model, used stepwise to get good variables and then removed all insignificant variables that were left
finalModel <- glm(formula = TARGET ~ MALE + CNT_CHILDREN + 
                    AMT_INCOME_TOTAL + AMT_CREDIT + AMT_ANNUITY + AMT_GOODS_PRICE + 
                    DAYS_BIRTH, family = binomial, data = trainData)

summary(finalModel)

lda <- lda(formula = TARGET ~ MALE + CNT_CHILDREN + 
             AMT_INCOME_TOTAL + AMT_CREDIT + AMT_ANNUITY + AMT_GOODS_PRICE + 
             DAYS_BIRTH, data = trainData)

#Predict class with LDA model
lda.pred <- predict(lda, testData)
names(lda.pred)
lda.class <- lda.pred$class
table(lda.class, testData$TARGET)
mean(lda.class == testData$TARGET)

#ROC Curves
library(verification)
roc.plot(testData$TARGET, lda.pred$x
         )

install.packages("pROC")
library(pROC)
plot(roc(testData$TARGET, lda.pred$x
         ))

lda.pred$posterior
lda.pred$x
