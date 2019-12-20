library(MASS)

data <- read.csv("C:\\Users\\eaugu\\Google Drive\\Classes\\DSA 6000\\FINAL_DATA.csv")
data <- na.omit(data)

trainSize <- round(0.6*270289)
testSize <- 270289-trainSize

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

#Model determined from stepwise, removed FLAG_EDU_SEC since p-value was 0.06
reducedModel <- glm(formula = TARGET ~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                      AMT_INCOME_TOTAL + AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + 
                      FLAG_EDU_HIGHER + FLAG_MS_SEP + FLAG_MS_CM + 
                      FLAG_MS_SINGLE + DAYS_AGE + DAYS_EMPLOYED + CNT_PREV_APPS + 
                      DAYS_PREV_DPD, family = binomial, data = trainData)

summary(reducedModel)

#Make male dummy variable
#trainData$MALE <- as.numeric(trainData$CODE_GENDER == "M")
#trainData$MALE <- as.factor(trainData$MALE)

#testData$MALE <- as.numeric(testData$CODE_GENDER == "M")
#testData$MALE <- as.factor(testData$MALE)

#Final model, used stepwise to get good variables and then removed all insignificant variables that were left
finalModel <- glm(formula = TARGET ~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
                    AMT_INCOME_TOTAL + AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + 
                    FLAG_EDU_HIGHER + FLAG_MS_SEP + FLAG_MS_CM + 
                    FLAG_MS_SINGLE + DAYS_AGE + DAYS_EMPLOYED + CNT_PREV_APPS + 
                    DAYS_PREV_DPD, family = binomial, data = trainData)

summary(finalModel)

lda <- lda(formula = TARGET ~ FLAG_CASH_LOAN + CODE_FEMALE + FLAG_OWN_CAR + 
             AMT_INCOME_TOTAL + AMT_CREDIT + AMT_PAYMENT + AMT_ITEM_PRICE + 
             FLAG_EDU_HIGHER + FLAG_MS_SEP + FLAG_MS_CM + 
             FLAG_MS_SINGLE + DAYS_AGE + DAYS_EMPLOYED + CNT_PREV_APPS + 
             DAYS_PREV_DPD, data = trainData)

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

library(pROC)
plot(roc(testData$TARGET, lda.pred$x
         ))
