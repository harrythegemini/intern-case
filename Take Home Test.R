####### Happy Money Intern Case

##### Section 2
#install.packages("ggplot2")
library(ggplot2)
#install.packages("tidyverse")
library(tidyverse)
#install.packages("dplyr")
library(dplyr)
#install.packages("caret")
library(caret)
#install.packages("ROSE")
library(ROSE)

# For confusion matrix
#install.packages("cvms")
library(cvms)
#install.packages("broom")
library(broom)
#install.packages("tibble")
library(tibble)
#install.packages("ggimage")
library(ggimage)


## read csv
risk = read.csv("risk_dataset.csv")
head(risk)
str(risk)

## Check missing value
#install.packages("mice")
library(mice)
md.pattern(risk)

# There is no missing value in the dataset.

## Change BadFlag into factors
risk$BadFlag = as.factor(risk$BadFlag)

## Select distinct rows
risk <- risk %>% distinct(.keep_all = FALSE)

####EDA####

## Pairwise plots
pairs(risk[,3:9], col = risk$BadFlag)

## Scatter plot - What's the distribution of credit card balance and annual income of customers? 
scatter1 = ggplot(risk, aes(x=CreditCardBalance, y=IncomeAnnual, color = factor(BadFlag))) + 
  geom_point() + 
  labs(title = "Income vs Credit Card Balance", x= "credit card balance", y = "annual income") + 
  theme_classic()
plot(scatter1)

## boxplot - FICO score between default and non-default people
box_FICO = ggplot(risk, aes(x=factor(BadFlag), y=FicoScore)) + 
  geom_boxplot() + 
  labs(title = "FICO Score by dafault status", x = "BadFlag", y = "FICO score") + 
  theme_classic()
plot(box_FICO)



## boxplot - CustomScore1 between default and non-default people
box_Score1 = ggplot(risk, aes(x=BadFlag, y=CustomScore1)) + 
  geom_boxplot() +
  labs(title = "Custom Score 1 by dafault status", x = "BadFlag", y = "Custom score 1") + 
  theme_classic()

plot(box_Score1)

## boxplot - CustomeScore2 between default and non-default people
box_Score2 = ggplot(risk, aes(x=BadFlag, y=CustomScore2)) + 
  geom_boxplot() +
  labs(title = "Custom Score 2 by dafault status", x = "BadFlag", y = "Custom score 2") + 
  theme_classic()
plot(box_Score2)

## Bar plot
bar = ggplot(risk_Tr_FICO, aes(x = BadFlag)) + 
  labs(title = "Count of BadFlag") + 
  geom_bar() + 
  theme_classic()
  
bar

###### Problem 1 Of the different model scores (FICO, Score1, Score2) which one does a better job of separating good accounts and bad accounts?

## Seperate into training and testing data.(70% training data, 30% testing data)
set.seed(468)
spliter = sample(c(rep(0,0.7 * nrow(risk)), rep(1,0.3 * nrow(risk)))) # create a fake indicator to indicate training or testing
risk_Tr = risk[spliter == 0,]
risk_Te = risk[spliter == 1,]

################ Logistic Regression #################
################ FICO Score #################

#### Fit into the model
## Model training
fit.FICO = glm(BadFlag ~ FicoScore, data = risk_Tr, family = binomial) #"family = binomial" tell glm to fit a logistic model.
summary(fit.FICO)

## Predict test data
FICO.prob = predict(fit.FICO, newdata = risk_Te,type = "response")
FICO.prob[1:5]

## Plot check the probability distribution
plot(x = risk_Te$FicoScore, y = FICO.prob)
hist(FICO.prob)

## Changing probabilities
FICO.pred = ifelse(FICO.prob > 0.5, 1,0)

#### Model Evaluabtion
table(risk_Te$BadFlag, FICO.pred)
error = mean(FICO.pred != risk_Te$BadFlag)
print(paste('Accuracy = ', 1 - error))
plot(x = risk_Te$FicoScore, y = FICO.prob)
# Though the model accuracy is 88%, we can see the all predictied results are below 0.5, which means the model doesn't predict a "1" successfully on the test dataset. The model is unreliable in credit risk predicting.
# The reason is that the dataset is imbalanced in BadFlag column, which means the default is much smaller than non-default. Hence, we here use oversampling to make the dataset balance in BadFlag
# We want to make the dataset balanced in BadFlag, so the next line of code over sample the minority class until the two classes are equal, and the total dataset goes to N = 118758.

#### Deal with imbalanced data
## Oversampling
table(risk_Tr$BadFlag)
risk_Tr_FICO = ovun.sample(BadFlag ~ FicoScore, data = risk_Tr, method = "over", N = 118758)$data
table(risk_Tr_FICO$BadFlag) # same rows in two classes.

#### Refit the logistic regression model
## Training model
fit_FICO= glm(BadFlag ~ FicoScore, data = risk_Tr_FICO, family = binomial) #"family = binomial" tell glm to fit a logistic model.
summary(fit_FICO)

## Predict the test data
FICO_prob = predict(fit_FICO, newdata = risk_Te,type = "response")

## Plot FICO score and probability
plot(x = risk_Te$FicoScore, y = FICO_prob)
hist(FICO_prob,  main = "Histogram of prediction based on Fico Score model")

# The plots seems the model reflects more comprehensively on the data. It has predictions above and lower than 0.5.

## Change probability into class
FICO_pred = ifelse(FICO_prob > 0.5, 1,0)

#### Evaluating model
## Accuracy
error_FICO = mean(FICO_pred != risk_Te$BadFlag)
print(paste('Accuracy = ', 1 - error_FICO))
# The actual model accuracy is 59.18%

## Precision & Recall & F-statistics
FICO_table = table(as.factor(FICO_pred),risk_Te$BadFlag)
recall(FICO_table)
precision(FICO_table)
F_meas(FICO_table)

## ROC and AUC curve
#install.packages("ROCR")
library(ROCR)
ROCPred_FICO = prediction(FICO_pred_oversampled, risk_Te$BadFlag)
ROC_FICO = performance(ROCPred_FICO, measure = "tpr", x.measure = "fpr")

auc_FICO = performance(ROCPred_FICO, measure = "auc")
auc_FICO = auc_FICO@y.values[[1]]

plot(ROC_FICO)
plot(ROC_FICO, colorize = TRUE,
     main = "ROC Curve_FICO")
abline(a = 0, b = 1)

auc_FICO = round(auc_FICO,4)
legend(.6,.4,auc_FICO, title = "AUC",cex=1)

## Confusion matrix
FICO_table = table(risk_Te$BadFlag, FICO_pred_oversampled)
cfm_FICO = as.data.frame(table(risk_Te$BadFlag, FICO_pred_oversampled))
cfm_FICO_plot = ggplot(data = cfm_FICO, mapping = aes(x = Var1, y =FICO_pred_oversampled)) + 
  geom_tile(aes(fill = Freq), colour = "white") + 
  geom_text(aes(label = sprintf("%1.0f",Freq)), vjust = 1) + 
  labs(title = "Confusion Matrix of Fico Score Model", x= "True_BadFlag", y = "Prediction_BadFlag") +
  scale_fill_gradient(low = "white", high = "steelblue")

cfm_FICO_plot

# Using cvms package (with percentage in the plot)

FICO_tibble = tibble("True_BadFlag" = risk_Te$BadFlag, "Prediction_BadFlag" = as.factor(FICO_pred_oversampled))
FICO_cvms = confusion_matrix(targets = FICO_tibble$True_BadFlag, predictions = FICO_tibble$Prediction_BadFlag)
plot_confusion_matrix(
  FICO_cvms$`Confusion Matrix`[[1]],
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)

# In order to minimize the credit risk, we want to reduce the type2 error which is false negative. In order to help the good customers get loan and improve user experience, we want to reduce type 1 error which false positive.

################ Custom Score 1 #################

### Oversampling
risk_Tr_score1 = ovun.sample(BadFlag ~ CustomScore1, data = risk_Tr, method = "over", N = 118758)$data
str(risk_Tr_score1)

### Fit in the model
## Training
fit_score1 = glm(BadFlag ~ CustomScore1, data = risk_Tr_score1, family = binomial) #"family = binomial" tell glm to fit a logistic model.
summary(fit_score1)

## Predict the test data
score1_prob = predict(fit_score1, newdata = risk_Te,type = "response")

## Plot Custom score 1 and probability
hist(score1_prob, main = "Histogram of predictions based on CustomScore1 model")

## Change probability into class
score1_pred = ifelse(score1_prob > 0.5, 1,0)

#### Evaluating model
## Accuracy
error_score1 = mean(score1_pred != risk_Te$BadFlag)
print(paste('Accuracy = ', 1 - error_score1))

# The model accuracy of Custom Score 1 is 74.85%. 

## Precision & Recall & F-statistics
score1_table = table(as.factor(score1_pred),risk_Te$BadFlag)
recall(score1_table)
precision(score1_table)
F_meas(score1_table)

## Confusion matrix
score1_tibble = tibble("True_BadFlag" = risk_Te$BadFlag, "Prediction_BadFlag" = as.factor(score1_pred))
score1_cvms = confusion_matrix(targets = score1_tibble$True_BadFlag, predictions = score1_tibble$Prediction_BadFlag)
plot_confusion_matrix(
  score1_cvms$`Confusion Matrix`[[1]],
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)

## ROC and AUC curve
ROCPred_score1 = prediction(score1_pred, risk_Te$BadFlag)
ROC_score1 = performance(ROCPred_score1, measure = "tpr", x.measure = "fpr")

auc_score1 = performance(ROCPred_score1, measure = "auc")
auc_score1 = auc_score1@y.values[[1]]

plot(ROC_score1)
plot(ROC_score1, colorize = TRUE,
     main = "ROC Curve_CustomScore1")
abline(a = 0, b = 1)

auc_score1 = round(auc_score1,4)
legend(.6,.4,auc_score1, title = "AUC",cex=1)



################ Custom Score 2 #################
### Oversampling
risk_Tr_score2 = ovun.sample(BadFlag ~ CustomScore2, data = risk_Tr, method = "over", N = 118758)$data
str(risk_Tr_score2)

### Fit in the model
## Training
fit_score2 = glm(BadFlag ~ CustomScore2, data = risk_Tr_score2, family = binomial) #"family = binomial" tell glm to fit a logistic model.
summary(fit_score2)

## Predict the test data
score2_prob = predict(fit_score2, newdata = risk_Te,type = "response")

## Plot Custom score 2 and probability
plot(x = risk_Te$CustomScore2, y = score2_prob)
hist(score2_prob,  main = "Histogram of predictions based on CustomScore2 model")

## Change probability into class
score2_pred = ifelse(score2_prob > 0.5, 1,0)

#### Evaluating model
## Accuracy
error_score2 = mean(score2_pred != risk_Te$BadFlag)
print(paste('Accuracy = ', 1 - error_score2))

# The model accuracy of Custom Score 2 is 75.64%. 

## Precision & Recall & F-statistics
score2_table = table(as.factor(score2_pred),risk_Te$BadFlag)
recall(score2_table)
precision(score2_table)
F_meas(score2_table)

## Confusion matrix
score2_tibble = tibble("True_BadFlag" = risk_Te$BadFlag, "Prediction_BadFlag" = as.factor(score2_pred))
score2_cvms = confusion_matrix(targets = score2_tibble$True_BadFlag, predictions = score2_tibble$Prediction_BadFlag)
plot_confusion_matrix(
  score2_cvms$`Confusion Matrix`[[1]],
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)


## ROC and AUC curve
ROCPred_score2 = prediction(score2_pred, risk_Te$BadFlag)
ROC_score2 = performance(ROCPred_score2, measure = "tpr", x.measure = "fpr")

auc_score2 = performance(ROCPred_score2, measure = "auc")
auc_score2 = auc_score2@y.values[[1]]

plot(ROC_score2)
plot(ROC_score2, colorize = TRUE,
     main = "ROC Curve_CustomScore2")
abline(a = 0, b = 1)

auc_score2 = round(auc_score2,4)
legend(.6,.4,auc_score2, title = "AUC",cex=1)

###Summary: Custom Score 2 does a better job in seprating good and bad accounts because the its accuracy on test dataset is the highest, the F value is the highest, and AUC is the highest.




###Problem 2 : Are CreditCardBalance and InquiriesInLast6Months predictive of risk in addition to the best score you identified from the last question?

################ Credit Card Balance #################
# Adding Credit Card Balance into the logistic regression model.

### Oversampling
risk_Tr_cardbalance = ovun.sample(BadFlag ~ CustomScore2 + CreditCardBalance, data = risk_Tr, method = "over", N = 118758)$data
table(risk_Tr_cardbalance$BadFlag)

### Fit in the model
## Training
fit_cardbalance = glm(BadFlag ~ CustomScore2 + CreditCardBalance, data = risk_Tr_cardbalance, family = binomial) #"family = binomial" tell glm to fit a logistic model.
summary(fit_cardbalance)

## Predict the test data
cardbalance_prob = predict(fit_cardbalance, newdata = risk_Te,type = "response")

## Plot the probability distribution in addition to card balance
hist(cardbalance_prob, main = "Histogram of predictions based on 
     CustomScore2 + Credit Card Balance model")

## Change probability into class
cardbalance_pred = ifelse(cardbalance_prob > 0.5, 1,0)

#### Evaluating model
## Accuracy
error_cardbalance = mean(cardbalance_pred != risk_Te$BadFlag)
print(paste('Accuracy = ', 1 - error_cardbalance))

# The model accuracy of Custom Score 1 is 75.392%. 

## Precision & Recall & F-statistics
cardbalance_table = table(as.factor(cardbalance_pred),risk_Te$BadFlag)
recall(cardbalance_table)
precision(cardbalance_table)
F_meas(cardbalance_table)

## Confusion matrix
cardbalance_tibble = tibble("True_BadFlag" = risk_Te$BadFlag, "Prediction_BadFlag" = as.factor(cardbalance_pred))
cardbalance_cvms = confusion_matrix(targets = cardbalance_tibble$True_BadFlag, predictions = cardbalance_tibble$Prediction_BadFlag)
plot_confusion_matrix(
  cardbalance_cvms$`Confusion Matrix`[[1]],
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)

## ROC and AUC curve
ROCPred_cardbalance = prediction(cardbalance_pred, risk_Te$BadFlag)
ROC_cardbalance = performance(ROCPred_cardbalance, measure = "tpr", x.measure = "fpr")

auc_cardbalance = performance(ROCPred_cardbalance, measure = "auc")
auc_cardbalance = auc_cardbalance@y.values[[1]]

plot(ROC_cardbalance)
plot(ROC_cardbalance, colorize = TRUE,
     main = "ROC Curve_Score2+Credit card balance")
abline(a = 0, b = 1)

auc_cardbalance = round(auc_cardbalance,4)
legend(.6,.4,auc_cardbalance, title = "AUC",cex=1)



################ Inquiries In Last 6 Months #################
# Adding credit inquiries in last 6 months into the logistic regression model.

### Oversampling
risk_Tr_Inquries = ovun.sample(BadFlag ~ CustomScore2 + InquriesInLast6Months, data = risk_Tr, method = "over", N = 118758)$data
table(risk_Tr_Inquries$BadFlag)

### Fit in the model
## Training
fit_Inquries = glm(BadFlag ~ CustomScore2 + InquriesInLast6Months, data = risk_Tr_Inquries, family = binomial) #"family = binomial" tell glm to fit a logistic model.
summary(fit_Inquries)

## Predict the test data
Inquries_prob = predict(fit_Inquries, newdata = risk_Te,type = "response")

## Plot the probability distribution in addition to card balance
hist(Inquries_prob, main = "Histogram of predictions based on 
     CustomScore2 + Inquiry model")

## Change probability into class
Inquries_pred = ifelse(Inquries_prob > 0.5, 1,0)

#### Evaluating model
## Accuracy
error_Inquries = mean(Inquries_pred != risk_Te$BadFlag)
print(paste('Accuracy = ', 1 - error_Inquries))

# The model accuracy is 75.62%. 

## Precision & Recall & F-statistics
Inquries_table = table(as.factor(Inquries_pred),risk_Te$BadFlag)
recall(Inquries_table)
precision(Inquries_table)
F_meas(Inquries_table)

## Confusion matrix
Inquries_tibble = tibble("True_BadFlag" = risk_Te$BadFlag, "Prediction_BadFlag" = as.factor(Inquries_pred))
Inquries_cvms = confusion_matrix(targets = Inquries_tibble$True_BadFlag, predictions = Inquries_tibble$Prediction_BadFlag)
plot_confusion_matrix(
  Inquries_cvms$`Confusion Matrix`[[1]],
  add_sums = TRUE,
  sums_settings = sum_tile_settings(
    palette = "Oranges",
    label = "Total",
    tc_tile_border_color = "black"
  )
)

## ROC and AUC curve
ROCPred_Inquries = prediction(Inquries_pred, risk_Te$BadFlag)
ROC_Inquries = performance(ROCPred_Inquries, measure = "tpr", x.measure = "fpr")

auc_Inquries = performance(ROCPred_Inquries, measure = "auc")
auc_Inquries = auc_Inquries@y.values[[1]]

plot(ROC_Inquries)
plot(ROC_Inquries, colorize = TRUE,
     main = "ROC Curve_Score2+Inquiry")
abline(a = 0, b = 1)

auc_Inquries = round(auc_Inquries,4)
legend(.6,.4,auc_Inquries, title = "AUC",cex=1)

# According to the model results, in addition to Custome Score 2, InquriesInLast6Months + Custom Score 2 has are slightly better performance than credit card balance + Custom Score 2.
# However, InquriesInLast6Months + Custom Score 2 model has slightly lower result comparing to the only Custom Score 2 fits into the model. 
# So, the conclusion is that credit card balance and InquriesInLast6Months have litte effect in predicting risk in addtion to the best score.