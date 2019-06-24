---
title: "Pratical Machine Learning Course Project"
author: "Lu Li"
date: "June 21, 2019"
output: 
 html_document:
  keep_md: true
---



## Summary
The objective of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.


##Data Loading
The training and testing data were downloaded from the Internet using the following website.

The training data
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

```r
rawtraindata<-read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
rawtestdata<-read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
```

##Loading the requried packages

```r
library(dplyr)
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
```
##Data cleaning and processing
By looking through the raw dataset(rawtraindata), it is noticed that when a NA appears in the sencond row of the dataset, the observations in its corresponding variable will be mostly NA. For example, varible krutosis_roll_belt has a NA in its second row (rawtrindata$krutosis_roll_belt[2,] is NA), all the ovservations of krutosis_roll_belt is NA. This trick was then used in the data cleaning step of this project. The first 7 varibles including the ID informations were also removed since they don't contribute to tracking of the movement. The train data and test data for the assignment were treated in the same manner.


```r
traindata1<-rawtraindata[,!is.na(rawtraindata[2,])]
testdata1<-rawtestdata[,!is.na(rawtestdata[2,])]
traindata2<-traindata1[-c(1:7)]
testdata2<-testdata1[-c(1:7)]
dim(traindata2)
```

```
## [1] 19622    53
```

```r
dim(testdata2)
```

```
## [1] 20 53
```
##Data partitioning
The training data was partioned into two parts, 70% for myTraining, 30% for myTesting

```r
inTrain <- createDataPartition(y=traindata2$classe, p=0.7, list=FALSE)
myTraining <- traindata2[inTrain, ]
myTesting <- traindata2[-inTrain, ]
dim(myTraining)
```

```
## [1] 13737    53
```
##Machine leaning model: Decision Tree

```r
set.seed(12345)
modFitDecTree <- rpart(classe ~ ., data=myTraining, method="class")
predictDecTree <- predict(modFitDecTree, newdata=myTesting, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, myTesting$classe)
confMatDecTree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1481  175   24   59   52
##          B   61  677   76  115  128
##          C   38  168  830  129   98
##          D   70   81   75  621   62
##          E   24   38   21   40  742
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7393          
##                  95% CI : (0.7279, 0.7505)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6696          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8847   0.5944   0.8090   0.6442   0.6858
## Specificity            0.9264   0.9199   0.9109   0.9415   0.9744
## Pos Pred Value         0.8269   0.6405   0.6572   0.6832   0.8578
## Neg Pred Value         0.9529   0.9043   0.9576   0.9311   0.9323
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2517   0.1150   0.1410   0.1055   0.1261
## Detection Prevalence   0.3043   0.1796   0.2146   0.1545   0.1470
## Balanced Accuracy      0.9055   0.7572   0.8599   0.7928   0.8301
```
##Machine leaning model:Generalized Boosted Model

```r
set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=myTraining, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
predictGBM <- predict(modFitGBM, newdata=myTesting)
confMatGBM <- confusionMatrix(predictGBM, myTesting$classe)
confMatGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1649   48    0    2    3
##          B   20 1045   35    4    1
##          C    1   42  980   33    9
##          D    2    4    9  921   20
##          E    2    0    2    4 1049
## 
## Overall Statistics
##                                          
##                Accuracy : 0.959          
##                  95% CI : (0.9537, 0.964)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9482         
##  Mcnemar's Test P-Value : 4.577e-06      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9851   0.9175   0.9552   0.9554   0.9695
## Specificity            0.9874   0.9874   0.9825   0.9929   0.9983
## Pos Pred Value         0.9689   0.9457   0.9202   0.9634   0.9924
## Neg Pred Value         0.9940   0.9803   0.9905   0.9913   0.9932
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2802   0.1776   0.1665   0.1565   0.1782
## Detection Prevalence   0.2892   0.1878   0.1810   0.1624   0.1796
## Balanced Accuracy      0.9862   0.9524   0.9688   0.9741   0.9839
```
##Machine leaning model:Random Forest

```r
set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=myTraining, method="rf",
                          trControl=controlRF)
predictRandForest <- predict(modFitRandForest, newdata=myTesting)
confMatRandForest <- confusionMatrix(predictRandForest, myTesting$classe)
confMatRandForest
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    4    0    0    0
##          B    1 1132   10    0    0
##          C    0    3 1016   20    0
##          D    0    0    0  944    3
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                          
##                Accuracy : 0.993          
##                  95% CI : (0.9906, 0.995)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9912         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9939   0.9903   0.9793   0.9972
## Specificity            0.9991   0.9977   0.9953   0.9994   1.0000
## Pos Pred Value         0.9976   0.9904   0.9779   0.9968   1.0000
## Neg Pred Value         0.9998   0.9985   0.9979   0.9959   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1924   0.1726   0.1604   0.1833
## Detection Prevalence   0.2850   0.1942   0.1766   0.1609   0.1833
## Balanced Accuracy      0.9992   0.9958   0.9928   0.9893   0.9986
```
##Conclusions
The accuracy of the 3 regression modeling methods above are:
Decision Tree : 0.7446
GBM : 0.9682
Random Forest : 0.9941
From the above disscussion, the results indicted that random forest was the best model in this study. Therefore, the random forest model was applied to the test data for the assignment in the next part.

##Applying the Random Forest Model to the test data set

```r
predicttest <- predict(modFitRandForest, newdata=testdata2)
predicttest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

