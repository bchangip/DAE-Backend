#Mario Barrientos - 13039
#Support Vectors Machine
library(readr)
library(wavelets)
library(data.table)
library(signal)

trainWfiltered <- read_csv("C:/Users/mario/Desktop/Emotrix/finalData/trainWfiltered.csv")
testWfiltered <- read_csv("C:/Users/mario/Desktop/Emotrix/finalData/testWfiltered.csv")

total <- rbind(trainWfiltered,testWfiltered)
colnames(total)[2] <- "teorico"

#Separar en train y test
install.packages("caTools")
library("caTools")
createSample<-function(data,column)
{
  require(caTools)
  sample <- sample.split(data[,c(column)], SplitRatio = .70)
  return(sample)
}
#Un ejemplo de como usarla
#la funcion tiene 2 argumentos: 
#1- el dataframe de datos
#2- el nombre de cualquier columna del dataframe
sample<-createSample(total, "teorico")
sample <- sample.split(total$teorico, SplitRatio = .70)
train<-subset(total, sample==TRUE)
test<-subset(total, sample==FALSE)
train$j <- NULL
test$j <- NULL


#--------------------------
#TRAINING


install.packages("e1071")
library(e1071)

train <- na.omit(train)
test <- na.omit(test)

svm.model<-svm(teorico~., data=train, probability = TRUE)

svm.predict<-predict(svm.model,test, probability = TRUE, na.action = na.exclude)
roundprediction<-round(svm.predict)
t2<-unlist(test$teorico)
a <- table(roundprediction,t2)
print(a)
accuracy <- (a[1]+a[4])/sum(a)
print(accuracy)

