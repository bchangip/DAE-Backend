#Mario Barrientos - 13039
#Regresion Logistica 
library(readr)
library(wavelets)
library(data.table)
library(signal)

library(readr)
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
train<-subset(total, sample==TRUE)
test<-subset(total, sample==FALSE)
train$j <- NULL
test$j <- NULL

#---------------------------------------------------
#TRAINING
train <- na.omit(train)
test <- na.omit(test)

modeloLR<-glm(teorico~., data=train, family=binomial())
# summary(modeloLR)
testmodel<-predict(modeloLR, test, type="response")
a <- table(test$teorico, testmodel > 0.5)
accuracy <- (a[1]+a[4])/sum(a)
print(a)
print(accuracy)









