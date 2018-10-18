#Mario Barrientos - 13039
#KMeans - Clustering

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
# sample<-createSample(total, "teorico")
sample <- sample.split(total$teorico, SplitRatio = .70)
train<-subset(total, sample==TRUE)
test<-subset(total, sample==FALSE)
train$j <- NULL
test$j <- NULL


#--------------------------
#TRAINING
# K-Means Cluster Analysis
train2 <- na.omit(train)
test2 <- na.omit(test)

train2 <- scale(train2)
fit <- kmeans(train2, 2)
# get cluster means 
aggregate(train2,by=list(fit$cluster),FUN=sd)
# append cluster assignment
train2 <- data.frame(train2, fit$cluster)
View(train2)

dfw <- data.frame(matrix(, nrow=17721, ncol=2))
colnames(dfw) <- c("teorico","kmeans")
dfw$teorico <- train2$teorico
dfw$kmeans <- train2$fit.cluster
a = table(dfw)
accuracy <- (a[1]+a[4])/sum(a)
print(accuracy)



#_------------------------------
#TESTING
test2 <- scale(test2)
fit <- kmeans(test2, 2)
# get cluster means 
aggregate(test2,by=list(fit$cluster),FUN=sd)
# append cluster assignment
test2 <- data.frame(test2, fit$cluster)
#View(test2)

dfw <- data.frame(matrix(, nrow=7597, ncol=2))
colnames(dfw) <- c("teorico","kmeans")
dfw$teorico <- test2$teorico
dfw$kmeans <- test2$fit.cluster
a = table(dfw)
accuracy <- (a[1]+a[4])/sum(a)
# print(a)
print(accuracy)