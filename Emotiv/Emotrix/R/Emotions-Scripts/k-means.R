########################### HAPPY VS SAD ############################
library(readr)
library(wavelets)
library(data.table)
library(signal)

setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data/Emotions-Unique-Data/Caracteristicas-IMG-W")

csv_happy <- read_csv("Training-Happy.csv")
csv_sad <- read_csv("Training-Sad.csv")
# csv_cross_happy <- read_csv("Cross-Happy.csv")
# csv_cross_sad <- read_csv("Cross-Sad.csv")

drops <- c("count_happy", "count_sad", "count_co_happy", "count_co_sad", "count_other", "count_co_other",
           "Image/Audio", "Time", "Exact Time"
           , "F3_Quality", "F4_Quality", "AF3_Quality", "AF4_Quality", "O1_Quality", "O2_Quality"
           ,"Emotion", "Selected Emotion", "i","t")

csv_happy <- csv_happy[ , !(names(csv_happy) %in% drops)]
csv_sad <- csv_sad[ , !(names(csv_sad) %in% drops)]
# csv_cross_happy <- csv_cross_happy[ , !(names(csv_cross_happy) %in% drops)]
# csv_cross_sad <- csv_cross_sad[ , !(names(csv_cross_sad) %in% drops)]

csv_happy$ID <- 1
csv_sad$ID <- 0
# csv_cross_happy$ID <- 1
# csv_cross_sad$ID <- 0

total <- rbind(csv_happy, csv_sad)
# total <- rbind(total, csv_cross_happy)
# total <- rbind(total, csv_cross_sad)
# total <- rbind(total, csv_other)
# total <- rbind(total, csv_cross_other)
#Separar en train y test
#install.packages("caTools")
library("caTools")

#Un ejemplo de como usarla
#la funcion tiene 2 argumentos: 
#1- el dataframe de datos
#2- el nombre de cualquier columna del dataframe
# sample<-createSample(total, "teorico")
sample <- sample.split(total$ID, SplitRatio = .70)
train<-subset(total, sample==TRUE)
test<-subset(total, sample==FALSE)


#--------------------------
#TRAINING
# K-Means Cluster Analysis
train <- scale(train)
fit <- kmeans(train, 2)
# get cluster means 
aggregate(train,by=list(fit$cluster),FUN=sd)
# append cluster assignment
train <- data.frame(train, fit$cluster)
#View(train)

n <- nrow(train)
dfw <- data.frame(matrix(, nrow=n, ncol=2))
colnames(dfw) <- c("ID","kmeans")
dfw$ID <- train$ID
dfw$kmeans <- train$fit.cluster
a <- table(dfw)
accuracy <- (a[1]+a[4])/sum(a)
print(accuracy)



#_------------------------------
#TESTING
test <- scale(test)
fit <- kmeans(test, 2)
# get cluster means 
aggregate(test,by=list(fit$cluster),FUN=sd)
# append cluster assignment
test <- data.frame(test, fit$cluster)
#View(test2)
n <- nrow(test)
dfw <- data.frame(matrix(, nrow=n, ncol=2))
colnames(dfw) <- c("ID","kmeans")
dfw$ID <- test$ID
dfw$kmeans <- test$fit.cluster
a <- table(dfw)
accuracy <- (a[1]+a[4])/sum(a)
print(accuracy)


