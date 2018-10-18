#Codigo para correr Logistic Regression
#Diego Jacobs - 13160
#http://www.statmethods.net/advstats/glm.html
library(readr)

setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data/Emotions-Unique-Data/Caracteristicas-Audio-W")

#Happy vs Sad
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

#Separar en train y test
#install.packages("caTools")
library("caTools")

sample <- sample.split(total$ID, SplitRatio = .70)
train<-subset(total, sample==TRUE)
test<-subset(total, sample==FALSE)

modelo_lr_hs<-glm(ID~., data=train, family=binomial())

final_hs <- predict(modelo_lr_hs, test, type="response")

a <- table(test$ID, final_hs > 0.5)
accuracy <- (a[1]+a[4])/sum(a)
print(accuracy*100)

library(readr)
setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data/Emotions-Unique-Data/Caracteristicas-ALFA-W")
#Happy vs Other
csv_happy <- read_csv("Training-Happy.csv")
csv_other <- read_csv("Training-Other.csv")
csv_cross_happy <- read_csv("Cross-Happy.csv")
csv_cross_other <- read_csv("Cross-Other.csv")

drops <- c("count_happy", "count_sad", "count_co_happy", "count_co_sad", "count_other", "count_co_other",
           "Image/Audio", "Time", "Exact Time"
           , "F3_Quality", "F4_Quality", "AF3_Quality", "AF4_Quality", "O1_Quality", "O2_Quality"
           ,"Emotion", "Selected Emotion", "i","t")


csv_happy <- csv_happy[ , !(names(csv_happy) %in% drops)]
csv_other <- csv_other[ , !(names(csv_other) %in% drops)]
csv_cross_happy <- csv_cross_happy[ , !(names(csv_cross_happy) %in% drops)]
csv_cross_other <- csv_cross_other[ , !(names(csv_cross_other) %in% drops)]

csv_happy$ID <- 1
csv_other$ID <- 0
csv_cross_happy$ID <- 1
csv_cross_other$ID <- 0

total <- rbind(csv_happy, csv_other)
total <- rbind(total, csv_cross_happy)
total <- rbind(total, csv_cross_other)

#Separar en train y test
install.packages("caTools")
library("caTools")

sample <- sample.split(total$ID, SplitRatio = .70)
train<-subset(total, sample==TRUE)
test<-subset(total, sample==FALSE)

modelo_lr_ho<-glm(ID~., data=train, family=binomial())


final_ho <- predict(modelo_lr_ho, test, type="response")

a <- table(test$ID, final_ho > 0.5)

accuracy <- (a[1]+a[4])/sum(a)
print(accuracy*100)

library(readr)
setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data/Emotions-Unique-Data/Caracteristicas-1s/Filter")
#Other vs Sad
csv_sad <- read_csv("Training-Sad.csv")
csv_other <- read_csv("Training-Other.csv")
csv_cross_sad <- read_csv("Cross-Sad.csv")
csv_cross_other <- read_csv("Cross-Other.csv")

drops <- c("count_happy", "count_sad", "count_co_happy", "count_co_sad", "count_other", "count_co_other",
           "Image/Audio", "Time", "Exact Time"
           , "F3_Quality", "F4_Quality", "AF3_Quality", "AF4_Quality", "O1_Quality", "O2_Quality"
           ,"Emotion", "Selected Emotion", "i","t")

csv_other <- csv_other[ , !(names(csv_other) %in% drops)]
csv_sad <- csv_sad[ , !(names(csv_sad) %in% drops)]
csv_cross_other <- csv_cross_other[ , !(names(csv_cross_other) %in% drops)]
csv_cross_sad <- csv_cross_sad[ , !(names(csv_cross_sad) %in% drops)]

csv_sad$ID <- 1
csv_other$ID <- 0
csv_cross_sad$ID <- 1
csv_cross_other$ID <- 0

total <- rbind(csv_sad, csv_other)
total <- rbind(total, csv_cross_sad)
total <- rbind(total, csv_cross_other)

#Separar en train y test
install.packages("caTools")
library("caTools")

sample <- sample.split(total$ID, SplitRatio = .70)
train<-subset(total, sample==TRUE)
test<-subset(total, sample==FALSE)

modelo_lr_so<-glm(ID~., data=train, family=binomial())

final_so <- predict(modelo_lr_so, test, type="response")

a <- table(test$ID, final_so > 0.5)

accuracy <- (a[1]+a[4])/sum(a)
print(accuracy*100)