#install.packages('neuralnet')
library(neuralnet)
library(readr)

setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data/Emotions-Unique-Data/Caracteristicas-1s/Filter")

csv_happy <- read_csv("Training-Happy.csv")
csv_sad <- read_csv("Training-Sad.csv")
csv_cross_happy <- read_csv("Cross-Happy.csv")
csv_cross_sad <- read_csv("Cross-Sad.csv")

drops <- c("count_happy", "count_sad", "count_co_happy", "count_co_sad", "count_other", "count_co_other",
           "Image/Audio", "Time", "Exact Time"
           , "F3_Quality", "F4_Quality", "AF3_Quality", "AF4_Quality", "O1_Quality", "O2_Quality"
           ,"Emotion", "Selected Emotion", "i","t")

csv_happy <- csv_happy[ , !(names(csv_happy) %in% drops)]
csv_sad <- csv_sad[ , !(names(csv_sad) %in% drops)]
csv_cross_happy <- csv_cross_happy[ , !(names(csv_cross_happy) %in% drops)]
csv_cross_sad <- csv_cross_sad[ , !(names(csv_cross_sad) %in% drops)]

csv_happy$ID <- 1
csv_sad$ID <- 0
csv_cross_happy$ID <- 1
csv_cross_sad$ID <- 0

total <- rbind(csv_happy, csv_sad)
total <- rbind(total, csv_cross_happy)
total <- rbind(total, csv_cross_sad)

#Separar en train y test
install.packages("caTools")
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
# Neural Network Cluster Analysis
train2 <- na.omit(train)
test2 <- na.omit(test)

n <- nrow(train2)
dft <- data.frame(matrix(, nrow=n, ncol=1))
colnames(dft) <- "ID"
dft$ID <- train2$ID
ttrain <- train2
ttrain$ID <- NULL
feats <- names(ttrain)
# Concatenate strings
f <- paste(feats,collapse=' + ')
y <- names(dft)
f <- paste(y,' ~ ',f)

# Convert to formula
f <- as.formula(f)


#TRAINING
library(neuralnet)
nn <- neuralnet(f,train2,hidden=c(2,2,2),linear.output=FALSE)

# Compute Predictions off Test Set
predicted.nn.values <- compute(nn,test2)

# Check out net.result
print(head(predicted.nn.values$net.result))


predicted.nn.values$net.result <- sapply(predicted.nn.values$net.result,round,digits=0)
a <- table(test$Private,predicted.nn.values$net.result)
print(a)
accuracy <- (a[1]+a[4])/sum(a)
print(accuracy)