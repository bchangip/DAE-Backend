#Mario Barrientos - 13039 
#NEURAL NETWORKS
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
# Neural Network Cluster Analysis
train2 <- na.omit(train)
test2 <- na.omit(test)

dft <- data.frame(matrix(, nrow=17724, ncol=1))
colnames(dft) <- "teorico"
dft$teorico <- train2$teorico
ttrain <- train2
ttrain$teorico <- NULL
feats <- names(ttrain)
# Concatenate strings
f <- paste(feats,collapse=' + ')
y <- names(dft)
f <- paste(y,' ~ ',f)

# Convert to formula
f <- as.formula(f)


#TRAINING
install.packages('neuralnet')
library(neuralnet)
nn <- neuralnet(f,train2,hidden=c(2,2,2), threshold = 0.1)

# Compute Predictions off Test Set
predicted.nn.values <- compute(nn,test)

# Check out net.result
print(head(predicted.nn.values$net.result))


predicted.nn.values$net.result <- sapply(predicted.nn.values$net.result,round,digits=0)
a <- table(test$Private,predicted.nn.values$net.result)
# print(a)  
accuracy <- (a[1]+a[4])/sum(a)
print(accuracy)




