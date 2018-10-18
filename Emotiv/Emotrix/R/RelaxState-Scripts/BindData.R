library(readr)
library(wavelets)
library(data.table)
library(signal)

setwd("C:/Users/mario/Desktop/Emotrix/cleanData")

files <- list.files(path="C:/Users/mario/Desktop/Emotrix/cleanData", pattern="csv$", full.names=FALSE, recursive=FALSE)


df <- data.frame(matrix(, nrow=0, ncol=17))
colnames(df)<- c("Image/Audio","Time","ExactTime","F3","F3_Quality","F4","F4_Quality","AF3","AF3_Quality","AF4","AF4_Quality","O1","O1_Quality","O2","O2_Quality","Emotion","Teorico")
for(i in 1:length(files)){
  csv <- read_csv(files[i], col_types = cols(`Exact Time` = col_double()))
  df<-rbind(df,csv)
}

# df2 <- unique(df)
# df2$`Exact Time` <- NULL
# df2 <- unique(df2)
df2 <- df

#Info para saber cuanta data de non-relax y relax tengo en total previo a separar en TRAIN y TEST
nonrelax <- nrow(df2[(df$Emotion == "NON-RELAX"),])
relax <- nrow(df2[(df2$Emotion == "RELAX"),])
info <- data.frame(relax,nonrelax,nrow(df2))
View(info)


#Separar en train y test
install.packages("caTools")
library("caTools")
createSample<-function(data,column)
{
  require(caTools)
  sample <- sample.split(data[,c(column)], SplitRatio = .70)
  return(sample)
}
temp_data <- df2
#Un ejemplo de como usarla
#la funcion tiene 2 argumentos: 
#1- el dataframe de datos
#2- el nombre de cualquier columna del dataframe
sample<-createSample(df2, "Emotion")
train<-subset(df2, sample==TRUE)
test<-subset(df2, sample==FALSE)


#------------]
#Agregarle a la data una columna con el numero de observacion distinto que despues usara wavelets tanto para TRAIN como para TEST
temp= -1
cont = -1
obs = c()
for (i in 1:nrow(train)){
  seg <- train$Time[i]
  if(seg != temp){
    cont = cont + 1
    obs[i] = cont
    temp = seg
  }
  if(seg == temp){
    obs[i] = cont
  }
  print(i)
}
train <- cbind(train,obs)
write.csv(train, file.path("C:/Users/mario/Desktop/Emotrix/finalData/train.csv"), row.names=FALSE)



temp= -1
cont = -1
obs = c()
for (i in 1:nrow(test)){
  seg <- test$Time[i]
  if(seg != temp){
    cont = cont + 1
    obs[i] = cont
    temp = seg
  }
  if(seg == temp){
    obs[i] = cont
  }
  print(i)
}
test <- cbind(test,obs)
write.csv(test, file.path("C:/Users/mario/Desktop/Emotrix/finalData/test.csv"), row.names=FALSE)


#escribir toda la data
write.csv(df, file.path("C:/Users/mario/Desktop/Emotrix/finalData/alldata.csv"), row.names=FALSE)

#escribir toda la data limpia
write.csv(df2, file.path("C:/Users/mario/Desktop/Emotrix/finalData/alldataclean.csv"), row.names=FALSE)
