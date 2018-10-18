#Instale el paquete caTools para poder usar esta funcion
#Mario Barrientos - 13039
#Script para crear train y test data

install.packages("caTools")

createSample<-function(data,column)
{
  require(caTools)
  sample <- sample.split(data[,c(column)], SplitRatio = .75)
  return(sample)
}

#Un ejemplo de como usarla
#la funcion tiene 2 argumentos: 
#1- el dataframe de datos
#2- el nombre de cualquier columna del dataframe
sample<-createSample(movies, "score")
train<-subset(movies, sample==TRUE)
test<-subset(movies, sample==FALSE)
