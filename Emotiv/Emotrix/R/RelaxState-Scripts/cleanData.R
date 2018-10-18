#Script para sanitizar toda la data leida durante las experimentaciones
#utilizar solo data del estado de relajacion
#Mario Barrientos - 13039

#Librerias

install.packages("readr")
install.packages("data.table")

library(readr)
library(data.table)

setwd("C:/Users/mario/Desktop/Emotrix/allData")


files <- list.files(path="C:/Users/mario/Desktop/Emotrix/allData", pattern="csv$", full.names=FALSE, recursive=FALSE)
folder <- "../cleanData"

for(i in 1:length(files)){
  csv <- read_csv(files[i], col_types = cols(`Exact Time` = col_double()))
  csv$`Selected Emotion`<-NULL
  csv$Teorico <- ifelse(csv$Emotion == "NON-RELAX",0,1)
  DT <- data.table(unique(csv[(csv$Emotion == "NON-RELAX" | csv$Emotion == "RELAX"),]))
  
  if("Image/Color" %in% colnames(DT)){
    setnames(DT,"Image/Color","Image/Audio")
  } else {
    setnames(DT,"Audio","Image/Audio")
  }
  
  dir.create(folder, showWarnings = FALSE)
  write.csv(DT, file.path(folder, files[i]), row.names=FALSE)
}
