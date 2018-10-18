#Script para obtener el comienzo de la experimentación por cada persona
# Diego Jacobs - 13160

install.packages("readr")
install.packages("data.table")

library(readr)
library(data.table)

setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data")

folder <- "Data-Header"
files <- list.files(path="D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data", pattern="csv$", full.names=FALSE, recursive=FALSE)

for(i in 1:length(files)){
  csv <- read_csv(files[i])
  DT <- data.table(csv)
  
  dir.create(folder, showWarnings = FALSE)
  write.csv(head(DT), file.path(folder, files[i]), row.names=FALSE)
}

remove(csv)
remove(DT)
remove(files)
remove(folder)
remove(i)

print("Done")