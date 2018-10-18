# Diego Jacobs - 13160

install.packages("readr")
install.packages("data.table")

library(readr)
library(data.table)

setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data")

files <- list.files(path="D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data", pattern="csv$", full.names=FALSE, recursive=FALSE)
folder <- "Emotions-Unique-Data"

for(i in 1:length(files)){
  csv <- read_csv(files[i], col_types = cols(`Exact Time` = col_double()))
  DT <- data.table(unique(csv[(csv$Emotion != "NON-RELAX" & csv$Emotion != "RELAX"),]))
  
  if("Image/Color" %in% colnames(DT)){
    setnames(DT,"Image/Color","Image/Audio")
  } else {
    setnames(DT,"Audio","Image/Audio")
  }
  
  dir.create(folder, showWarnings = FALSE)
  write.csv(DT, file.path(folder, files[i]), row.names=FALSE)
}




