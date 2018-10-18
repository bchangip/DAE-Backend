#Diego Jacobs - 13160
library(readr)
library(data.table)

setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data/Emotions-Unique-Data")

files <- list.files(path="D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data/Emotions-Unique-Data", pattern="csv$", full.names=FALSE, recursive=FALSE)

happy <- 0
sad <- 0
other <- 0

for(i in 1:length(files)){
  csv <- read_csv(files[i], col_types = cols(`Exact Time` = col_double()))
  DT <- data.table(csv)
  
  tuple <- DT[`Selected Emotion` != "", c("Image/Audio", "Selected Emotion")]
  
  happy <- happy + nrow(tuple[`Selected Emotion` == "happy"])
  sad <- sad + nrow(tuple[`Selected Emotion` == "sad"])
  other <- other + nrow(tuple[`Selected Emotion` == "other"])
}




