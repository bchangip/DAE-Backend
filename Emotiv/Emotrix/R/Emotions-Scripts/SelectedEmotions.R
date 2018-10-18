#Script para obtener emociones presionadas por cada persona
# Diego Jacobs - 13160

install.packages("readr")
install.packages("data.table")

library(readr)
library(data.table)

setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data")

folder <- "Selected-Emotions"

files <- list.files(path="D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data", pattern="csv$", full.names=FALSE, recursive=FALSE)

for(i in 1:length(files)){
  csv <- read_csv(files[i], col_types = cols(`Exact Time` = col_double()))
  DT <- data.table(csv)
  
  if("Image/Color" %in% colnames(DT)){
    tuple <-unique(DT[Emotion != "NON-RELAX"  & Emotion != "RELAX", c("Image/Color", "Time")])
  } else {
    tuple <-unique(DT[Emotion != "NON-RELAX"  & Emotion != "RELAX", c("Audio", "Time")])
  }
  
  emotions <- DT[Emotion != "NON-RELAX"  & Emotion != "RELAX" & `Selected Emotion` != "NA"]
  
  if("Image/Color" %in% colnames(DT)){
    table <- emotions[, c("Image/Color", "Time", "Emotion", "Selected Emotion")]
  } else {
    table <- emotions[, c("Audio", "Time", "Emotion", "Selected Emotion")]
  }
  
  final_table <- merge(tuple, table, all.x=TRUE)
  
  table_order <- unique(final_table[order(final_table$Time)])
  
  dir.create(folder, showWarnings = FALSE)
  write.csv(table_order, file.path(folder, files[i]), row.names=FALSE)
}

remove(csv)
remove(DT)
remove(tuple)
remove(emotions)
remove(table)
remove(final_table)
remove(table_order)
remove(folder)
remove(files)
remove(i)

print("Done")