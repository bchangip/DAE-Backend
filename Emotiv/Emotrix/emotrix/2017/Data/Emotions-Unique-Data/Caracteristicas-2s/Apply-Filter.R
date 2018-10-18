library(readr)
library(signal)
library(data.table)
setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data/Emotions-Unique-Data/Caracteristicas-1s")
files <- list.files(path="D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data/Emotions-Unique-Data/Caracteristicas-1s", pattern="csv$", full.names=FALSE, recursive=FALSE)
folder <- "Filter"
#BAND PASS FILTER 
#arguments
# n: filter order
# W: (low, high) / Nyquist Frequency
# type: Pass Band
# plane: analog filter
bf <- butter(n=1,W=c(0.1, 30)/1024, type="pass",plane="s")

for(i in 1:length(files)){
  csv <- read_csv(files[i], col_types = cols(`Exact Time` = col_double()))
  DT <- data.table(csv)
  
  DT$F3 <- filter(bf,DT$F3)
  DT$F4 <- filter(bf,DT$F4)
  DT$AF3 <- filter(bf,DT$AF3)
  DT$AF4 <- filter(bf,DT$AF4)
  DT$O1 <- filter(bf,DT$O1)
  DT$O2 <- filter(bf,DT$O2)

  
  dir.create(folder, showWarnings = FALSE)
  write.csv(DT, file.path(folder, files[i]), row.names=FALSE)
}