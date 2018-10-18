install.packages("readr")
install.packages("wavelets")
install.packages("data.table")
install.packages("signal")
library(readr)
library(wavelets)
library(data.table)
library(signal)

setwd("D:/Diego Jacobs/Documents/Emotrix/emotrix/2017/Data")
#setwd("C:/Users/mario/Desktop/resultados")


#BAND PASS FILTER 
#arguments
# n: filter order
# W: (low, high) / Nyquist Frequency
# type: Pass Band
# plane: analog filter
bf <- butter(n=1,W=c(0.1, 30)/1024, type="pass",plane="s")


csv <- read_csv("17M2311.csv", col_types = cols(`Exact Time` = col_double()))
csv2 <- csv[(csv$Emotion != "NON-RELAX" & csv$Emotion != "RELAX"),]
#csv2 <- csv[(csv$Emotion == "NON-RELAX" | csv$Emotion == "RELAX"),]




f3 <- filter(bf,csv2$F3)
f4 <- filter(bf,csv2$F4)
af3 <- filter(bf,csv2$AF3)
af4 <- filter(bf,csv2$AF4)
o1 <- filter(bf,csv2$O1)
o2 <- filter(bf,csv2$O2)

m <- cbind(f3,af3, f4, af4, o1, o2)

remove(csv)


wt <- dwt(as.numeric(f3), filter='d4', n.levels=4, boundary="periodic", fast=FALSE)


remove(csv)
remove(csv2)
