library(readr)
library(wavelets)
library(data.table)
library(signal)

setwd("C:/Users/mario/Desktop/Emotrix/cleanData")

files <- list.files(path="C:/Users/mario/Desktop/Emotrix/cleanData", pattern="csv$", full.names=FALSE, recursive=FALSE)

folder <- "../tiempos-alpha/O2"

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
  
  file_name <- gsub(".csv", "", files[i])
  
  initial_time <- DT$Time[1]
  DT$Time <- DT$Time - initial_time
  
  final_time <- DT$Time[nrow(DT)]
  array_f3 <- c(0)
  for(j in 0:final_time) {
    second_dt <- DT[Time == j]
    
    second_dt$F3 <- filter(bf,second_dt$F3)
    second_dt$F4 <- filter(bf,second_dt$F4)
    second_dt$AF3 <- filter(bf,second_dt$AF3)
    second_dt$AF4 <- filter(bf,second_dt$AF4)
    second_dt$O1 <- filter(bf,second_dt$O1)
    second_dt$O2 <- filter(bf,second_dt$O2)
    
    result = tryCatch({
      wt_f3 <- dwt(as.numeric(second_dt$F3), filter='d4', n.levels=4, boundary="periodic", fast=FALSE)
      wt_f4 <- dwt(as.numeric(second_dt$F4), filter='d4', n.levels=4, boundary="periodic", fast=FALSE)
      wt_af3 <- dwt(as.numeric(second_dt$AF3), filter='d4', n.levels=4, boundary="periodic", fast=FALSE)
      wt_af4 <- dwt(as.numeric(second_dt$AF4), filter='d4', n.levels=4, boundary="periodic", fast=FALSE)
      wt_o1 <- dwt(as.numeric(second_dt$O1), filter='d4', n.levels=4, boundary="periodic", fast=FALSE)
      wt_o2 <- dwt(as.numeric(second_dt$O2), filter='d4', n.levels=4, boundary="periodic", fast=FALSE)
      
      f3_delta <- mean(as.numeric(wt_o2@W$W1))
      f3_theta <- mean(as.numeric(wt_o2@W$W2))
      f3_alfa <- mean(as.numeric(wt_o2@W$W3))
      f3_beta <- mean(as.numeric(wt_o2@W$W4))
      
      max_f3 <- max(f3_delta, f3_theta, f3_alfa, f3_beta)
      
      if(f3_delta == max_f3){
        array_f3[j] <- 1
      }
      
      if(f3_theta == max_f3){
        array_f3[j] <- 2
      }
      
      if(f3_alfa == max_f3){
        array_f3[j] <- 3
      }
      
      if(f3_beta == max_f3){
        array_f3[j] <- 4
      }  
    }, warning = function(w) {
      array_f3[j] <- 0
    }, error = function(e) {
      array_f3[j] <- 0
    }, finally = {
      
    })
  }
  
  #INFORMACION DE LOS TIEMPOS EN DONDE PREDOMINA ONDA ALFA
  
  #para obtener cuanto tiempo predomina alfa
  a <- table(array_f3)
  tiempo_total <- c("Tiempo Total", unname(a[names(a)== 3]))
  
  #para obtener el primer segundo en donde aparece alfa
  tiempo_inicial <- c("Primer Segundo", which.max(array_f3 == 3) -1)
  
  #para obtener el segundo en donde aparece la mayor secuencia de alfa
  t <- 0
  temp_t <- 0
  flag <- FALSE
  count <- 0
  temp_count <- 0
  alfa <- 3
  for(k in 1:length(array_f3)) {
    if(!is.na(array_f3[k])){
      if(array_f3[k] == alfa){
        if(flag){
          temp_count <- temp_count + 1
          
        } else {
          temp_t <- k - 1
          flag <- TRUE
          temp_count <- temp_count + 1
        }
      } else {
        if(flag & temp_count > count){
          count <- temp_count
          t <- temp_t
          temp_count <- 0
          temp_t <- 0
        }
        flag <- FALSE
        
      }
    }
  }
  
  tiempo_seq <- c("Segundo Secuencia", t)
  
  final_table <- rbind(tiempo_total, tiempo_inicial, tiempo_seq)
  
  dir.create(folder, showWarnings = FALSE)
  write.csv(final_table, file.path(folder, files[i]), row.names=FALSE)
  
  
  #PARA IMPRIMIR LAS IMAGENES
  # image_name <-  paste(file_name, ".png") 
  # image_path  <- paste("C:/Users/mario/Desktop/Emotrix/alpha-time/AF3/AF3", image_name)
  # png(filename=image_path)
  # plot(array_f3,yaxt="n",xlab = "Tiempo(s)", ylab = "Onda más Presente", col="blue", main="Lectura de Electrodo AF4")
  # axis(2, at=1:5, labels=c("Delta","Theta","Alpha","Beta","Gamma"))
  # dev.off()
}

remove(csv)
remove(DT)
remove(second_dt)
remove(array_f3)
remove(bf)
remove(f3_alfa)
remove(f3_beta)
remove(f3_delta)
remove(f3_theta)
remove(file_name)
remove(files)
remove(final_time)
remove(i)
remove(image_name)
remove(image_path)
remove(initial_time)
remove(j)
remove(max_f3)