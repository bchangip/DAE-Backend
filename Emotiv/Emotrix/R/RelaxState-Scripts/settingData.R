library(readr)
library(wavelets)
library(data.table)
library(signal)

train <- read_csv("C:/Users/mario/Desktop/Emotrix/finalData/train.csv")

#BAND PASS FILTER 
#arguments
# n: filter order
# W: (low, high) / Nyquist Frequency
# type: Pass Band
# plane: analog filter
bf <- butter(n=1,W=c(0.1, 30)/1024, type="pass",plane="s")
DT <- data.table(train)
final_time <- DT$obs[nrow(DT)]
array_f3 <- c(0)
dfw <- data.frame(matrix(, nrow=0, ncol=50))
for(j in 0:final_time) {
  second_dt <- DT[obs == j]
  
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
    
    #F3
    mean_delta_f3 <- mean(as.numeric(wt_f3@W$W1))
    mean_theta_f3 <- mean(as.numeric(wt_f3@W$W2))
    mean_alfa_f3 <- mean(as.numeric(wt_f3@W$W3))
    mean_beta_f3 <- mean(as.numeric(wt_f3@W$W4))
    sd_delta_f3 <- sd(as.numeric(wt_f3@W$W1))
    sd_theta_f3 <- sd(as.numeric(wt_f3@W$W2))
    sd_alfa_f3 <- sd(as.numeric(wt_f3@W$W3))
    sd_beta_f3 <- sd(as.numeric(wt_f3@W$W4))
    
    #F4
    mean_delta_f4 <- mean(as.numeric(wt_f4@W$W1))
    mean_theta_f4 <- mean(as.numeric(wt_f4@W$W2))
    mean_alfa_f4 <- mean(as.numeric(wt_f4@W$W3))
    mean_beta_f4 <- mean(as.numeric(wt_f4@W$W4))
    sd_delta_f4 <- sd(as.numeric(wt_f4@W$W1))
    sd_theta_f4 <- sd(as.numeric(wt_f4@W$W2))
    sd_alfa_f4 <- sd(as.numeric(wt_f4@W$W3))
    sd_beta_f4 <- sd(as.numeric(wt_f4@W$W4))
    
    #AF3
    mean_delta_af3 <- mean(as.numeric(wt_af3@W$W1))
    mean_theta_af3 <- mean(as.numeric(wt_af3@W$W2))
    mean_alfa_af3 <- mean(as.numeric(wt_af3@W$W3))
    mean_beta_af3 <- mean(as.numeric(wt_af3@W$W4))
    sd_delta_af3 <- sd(as.numeric(wt_af3@W$W1))
    sd_theta_af3 <- sd(as.numeric(wt_af3@W$W2))
    sd_alfa_af3 <- sd(as.numeric(wt_af3@W$W3))
    sd_beta_af3 <- sd(as.numeric(wt_af3@W$W4))
    
    #AF4
    mean_delta_af4 <- mean(as.numeric(wt_af4@W$W1))
    mean_theta_af4 <- mean(as.numeric(wt_af4@W$W2))
    mean_alfa_af4 <- mean(as.numeric(wt_af4@W$W3))
    mean_beta_af4 <- mean(as.numeric(wt_af4@W$W4))
    sd_delta_af4 <- sd(as.numeric(wt_af4@W$W1))
    sd_theta_af4 <- sd(as.numeric(wt_af4@W$W2))
    sd_alfa_af4 <- sd(as.numeric(wt_af4@W$W3))
    sd_beta_af4 <- sd(as.numeric(wt_af4@W$W4))
    
    #O1
    mean_delta_o1 <- mean(as.numeric(wt_o1@W$W1))
    mean_theta_o1 <- mean(as.numeric(wt_o1@W$W2))
    mean_alfa_o1 <- mean(as.numeric(wt_o1@W$W3))
    mean_beta_o1 <- mean(as.numeric(wt_o1@W$W4))
    sd_delta_o1 <- sd(as.numeric(wt_o1@W$W1))
    sd_theta_o1 <- sd(as.numeric(wt_o1@W$W2))
    sd_alfa_o1 <- sd(as.numeric(wt_o1@W$W3))
    sd_beta_o1 <- sd(as.numeric(wt_o1@W$W4))
    
    #O2
    mean_delta_o2 <- mean(as.numeric(wt_o2@W$W1))
    mean_theta_o2 <- mean(as.numeric(wt_o2@W$W2))
    mean_alfa_o2 <- mean(as.numeric(wt_o2@W$W3))
    mean_beta_o2 <- mean(as.numeric(wt_o2@W$W4))
    sd_delta_o2 <- sd(as.numeric(wt_o2@W$W1))
    sd_theta_o2 <- sd(as.numeric(wt_o2@W$W2))
    sd_alfa_o2 <- sd(as.numeric(wt_o2@W$W3))
    sd_beta_o2 <- sd(as.numeric(wt_o2@W$W4))
    
    
    temp <- data.frame(j, max(second_dt$Teorico), mean_delta_f3, mean_theta_f3, mean_alfa_f3, mean_beta_f3, sd_delta_f3, sd_theta_f3, sd_alfa_f3, sd_beta_f3,
                       mean_delta_f4, mean_theta_f4, mean_alfa_f4, mean_beta_f4, sd_delta_f4, sd_theta_f4, sd_alfa_f4, sd_beta_f4,
                       mean_delta_af3, mean_theta_af3, mean_alfa_af3, mean_beta_af3, sd_delta_af3, sd_theta_af3, sd_alfa_af3, sd_beta_af3,
                       mean_delta_af4, mean_theta_af4, mean_alfa_af4, mean_beta_af4, sd_delta_af4, sd_theta_af4, sd_alfa_af4, sd_beta_af4,
                       mean_delta_o1, mean_theta_o1, mean_alfa_o1, mean_beta_o1, sd_delta_o1, sd_theta_o1, sd_alfa_o1, sd_beta_o1,
                       mean_delta_o2, mean_theta_o2, mean_alfa_o2, mean_beta_o2, sd_delta_o2, sd_theta_o2, sd_alfa_o2, sd_beta_o2)
    
    dfw <- rbind(dfw, temp)
    print(j)
    
  }, warning = function(w) {
    array_f3[j] <- 0
  }, error = function(e) {
    array_f3[j] <- 0
  }, finally = {
  })
}

write.csv(dfw, file.path("C:/Users/mario/Desktop/Emotrix/finalData/trainWfiltered.csv"), row.names=FALSE)

#CON TEST DATA
library(readr)
test <- read_csv("C:/Users/mario/Desktop/Emotrix/finalData/test.csv")
bf <- butter(n=1,W=c(0.1, 30)/1024, type="pass",plane="s")
DT <- data.table(test)
final_time <- DT$obs[nrow(DT)]
array_f3 <- c(0)
dfw <- data.frame(matrix(, nrow=0, ncol=50))
for(j in 0:final_time) {
  second_dt <- DT[obs == j]
  
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
    
    #F3
    mean_delta_f3 <- mean(as.numeric(wt_f3@W$W1))
    mean_theta_f3 <- mean(as.numeric(wt_f3@W$W2))
    mean_alfa_f3 <- mean(as.numeric(wt_f3@W$W3))
    mean_beta_f3 <- mean(as.numeric(wt_f3@W$W4))
    sd_delta_f3 <- sd(as.numeric(wt_f3@W$W1))
    sd_theta_f3 <- sd(as.numeric(wt_f3@W$W2))
    sd_alfa_f3 <- sd(as.numeric(wt_f3@W$W3))
    sd_beta_f3 <- sd(as.numeric(wt_f3@W$W4))
    
    #F4
    mean_delta_f4 <- mean(as.numeric(wt_f4@W$W1))
    mean_theta_f4 <- mean(as.numeric(wt_f4@W$W2))
    mean_alfa_f4 <- mean(as.numeric(wt_f4@W$W3))
    mean_beta_f4 <- mean(as.numeric(wt_f4@W$W4))
    sd_delta_f4 <- sd(as.numeric(wt_f4@W$W1))
    sd_theta_f4 <- sd(as.numeric(wt_f4@W$W2))
    sd_alfa_f4 <- sd(as.numeric(wt_f4@W$W3))
    sd_beta_f4 <- sd(as.numeric(wt_f4@W$W4))
    
    #AF3
    mean_delta_af3 <- mean(as.numeric(wt_af3@W$W1))
    mean_theta_af3 <- mean(as.numeric(wt_af3@W$W2))
    mean_alfa_af3 <- mean(as.numeric(wt_af3@W$W3))
    mean_beta_af3 <- mean(as.numeric(wt_af3@W$W4))
    sd_delta_af3 <- sd(as.numeric(wt_af3@W$W1))
    sd_theta_af3 <- sd(as.numeric(wt_af3@W$W2))
    sd_alfa_af3 <- sd(as.numeric(wt_af3@W$W3))
    sd_beta_af3 <- sd(as.numeric(wt_af3@W$W4))
    
    #AF4
    mean_delta_af4 <- mean(as.numeric(wt_af4@W$W1))
    mean_theta_af4 <- mean(as.numeric(wt_af4@W$W2))
    mean_alfa_af4 <- mean(as.numeric(wt_af4@W$W3))
    mean_beta_af4 <- mean(as.numeric(wt_af4@W$W4))
    sd_delta_af4 <- sd(as.numeric(wt_af4@W$W1))
    sd_theta_af4 <- sd(as.numeric(wt_af4@W$W2))
    sd_alfa_af4 <- sd(as.numeric(wt_af4@W$W3))
    sd_beta_af4 <- sd(as.numeric(wt_af4@W$W4))
    
    #O1
    mean_delta_o1 <- mean(as.numeric(wt_o1@W$W1))
    mean_theta_o1 <- mean(as.numeric(wt_o1@W$W2))
    mean_alfa_o1 <- mean(as.numeric(wt_o1@W$W3))
    mean_beta_o1 <- mean(as.numeric(wt_o1@W$W4))
    sd_delta_o1 <- sd(as.numeric(wt_o1@W$W1))
    sd_theta_o1 <- sd(as.numeric(wt_o1@W$W2))
    sd_alfa_o1 <- sd(as.numeric(wt_o1@W$W3))
    sd_beta_o1 <- sd(as.numeric(wt_o1@W$W4))
    
    #O2
    mean_delta_o2 <- mean(as.numeric(wt_o2@W$W1))
    mean_theta_o2 <- mean(as.numeric(wt_o2@W$W2))
    mean_alfa_o2 <- mean(as.numeric(wt_o2@W$W3))
    mean_beta_o2 <- mean(as.numeric(wt_o2@W$W4))
    sd_delta_o2 <- sd(as.numeric(wt_o2@W$W1))
    sd_theta_o2 <- sd(as.numeric(wt_o2@W$W2))
    sd_alfa_o2 <- sd(as.numeric(wt_o2@W$W3))
    sd_beta_o2 <- sd(as.numeric(wt_o2@W$W4))
    
    
    temp <- data.frame(j, max(second_dt$Teorico), mean_delta_f3, mean_theta_f3, mean_alfa_f3, mean_beta_f3, sd_delta_f3, sd_theta_f3, sd_alfa_f3, sd_beta_f3,
                       mean_delta_f4, mean_theta_f4, mean_alfa_f4, mean_beta_f4, sd_delta_f4, sd_theta_f4, sd_alfa_f4, sd_beta_f4,
                       mean_delta_af3, mean_theta_af3, mean_alfa_af3, mean_beta_af3, sd_delta_af3, sd_theta_af3, sd_alfa_af3, sd_beta_af3,
                       mean_delta_af4, mean_theta_af4, mean_alfa_af4, mean_beta_af4, sd_delta_af4, sd_theta_af4, sd_alfa_af4, sd_beta_af4,
                       mean_delta_o1, mean_theta_o1, mean_alfa_o1, mean_beta_o1, sd_delta_o1, sd_theta_o1, sd_alfa_o1, sd_beta_o1,
                       mean_delta_o2, mean_theta_o2, mean_alfa_o2, mean_beta_o2, sd_delta_o2, sd_theta_o2, sd_alfa_o2, sd_beta_o2)
    
    dfw <- rbind(dfw, temp)
    print(j)
    
  }, warning = function(w) {
    array_f3[j] <- 0
  }, error = function(e) {
    array_f3[j] <- 0
  }, finally = {
  })
}

write.csv(dfw, file.path("C:/Users/mario/Desktop/Emotrix/finalData/testWfiltered.csv"), row.names=FALSE)


