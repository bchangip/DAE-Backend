#Metodo: Logistic Regression
#Mario Barrientos
#Carne 13039

#Importo Train Data 
TrainDataC <- read.csv("C:/Users/mariobsdd/Desktop/Proyecto Machine/TrainDataC.csv")
View(TrainDataC)
   
#TrainDataC$PG  0.08167    0.02697   3.028  0.00581 **
#TrainDataC$GF 0.025744   0.008161   3.154  0.00429 **
#TrainDataC$ASIST  0.03748    0.01153   3.250   0.0034 **
#TrainDataC$PROM_GF  0.31287    0.11039   2.834  0.00917 **
#TrainDataC$Disp_arco 0.007889   0.002365   3.336  0.00276 **
#TrainDataC$mejor_promgol 0.005448   0.002787   1.954   0.0624 [][]
#TrainDataC$Pos 0.0018438  0.0004189   4.401  0.00019 ***
#TrainDataC$F_Sufr 0.003841   0.001145   3.356  0.00263 **
#TrainDataC$R_Uefa  0.029975   0.007978   3.757  0.00097 ***

#corro el modelo usando Logistic Regresion
lmTr<-glm(TrainDataC$Equipo~TrainDataC$PG+TrainDataC$GF+TrainDataC$ASIST+TrainDataC$PROM_GF+TrainDataC$Disp_arco+TrainDataC$F_Sufr+TrainDataC$R_Uefa+TrainDataC$Pos)
> summary(lmTr)


#----------- SVM (Usado SOLO para clasificar data) ----------------------------------
TrainEquipo <- read.table("~/TrainEquipo.csv", header=TRUE, quote="\"")
View(TrainEquipo)
TrainVariables <- read.csv("C:/Users/mariobsdd/Desktop/TrainVariables.csv")
View(TrainVariables)
svm.model<-svm(TrainVariables,TrainEquipo)
svm.predict<-predict(svm.model,TrainVariables)
plot(svm.model,TrainVariables)
roundprediction<-round(svm.predict)
TrainEquipo2<-unlist(TrainEquipo)
table(roundprediction,TrainEquipo2)
