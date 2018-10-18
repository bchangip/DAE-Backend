#Codigo para correr Logistic Regression
#Mario Barrientos - 13039
#http://www.statmethods.net/advstats/glm.html
modeloLR<-glm(poisonous~., data=hongosTrain, family=binomial())
hongosTrainLR<-hongosTrain
hongosTrainLR$veiltype<-NULL
modeloLR<-glm(poisonous~., data=hongosTrainLR, family=binomial())
#summary(modeloLR) y puedo ver en excel por p-value que variables son las mas importantes,
#cuales son sus coeficientes, etc. 


test<-predict(modeloLR, hongosTest, type="response")
