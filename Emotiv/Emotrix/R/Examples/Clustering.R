#Clustering por varios metodos
#Mario Barrientos - 13039

#Hierchical Agglomerative
#K-means
#http://www.statmethods.net/advstats/cluster.html

#Standarize data (when necessary)
mydata <- scale(mydata)

# Ward Hierarchical Clustering
d <- dist(mydata, method = "euclidean") # distance matrix
fit_h <- hclust(d, method="ward") 
plot(fit) # display dendogram
groups <- cutree(fit, k=5) # cut tree into 5 clusters 
#groups tiene el cluster membership

# draw dendogram with red borders around the 5 clusters 
rect.hclust(fit, k=5, border="red")

# K-Means Cluster Analysis
fit <- kmeans(mydata, 5) # 5 cluster solution
# get cluster means 
aggregate(mydata,by=list(fit$cluster),FUN=sd)
# append cluster assignment
mydata <- data.frame(mydata, fit$cluster)


#DETERMINE CLUSTER NUMBER 
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))

for (i in 2:15) wss[i] <- sum(kmeans(mydata, 
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")



#EJEMPLO
library(MASS)
x1<-mvrnorm(100, mu=c(2,2), Sigma=matrix(c(1,0,0,1),
                                         2))
x2<-mvrnorm(100, mu=c(-2,-2), Sigma=matrix(c(1,0,0,1),
                                           2))
x<-matrix(nrow=200,ncol=2)
x[1:100,]<-x1
x[101:200,]<-x2
pairs(x)



#Here we perform k=means clustering for a sequence of model
#sizes
x.km2<-kmeans(x,2)
x.km3<-kmeans(x,3)
x.km4<-kmeans(x,4)
x.km5<-kmeans(x,5)
x.km6<-kmeans(x,6)
x.km7<-kmeans(x,7)
plot(x[,1],x[,2],type="n")
text(x[,1],x[,2],labels=as.character(x.km2$cluster))
plot(x[,1],x[,2],type="n")
text(x[,1],x[,2],labels=as.character(x.km3$cluster))
plot(x[,1],x[,2],type="n")
text(x[,1],x[,2],labels=as.character(x.km4$cluster))

#DETERMINE CLUSTER NUMBER
(sum(x.km3$withinss)/sum(x.km4$withinss)-1)*(200-3-1)

 (sum(x.km4$withinss)/sum(x.km5$withinss)-1)*(200-4-1)

 (sum(x.km5$withinss)/sum(x.km6$withinss)-1)*(200-5-1)

(sum(x.km6$withinss)/sum(x.km7$withinss)-1)*(200-6-1)



