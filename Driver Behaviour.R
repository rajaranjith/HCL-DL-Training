library(jsonlite)
library(XLConnect)
library(stats)
library(rpart)
library(caret)
library(mlbench)
library(pROC)
library("e1071")
library(sampling)

#library(sqldf)


#Read Excel File & Prepare Data file

Data_Full<-read.csv("C:\\Users\\Administrator\\Desktop\\HCL-ML-Boot Camp\\Data_Final1.csv")
colnames(Data_Full)
Data<-Data_Full[,c(2,3,8,10,11,19,20,21,22,23)]
#Data<-Data_Full[,c(1,2,20,22,26)]
Data$brake_pedal_status<-as.factor(Data$brake_pedal_status)
Data$parking_brake_status<-as.factor(Data$parking_brake_status)
Data$transmission_gear_position_Change<-as.factor(Data$transmission_gear_position_Change)
Data$Speed_Cross<-as.factor(Data$Speed_Cross)

## Elbow Rule for K selection of K- Means Cluster

wss <- (nrow(Data)-1)*sum(apply(Data,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(Data,centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")


km<-kmeans(Data,3,iter.max = 20, nstart = 1,
       algorithm = c("Hartigan-Wong"), trace=FALSE)

Data$CNo<-km$cluster
#write.csv(Data,"D:/Driving_Pattern/cluster.csv")
Data$CNo<-as.factor(Data$CNo)


## Examining the Clusrer properties through Classification Tree

dtre<-rpart(CNo~.,data=Data,method = "class")
plot(dtre,uniform=TRUE,margin=0.1)
text(dtre,use.n=TRUE, all=TRUE, cex=.7)

pfit<- prune(dtre, cp=   dtre$cptable[which.min(dtre$cptable[,"xerror"]),"CP"])
plot(pfit, uniform=TRUE, 
     main="Classification Tree for Driving_Pattern",margin = 0.1)
text(pfit,use.n=TRUE, all=TRUE, cex=0.8)


## Label Clusters
Data$Label<-Data$CNo
levels(Data$Label)<-c("Risk Zone","Potential Risk Zone","Safe Zone")

#sqldf("select Label, count(*) from Data group by Label")


## Feature Selection for building SVM classification
## Recursive Feature Elimination

control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(Data[,1:10], Data[,12], sizes=c(1:10), rfeControl=control)
print(results)
predictors(results)
plot(results, type=c("g", "o"))


##Dividing Train Test Data Set

svm<-data.frame(Data$torque_at_transmission,Data$accelerator_pedal_position,Data$vehicle_speed,Data$Label)
colnames(svm)<-c("torque_at_transmission","accelerator_pedal_position","vechicle_speed","Label")

S1<-strata(svm, stratanames=c("Label"), size=c(48,35,61), method="srswor", pik,description=FALSE)
Unit<-S1$ID_unit

Test<-svm[Unit,]
Train<-svm[-Unit,]


tune.out      <- tune.svm(Label~.,data=Train,type='C',kernel='radial',gamma = 2^(-2:2), cost = 2^(2:4),degree=2)
svm.model     <- tune.out$best.model
summary(svm.model)
#prediction<-svm
prediction    <- predict(svm.model,Test[,-4],decision.values = FALSE)
tab             <- table(pred=prediction,true=Test[,4])
conf            <- confusionMatrix(tab)
conf



