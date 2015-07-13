
#######################################################################################################
# Insurance Hazard Prediction- Using Random Forests
# We have training set of 50999 observations with 33 independent variables which are unnamed and are
# Categorical and integeral. Our dependent variable is Hazard which is present in training set.
# Our goal is to predict the value of Hazard variable in the test set. More the hazard rating, higher the risk
# of insuring the property
# 
#The data is provided by Liberty Mutual Insurance, for the Kaggle Competition
# 
# The metric for evaualtion is the normalized Gini Coefficient
# 
# @author(Sethi,Anurag)
#  

# loading  packages

library(caret)
library(randomForest)
library(readr)
require(devtools)

devtools::install_github('dmlc/xgboost',subdir='R-package')
require(xgboost)

## randomForest and xgboost for implementing algorithms


#Set Working Directory
setwd("C:/Users/aset15/Desktop/RPractice/LibertyMutualGroupHazard")


# load raw data
train = read_csv('train.csv')
test = read_csv('test.csv')

#Exploring Data
str(train)
table(train$Hazard)
str(test)


#Visualizing the distribution of Hazard Rating with a Histogram
HazardFreqPlot<- ggplot(data=train, aes(train$Hazard)) + 
  geom_histogram(breaks=seq(0, 70, by = 1), 
                 col="red", 
                 fill="green", 
                 alpha = .2) + 
  labs(title="Count of Hazard Ratings") +
  labs(x="Hazard Rating Score", y="Frequency")

ggsave(file="HazardFreqPlot.png")

############################


###############Converting features into factors

extractFeatures <- function(data) {
  character_cols <- names(Filter(function(x) x=="character", sapply(data, class)))
  for (col in character_cols) {
    data[,col] <- as.factor(data[,col])
  }
  return(data)
}

#Calling the above defined function on 
trainFea <- extractFeatures(train)
testFea  <- extractFeatures(test)

cat("Training model\n")
rf <- randomForest(trainFea[,3:34], trainFea$Hazard, ntree=100, imp=TRUE, sampsize=10000, do.trace=TRUE)

####Extracting Important Features from the trainig set in the random forest model and plotting the graph
##using ggplot

ImpVarPlotModel <- function(data) {
imp <- importance(data, type=1)
featureImportance <- data.frame(Feature=row.names(imp), Importance=imp[,1])

p <- ggplot(featureImportance, aes(x=reorder(Feature, Importance), y=Importance)) +
  geom_bar(stat="identity", fill="#53cfff") +
  coord_flip() + 
  theme_light(base_size=20) +
  xlab("Importance") +
  ylab("") + 
  ggtitle("Random Forest Feature Importance\n") +
  theme(plot.title=element_text(size=18))

ggsave("2_feature_importance.png", p, height=12, width=8, units="in")
print(p)
}

######################## Calling the defined function on the RF model
ImpVarPlotModel(rf)

###########The plot we obtain is the order of Gini gain
# when using the corresponding variable as the splitting parameter, since we specify type=2
#when applying the function, which is the order of mean decrease in node impurity


####Our Next Step is to rebuild the model for random forests using the relevant variables from the plot earlier.
## As evident T2_V10, T2_V7 increase the node impurity, and variables T1_v13,T1_v9,T2_v3,T2_v8,T1_v6,T1_v10
#contribute very little in improving the model

###Removing the variables from both trainigm and test set
trainFeaNew<-trainFea[,-c(11,12,29,26,22,27,15)]
testFeaNew<-testFea[,-c(10,11,28,25,21,26,14)]


#Imporved RF model
rf_new <- randomForest(trainFeaNew[,3:26], trainFeaNew$Hazard, ntree=100, imp=TRUE, sampsize=10000, do.trace=TRUE)

cat("Making predictions\n")

rfnew_pred<-predict(rf_new, extractFeatures(testFeaNew[,2:25]))


submission <- data.frame(Id=test$Id)
submission$Hazard <- predict(rf_new, extractFeatures(testFeaNew[,2:25]))
write_csv(submission, "1_random_forest_impFeature.csv")

##This feature selection gives an improved normalized gini coefiicient of 0.353249,
##the benchmark Random Forest gives a gini coefiicient of 0.303405

# Set necessary parameters and use parallel threads
param <- list("objective" = "reg:linear", "nthread" = 8, "verbose"=0)

y<-trainFeaNew$Hazard

# Create the predictor data set and encode categorical variables using caret library.
mtrain = trainFeaNew[,-c(1,2)]
mtest = testFeaNew[,-c(1)]
dummies <- dummyVars(~ ., data = mtrain)
mtrain = predict(dummies, newdata = mtrain)
mtest = predict(dummies, newdata = mtest)


# Fit the model
xgb.fit = xgboost(param=param, data = mtrain, label = y, nrounds=1900, eta = .01, max_depth = 9, 
                  min_child_weight = 3, scale_pos_weight = 1.0, subsample=0.8) 
predict_xgboost <- predict(xgb.fit, mtest)

# Predict Hazard for the test set
submission.xgboost <- data.frame(Id=test$Id)
submission.xgboost$Hazard <- (predict_xgboost)
write_csv(submission, "submission_xgboost.csv")

####The submision score now obtained is 0.348557
# which is not an improvement over our model using selective random forests

# Create the predictor data set and encode categorical variables using caret library.


#We use the emsembling approach
#And take the mean of the predicted values in both the models

submission.ensemble <- data.frame(Id=test$Id)
submission.ensemble$Hazard <- (rfnew_pred+predict_xgboost)/2
write_csv(submission, "submission_ensemble.csv")

############Our final submission gives the normalized gini score as 0.377861 which is our best score yet.
