
library(tidyverse)
library(dplyr)
# LOCO code from iml_methods_limitations/code/03_6_fi_correlated.R
# "mlr" package description https://cran.r-project.org/web/packages/mlr/mlr.pdf


LOCO <- function(data,learner,target){
  task = makeRegrTask(data = data , target = target)
  # Perform 5-fold CV
  set.seed(101)
  #rin = makeResampleInstance("CV", iters = 5, task=task)
  rin = makeResampleInstance("Subsample", iters = 5, split = 4/5, task = task)
  learnerLOCO = makeLearner(learner)
  feat = getTaskFeatureNames(task)
  res = resample(learner = learnerLOCO, task = task, 
                 resampling = rin ,show.info = FALSE)
  resultLOCO = data.frame(matrix(nrow = 1, ncol = length(feat))) # create empty dataframe to store feature importance score
  resinstanceLOCO = data.frame(matrix(nrow = 5, ncol=length(feat)))
  for(i in 1:length(feat)){
    taskfeat = dropFeatures(task, feat[i])
    resfeat = resample(learner = learnerLOCO, task = taskfeat, resampling = rin ,show.info = FALSE);
    # "aggr" in regression task, by default is mean of mse, you define specific performance measure
    importance = data.frame(abs(resfeat$aggr-res$aggr))
    feature = c(getTaskFeatureNames(task))
    resultLOCO[i] = importance
    resinstanceLOCO[,i] = data.frame(abs(resfeat$measures.test[,2]-res$measures.test[,2]))
  } # under each fold CV, the difference in mse
  rank_l_s = rank(-resultLOCO) # the largest score is rank 1, rank from the largest to smallest
  rownames(resultLOCO) = "Feature Importance Score"
  lb = data.frame(apply(resinstanceLOCO, 2, quantile, probs = 0.05))
  ub = data.frame(apply(resinstanceLOCO, 2, quantile, probs = 0.95))
  FIP = data.frame(Feature_Importance_Score = t(resultLOCO),
                   Feature = feat,
                   Rank = rank_l_s,
                   LB = lb,
                   UB = ub)
  colnames(FIP) = c("Feature_Importance_Score", "Features", "Rank", "LB", "UB")
  rownames(FIP) = c(rep(1:length(feat)))
  return(FIP)
}



GCM_filter <- function(data,learner,target, alpha = 0.05){
  nn<-nrow(data)
  data_X <- data %>% dplyr::select(-target)
  task_Y = makeRegrTask(data = data , target = target)# this is for regression not classifiction
  learner_filter = makeLearner(learner)
  feat = getTaskFeatureNames(task_Y)
  resultGCM = data.frame(test.statistics = numeric(0), p.val = numeric(0), rejection = logical(0)) # create empty dataframe to store feature importance score
  
  for(i in 1:length(feat)){
    taskfeat_Y = dropFeatures(task_Y, feat[i])
    model_Y <- train(learner_filter, taskfeat_Y)#  By default subsets= NULL if all observations are used
    pred_Y <- predict(model_Y, task = taskfeat_Y)
    res_Y <- getPredictionTruth(pred_Y) - getPredictionResponse(pred_Y)
    taskfeat_Xj <- makeRegrTask(data = data_X , target = feat[i])# this is for regression not classifiction
    model_Xj <- train(learner_filter, taskfeat_Xj)#  By default subsets= NULL if all observations are used
    pred_Xj <- predict(model_Xj, task = taskfeat_Xj)
    res_Xj <- getPredictionTruth(pred_Xj) - getPredictionResponse(pred_Xj)
    # GCM Test Statistics Computing:
    R <- res_Xj*res_Y
    R.sq <- R^2
    meanR <- mean(R)
    test.stat <- sqrt(nn) * meanR / sqrt(mean(R.sq) - meanR^2)
    p.value <- 2 * pnorm(abs(test.stat), lower.tail = FALSE)
    new_row <- data.frame(Feature = feat[i],test.statistics = test.stat, p.val = p.value, rejection = p.value < alpha)
    resultGCM <- rbind(resultGCM, new_row)
  }
  rownames(resultGCM) = c(feat)
  selected_data <- data[, resultGCM$rejection]
  return(list(resultGCM = resultGCM, selected_data = selected_data))
}

# > task[["type"]]
#[1] "regr"