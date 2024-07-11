
library(tidyverse)
library(dplyr)
# LOCO is drawn on iml_methods_limitations/code/03_6_fi_correlated.R
# "mlr" package description https://cran.r-project.org/web/packages/mlr/mlr.pdf


LOCO_cv <- function(data,learner,target,alpha = 0.05){
  task = makeRegrTask(data = data , target = target)
  # Perform 5-fold CV
  rin = makeResampleInstance("CV", iters = 5, task=task)
  learnerLOCO = makeLearner(learner)
  feat = getTaskFeatureNames(task)
  res = resample(learner = learnerLOCO, task = task, 
                 resampling = rin ,show.info = FALSE)
  resultLOCO = data.frame(matrix(nrow = 1, ncol = length(feat))) # create empty dataframe to store feature importance score
  resinstanceLOCO = data.frame(matrix(nrow = 5, ncol=length(feat)))
  observed_diff = numeric(length(feat))
  se_diff = numeric(length(feat))
  
  for(i in 1:length(feat)){
    taskfeat = dropFeatures(task, feat[i])
    resfeat = resample(learner = learnerLOCO, task = taskfeat, resampling = rin ,show.info = FALSE);
    # "aggr" in regression task, by default is mean of mse, you can define specific performance measure
    importance = as.numeric(resfeat$aggr-res$aggr)
    # Here, we use difference of prediction error to measure feature importance
    #importance = mean(resfeat$pred$data$response - resfeat$pred$data$truth - 
                        #(res$pred$data$response - res$pred$data$truth))
    observed_diff[i] = importance
    
    # Calculate the standard deviation of the differences
    se_diff[i] = (sd(sapply(1:5, function(k) {
      mean((resfeat$pred$data[resfeat$pred$data$iter == k, "response"] - 
            resfeat$pred$data[resfeat$pred$data$iter == k, "truth"]))^2 - 
        mean((res$pred$data[res$pred$data$iter == k, "response"] - 
              res$pred$data[res$pred$data$iter == k, "truth"])^2)
    })))/sqrt(5)
    feature = c(getTaskFeatureNames(task))
    resultLOCO[i] = importance
    resinstanceLOCO[,i] = data.frame(resfeat$measures.test[,2]-res$measures.test[,2])# under each fold CV, the difference in mse
    #resinstanceLOCO[,i] = sapply(1:5, function(k) {
      #mean(resfeat$pred$data[resfeat$pred$data$iter == k, "response"] - 
                # resfeat$pred$data[resfeat$pred$data$iter == k, "truth"] - 
             #(res$pred$data[res$pred$data$iter == k, "response"] - 
                   #res$pred$data[res$pred$data$iter == k, "truth"]))
   # }) # compute difference in prediction error under each fold
  } 
  rank_l_s = rank(-resultLOCO) # the largest score is rank 1, rank from the largest to smallest
  rownames(resultLOCO) = "Feature Importance Score"
  lb = data.frame(apply(resinstanceLOCO, 2, quantile, probs = 0.05))
  ub = data.frame(apply(resinstanceLOCO, 2, quantile, probs = 0.95))
  test_stat = observed_diff / se_diff
  p_val = 2 * pnorm(-abs(test_stat))
  FIP = data.frame(Feature = feat,
                   Feature_Importance_Score = t(resultLOCO),
                   Test_Statistics = test_stat,
                   P.Value = p_val,
                   Rank = rank_l_s,
                   LB = lb,
                   UB = ub)
  colnames(FIP) = c("Features", "Feature_Importance_Score", "Test_Statistics"
                     ,"P.Value", "Rank", "LB", "UB")
  rownames(FIP) = c(rep(1:length(feat)))
  return(FIP)
}



LOCO_split <- function(data, learner, target, alpha = 0.05) {
  task = makeRegrTask(data = data, target = target)
  # Perform in-sample split
  rin = makeResampleInstance("Subsample", split = 1/2, iters = 1, task = task)
  learnerLOCO = makeLearner(learner)
  feat = getTaskFeatureNames(task)
  res = resample(learner = learnerLOCO, task = task, resampling = rin, show.info = FALSE)
  
  resultLOCO = numeric(length(feat))
  observed_diff = numeric(length(feat))
  se_diff = numeric(length(feat))
  
  for (i in 1:length(feat)) {
    taskfeat = dropFeatures(task, features = feat[i])
    resfeat = resample(learner = learnerLOCO, task = taskfeat, resampling = rin, show.info = FALSE)
    importance = as.numeric(resfeat$aggr - res$aggr)
    observed_diff[i] = importance
    
    allfeat_error = (res$pred$data$response - res$pred$data$truth)^2
    feat_error = (resfeat$pred$data$response - resfeat$pred$data$truth)^2
    error_diff = feat_error - allfeat_error
    se_diff[i] = sd(error_diff) / sqrt(length(error_diff))
    
    resultLOCO[i] = importance
  }
  
  rank_l_s = rank(-resultLOCO)
  lb = observed_diff - qnorm(1 - alpha / 2) * se_diff
  ub = observed_diff + qnorm(1 - alpha / 2) * se_diff
  test_stat = observed_diff / se_diff
  p_val = 2 * pnorm(-abs(test_stat))
  
  FIP = data.frame(
    Features = feat,
    Feature_Importance_Score = resultLOCO,
    Test_Statistics = test_stat,
    P.Value = p_val,
    Rank = rank_l_s,
    LB = lb,
    UB = ub
  )
  
  return(FIP)
}



LOCO_all <- function(data, learner, target, alpha = 0.05) {
  task = makeRegrTask(data = data, target = target)
  learnerLOCO = makeLearner(learner)
  feat = getTaskFeatureNames(task)
  size = nrow(data)
  
  # Train and predict using the entire dataset using mse
  model = train(learnerLOCO, task)
  pred = predict(model, task)
  allfeat_error = mean((pred$data$response - pred$data$truth)^2)
  
  resultLOCO = numeric(length(feat))
  observed_diff = numeric(length(feat))
  se_diff = numeric(length(feat))
  
  for (i in 1:length(feat)) {
    taskfeat = dropFeatures(task, feat[i])
    model_feat = train(learnerLOCO, taskfeat)
    pred_feat = predict(model_feat, taskfeat)
    feat_error =  mean((pred_feat$data$response - pred_feat$data$truth)^2) 
    
    importance = feat_error - allfeat_error
    observed_diff[i] = importance
    se_diff[i] =(sd((pred_feat$data$response - pred_feat$data$truth)^2 - 
                       (pred$data$response - pred$data$truth)^2))/(sqrt(size))
    
    resultLOCO[i] = importance
  }
  
  rank_l_s = rank(-resultLOCO) # the largest score is rank 1, rank from the largest to smallest
  lb = observed_diff - qnorm(1 - alpha / 2) * se_diff
  ub = observed_diff + qnorm(1 - alpha / 2) * se_diff
  test_stat = observed_diff / se_diff
  p_val = 2 * pnorm(-abs(test_stat))
  
  FIP = data.frame(Feature = feat,
                   Feature_Importance_Score = resultLOCO,
                   Test_Statistics = test_stat,
                   P.Value = p_val,
                   Rank = rank_l_s,
                   LB = lb,
                   UB = ub)
  
  colnames(FIP) = c("Features", "Feature_Importance_Score", "Test_Statistics",
                "P.Value", "Rank", "LB", "UB")
  
  return(FIP)
}





GCM_filter <- function(data,learner,target, alpha = 0.05){
  nn<-nrow(data)
  data_X <- data %>% dplyr::select(-target)
  task_Y = makeRegrTask(data = data , target = target)# this is for regression not classifiction
  learner_filter = makeLearner(learner)
  feat = getTaskFeatureNames(task_Y)
  resultGCM = data.frame(test.statistics = numeric(0), p.val = numeric(0), rejection = logical(0), R=numeric(0)) # create empty dataframe to store feature importance score
  
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
    p.value <- as.numeric(2 * pnorm(abs(test.stat), lower.tail = FALSE))
    new_row <- data.frame(Features = feat[i],test.statistics = test.stat, p.val = p.value, rejection = p.value < alpha, R=meanR)
    resultGCM <- rbind(resultGCM, new_row)
  }
  rownames(resultGCM) = c(feat)
  selected_data <- data[, resultGCM$rejection]
  return(list(resultGCM = resultGCM, selected_data = selected_data))
}

# > task[["type"]]
#[1] "regr"

cplx_cov_matrix <- function(n, rho) {
  cov_matrix <- matrix(0, n, n)
  
  #x_{i,j} = rho^{|i-j|}
  for (i in 1:n) {
    for (j in 1:n) {
      cov_matrix[i, j] <- rho^abs(i - j)
    }
  }
  return(cov_matrix)
}


aggregate_results <- function(results_list) {
        combined_results <- do.call(rbind, results_list)
        combined_results <- aggregate(. ~ Features, data = combined_results, 
        FUN = function(x) c(mean = mean(x)))
  return(combined_results)
}


