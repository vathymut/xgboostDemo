#### Clear workspace ####
rm(list = ls(all = TRUE))

#### Load packages ####
suppressPackageStartupMessages({
  packages <- c(
    "ggplot2", "DataExplorer", "pROC", "xgboost", "data.table",
    "broom", "recipes", "rsample", "rBayesianOptimization", "WVPlots" )
  isLoaded <- sapply(packages, library, character.only = T, logical.return = T, quietly = T )
  toInstall <- packages[!isLoaded]
  if (length(toInstall) > 0L) {
    install.packages(toInstall)
    sapply(toInstall, library, character.only = T, logical.return = T, quietly = T)
  }
  rm(packages, isLoaded, toInstall)
})

#### Set defaults ####
theme_set(theme_bw())
# Set seed (for reproducibility)
set.seed(12345)

#### Load (download) dataset ####
urlToData <- "https://assets.datacamp.com/production/course_1025/datasets/loan_data_ch1.rds"
savePath <- tempfile(fileext = ".rds")
download.file(urlToData, destfile = savePath, mode = "wb")
loanData <- readRDS(savePath)
# Clean up
rm(urlToData, savePath)

#### Convert default status to factor ####
setDT(loanData)
loanData[, loan_status := factor(loan_status)]

#### Explore dataset ####
# See missing values
plot_missing(loanData)
plot_bar(loanData)
plot_boxplot(loanData, "loan_status")

#### Split into training/test set ####
# 67% training; 33% test set (stratified to keep/reflect class imbalance)
splitData <- initial_split(loanData, prop = 2/3, strata = "loan_status")
trainData <- training(splitData)
# Clean up
rm(loanData)

#### Create design matrix/clean features ####
# Add steps to create design matrix
xgbRecipe <- recipe(loan_status ~ ., data = trainData) %>%
  # Encode variables as outcomes and predictors
  # Impute missing values with bagged trees
  step_bagimpute(emp_length, int_rate) %>%
  # Log transform income because of skewness
  step_log(annual_inc) %>%
  # Convert all categorical/nominal predictors into binary/indicator numeric
  # This converts: grade and home_ownership
  step_dummy(all_nominal())
# Inspect feature engineering step
xgbRecipe

#### Create (repeated) k-fold cross-validation ####
# Also called v-fold cross-validation; repeat k-fold cv 10 times; still stratified
trainingFolds <- vfold_cv( trainData, v = 10, repeats = 10, strata = "loan_status" )
setDT(trainingFolds)
# Quick peek
head(trainingFolds)

#### Plot training folds ####
# foldData <- tidy( trainingFolds )
# setDT( foldData )
# foldData[ , ID := .GRP, by = .( Repeat, Fold ) ]
# # Create plot
# plotFolds <- ggplot( foldData, aes(x = ID, y = Row, fill = Data) )
# plotFolds <- plotFolds + geom_tile() + scale_fill_brewer()
# # See plot of observations across cross-validation samples
# plotFolds
# # Clean up
# rm( foldData )

#### Obtain list of DMatrix objects for training ####
# Assign frequency/event probability to defaults
assignFrequency <- function(defaults) {
  # Get frequency table
  freqTable <- as.data.frame(prop.table(table(defaults)))
  mapClassToFreq <- freqTable[['Freq']]
  names(mapClassToFreq) <- freqTable[[1]] # Original levels
  return(unname(mapClassToFreq[defaults]))
}
# Format data to use xgboost library
extractDMatrix <- function(splitObj, recObj = xgbRecipe, colPredict = "loan_status") {
    # Use 90% for training the model
    fitData <- analysis(splitObj)
    prepObj <- prep(recObj, training = fitData, retain = TRUE, verbose = FALSE)
    # Create inverse frequency weights to correct imbalances
    frequencies <- assignFrequency(fitData[[colPredict]])
    weights <- 1 / frequencies
    # Create xgboost DMatrix for training model
    xgbTrainData <- xgb.DMatrix(
      data = juice(prepObj, -contains(colPredict), composition = "dgCMatrix"),
      label = juice(prepObj, contains(colPredict), composition = "dgCMatrix")
    )
    # Use the remaining 10% for validation
    holdoutData <- assessment(splitObj)
    # Create xgboost DMatrix for validating model
    xgbValidateData <- xgb.DMatrix(
      data = bake( prepObj, newdata = holdoutData, -contains(colPredict), composition = "dgCMatrix" ),
      label = bake( prepObj, newdata = holdoutData, contains(colPredict), composition = "dgCMatrix" )
    )
    return(list(train = xgbTrainData, validate = xgbValidateData, weight = weights))
}

system.time({
  listDMatrix <- lapply(trainingFolds[['splits']], function(x) extractDMatrix(x))
})
# Clean up
rm(trainingFolds)

#### Initiate hyper-parameters for xgboost ####
# Fixed parameters
fixedParams <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  nthread = parallel::detectCores() - 1L,
  eval_metric = "auc"
)
# Tuning parameters (set at their default)
# lambda, L2 regularization term on weights
# alpha, L1 regularization term on weights
tuneDefaultParams <- list(
  eta = 0.3,
  max_depth = 6L,
  subsample = 1,
  lambda = 1.0,
  alpha = 0.0
)
# Set initial parameters
initParams <- c(fixedParams, tuneDefaultParams)
# Define the bounds of the search for Bayesian optimization
boundsParams <- list(
  eta = c(0.1, 0.8),
  max_depth = c(3L, 6L),
  subsample = c(0.5, 1),
  lambda = c(1.0, 5.0),
  alpha = c(0.0, 5.0)
)

#### Optimize hyper-parameters with Bayesian Optimization ####
# Calculate AUC for a single train-validate pair using CV
getAUCFromCV <- function( listData, useInverseFrequencyWeight = TRUE, params = initParams, verbose = 0) {
  # Train model
  watchlist <- list(train = listData$train, test = listData$validate)
  weight <- NULL
  if (useInverseFrequencyWeight) weight <- listData$weight
  xbgModel <- xgb.train(
    params = params,
    data = listData$train,
    watchlist = watchlist,
    verbose = verbose,
    callbacks = list(xgboost::cb.evaluation.log()),
    nrounds = 50,
    early_stopping_rounds = 5,
    weight = weight
  )
  # Extract AUC
  testAUC <- tail(xbgModel$evaluation_log[["test_auc"]], 1)
  return(testAUC)
}
# Calculate mean AUC for all train-validate pairs
getMeanAUC <- function( listDMatrix, useInverseFrequencyWeight = TRUE, params = initParams) {
  scores <- sapply(
    listDMatrix,
    getAUCFromCV,
    useInverseFrequencyWeight = useInverseFrequencyWeight,
    verbose = 0,
    params = params
  )
  meanAUC <- mean(scores)
  return(meanAUC)
}

# Create wrapper function for Bayesian Optimization
maximizeAUC <- function( eta, max_depth, subsample, lambda, alpha ) {
    # Create list of updated parameters
    replaceParams <- list( eta = eta, max_depth = max_depth, subsample = subsample, lambda = lambda, alpha =  alpha )
    updatedParams <- modifyList(initParams, replaceParams)
    # Calculate AUC with updated parameters
    scoreAUC <- getMeanAUC( listDMatrix, params = updatedParams, useInverseFrequencyWeight = TRUE)
    resultList <- list(Score = scoreAUC, Pred = 0)
    return(resultList)
}

# Run the bayesian optimization
bayesSearch <- BayesianOptimization(
  maximizeAUC,
  bounds = boundsParams,
  init_grid_dt = as.data.table(boundsParams),
  init_points = 10,
  n_iter = 30,
  acq = "ucb",
  kappa = 2.576,
  eps = 0.0,
  verbose = TRUE
)
# Get optimized parameters
tunedBayesianParams <- modifyList(fixedParams, as.list(bayesSearch$Best_Par))
# Peek at history of bayesian search
tail(bayesSearch$History)

#### Train final xgboost model ####
# Get data
listData <- extractDMatrix(splitData)
# Get iteration (boosting round) with best AUC using 10-fold cross-validation
# tunedBayesianParams
nRoundsBest <- xgb.cv(
  params = tunedBayesianParams,
  data = listData$train,
  nfold = 10,
  verbose = 0,
  callbacks = list(xgboost::cb.evaluation.log()),
  nrounds = 100,
  early_stopping_rounds = 5,
  weight = listData$weight
)$best_iteration
# Train model
xgbModel <- xgb.train(
  params = tunedBayesianParams,
  data = listData$train,
  verbose = 0,
  nrounds = nRoundsBest,
  weight = listData$weight
)

#### Compare model to benchmark ####
# Get xgboost predictions on testing set
xgbPredicted <- predict(xgbModel, listData$validate)
actualDefaults <- getinfo(listData$validate, "label")
# Create ROC table and plot
xgbROC <- pROC::roc(
  response = actualDefaults,
  predictor = xgbPredicted )
WVPlots::ROCPlot(
  data.table( predict = xgbPredicted, default = actualDefaults ),
  xvar = "predict",
  truthVar = "default",
  truthTarget = 1,
  title = "ROC Plot for xgboost Model" )

# Get strategy curve (trade-off between TNR and TPR)
# TNR (sensitivities) and TPR (specificities)
xgbStrategyCurve <- data.table(
  thresholds = xgbROC$thresholds,
  TPR = xgbROC$sensitivities,
  TNR = xgbROC$specificities
)
# Find relevant range of the strategy curve
buffer <- round( .1/.9, 2)
relevantStrategyCurve <- xgbStrategyCurve[ between(TNR, 1 - buffer, 1) ]
relevantStrategyCurve[ seq(1, .N, by = 5) ]
