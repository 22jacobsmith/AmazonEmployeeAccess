# Amazon Analysis file
### libraries
library(tidyverse)
library(vroom)
#install.packages("embed")
library(embed)
#library(ggmosaic)
library(tidymodels)
### read in test and training data
az_train <- vroom("train.csv")

az_test <- vroom("test.csv")

## make action a factor

az_train$ACTION = factor(az_train$ACTION)

#head(az_train)

## explore the data
# proportion of 1's and 0's
#ggplot(data=az_train) +
#  geom_bar(aes(x = ACTION))

# number of 1's for job titles with over 500 1's
#tibble(ROLE_TITLE = az_train$ROLE_TITLE,ACTION = az_train$ACTION) %>%
#  group_by(ROLE_TITLE) %>% summarize(approved = sum(ACTION)) %>% filter(approved > 500) %>%
 # ggplot() +
 # geom_col(aes(x = as.factor(ROLE_TITLE), y = approved))



az_recipe <- recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <1% into an "other" value
  step_dummy(all_nominal_predictors())
   


# apply the recipe to the data
prep <- prep(az_recipe)
baked <- bake(prep, new_data = az_train)


## 112

### LOGISTIC REGRESSION
# fit a logistic regression model

logistic_model <- logistic_reg() %>%
  set_engine("glm")


# set up the workflow

logistic_wf <- workflow() %>%
  add_recipe(az_recipe) %>%
  add_model(logistic_model) %>%
  fit(data = az_train)


logistic_preds <- predict(logistic_wf, new_data = az_test,
                          type = "prob")

# prepare for submission to kaggle

logistic_output <- tibble(id = az_test$id, Action = logistic_preds$.pred_1)

vroom_write(logistic_output, "logisticPreds.csv", delim = ",")




### PENALIZED LOGISTIC REGRESSION

# use target encoding

# set up penalized regression recipe
az_pen_recipe <- recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))



# apply the recipe to the data
prep <- prep(az_pen_recipe)

baked <- bake(prep, new_data = az_train)

# set up model and workflow
plog_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

az_pen_wf <- 
    workflow() %>%
    add_recipe(az_pen_recipe) %>%
    add_model(plog_mod)

## set up a tuning grid
tuning_grid <-
  grid_regular(penalty(),
               mixture(),
               levels = 5)

## split into folds
folds <- vfold_cv(az_train, v = 5, repeats = 1)

# run cv

CV_results <-
  az_pen_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  az_pen_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

plog_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

plog_output <- tibble(id = az_test$id, Action = plog_preds$.pred_1)

vroom_write(plog_output, "PenLogPreds.csv", delim = ",")




### batch computing

# R (open R session)
# R CMD BATCH --no-save --no-restore AmazonAnalysis.R &
# 

# use save() to save R objects
# save(file = "filename.RData", list = c("logReg_wf"))
# then, load("filename.RData")
#, or vroom write


## parallel computing

#library(doParallel)
#parallel::detectCores() # count cores (12)
#cl <- makePSOCKcluster(num_cores)
#registerDoParallel(cl)

#... ## insert code

#stopCluster(cl)




#### RANDOM FORESTS
library(doParallel)
 cl <- makePSOCKcluster(5)
 registerDoParallel(cl)

# set up rf recipe
az_rf_recipe <- recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))



# apply the recipe to the data
prep <- prep(az_rf_recipe)

baked <- bake(prep, new_data = az_train)


# set up model and workflow
rf_mod <- rand_forest(mtry = tune(), min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

az_rf_wf <- 
  workflow() %>%
  add_recipe(az_rf_recipe) %>%
  add_model(rf_mod)

## set up a tuning grid
tuning_grid <-
  grid_regular(mtry(range = c(1,9)),
               min_n(),
               levels = 9)

## split into folds
folds <- vfold_cv(az_train, v = 5, repeats = 1)

# run cv

CV_results <-
  az_rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  az_rf_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

rf_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

rf_output <- tibble(id = az_test$id, Action = rf_preds$.pred_1)

vroom_write(rf_output, "AmazonRFPreds.csv", delim = ",")

 stopCluster(cl)

 
 
 
#### naive bayes
 
library(doParallel)
cl <- makePSOCKcluster(10)
registerDoParallel(cl) 
 
 
 
 
library(discrim)
 
nb_model <-
  naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")


nb_wf <-
  workflow() %>%
  add_recipe(az_rf_recipe) %>%
  add_model(nb_model)

## set up a tuning grid
tuning_grid <-
  grid_regular(Laplace(),
               smoothness(),
               levels = 10)

## split into folds
folds <- vfold_cv(az_train, v = 5, repeats = 1)

# run cv

CV_results <-
  nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  nb_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

nb_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

nb_output <- tibble(id = az_test$id, Action = nb_preds$.pred_1)

vroom_write(nb_output, "AmazonNBPreds.csv", delim = ",")



stopCluster(cl) 


### KNN Classification

library(doParallel)
cl <- makePSOCKcluster(10)
registerDoParallel(cl) 


## set up recipe

az_knn_recipe <- recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

## set up model and workflow
knn_model <-
  nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")


knn_wf <-
  workflow() %>%
  add_recipe(az_knn_recipe) %>%
  add_model(knn_model)

## set up a tuning grid
tuning_grid <-
  grid_regular(neighbors(),
               levels = 20)

## split into folds
folds <- vfold_cv(az_train, v = 5, repeats = 1)

# run cv

CV_results <-
  knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  knn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

knn_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

knn_output <- tibble(id = az_test$id, Action = knn_preds$.pred_1)


vroom_write(knn_output, "AmazonKNNPreds.csv", delim = ",")


stopCluster(cl) 



### Principal Components Regression


library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl) 


## set up recipe

az_pcr_recipe <- recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.99)

# apply the recipe to the data
prep <- prep(az_pcr_recipe)

baked <- bake(prep, new_data = az_train)

## set up model and workflow
knn_model <-
  nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")


knn_wf <-
  workflow() %>%
  add_recipe(az_pcr_recipe) %>%
  add_model(knn_model)

## set up a tuning grid
tuning_grid <-
  grid_regular(neighbors(),
               levels = 5)

## split into folds
folds <- vfold_cv(az_train, v = 5, repeats = 1)

# run cv

CV_results <-
  knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  knn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

knn_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

knn_output <- tibble(id = az_test$id, Action = knn_preds$.pred_1)


vroom_write(knn_output, "AmazonKNNPreds.csv", delim = ",")




## pca nb model


library(discrim)


nb_model <-
  naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")


nb_wf <-
  workflow() %>%
  add_recipe(az_pcr_recipe) %>%
  add_model(nb_model)

## set up a tuning grid
tuning_grid <-
  grid_regular(Laplace(),
               smoothness(),
               levels = 5)


## split into folds
folds <- vfold_cv(az_train, v = 5, repeats = 1)

# run cv

CV_results <-
  nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  nb_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

nb_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

nb_output <- tibble(id = az_test$id, Action = nb_preds$.pred_1)

vroom_write(nb_output, "AmazonNBPreds.csv", delim = ",")

## PCA Random Forest



# set up model and workflow
rf_mod <- rand_forest(mtry = tune(), min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


az_rf_wf <- 
  workflow() %>%
  add_recipe(az_pcr_recipe) %>%
  add_model(rf_mod)

## set up a tuning grid
tuning_grid <-
  grid_regular(mtry(range = c(1,9)),
               min_n(),
               levels = 5)

## split into folds
folds <- vfold_cv(az_train, v = 5, repeats = 1)

# run cv

CV_results <-
  az_rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  az_rf_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

rf_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

rf_output <- tibble(id = az_test$id, Action = rf_preds$.pred_1)

vroom_write(rf_output, "AmazonRFPreds.csv", delim = ",")



stopCluster(cl)




#### Support vector machines


# library(doParallel)
# cl <- makePSOCKcluster(5)
# registerDoParallel(cl)

az_svm_recipe <- recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())




svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")


az_svm_wf <- 
  workflow() %>%
  add_recipe(az_svm_recipe) %>%
  add_model(svmRadial)

## set up a tuning grid
tuning_grid <-
  grid_regular(rbf_sigma(),
               cost(),
               levels = 8)

## split into folds
folds <- vfold_cv(az_train, v = 3, repeats = 1)

# run cv

CV_results <-
  az_svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  az_svm_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

svm_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

svm_output <- tibble(id = az_test$id, Action = svm_preds$.pred_1)

vroom_write(svm_output, "AmazonSVMPreds.csv", delim = ",")

# stopCluster(cl)



### make best model possible


# set up rf recipe
az_rf_recipe <- recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))



# apply the recipe to the data
prep <- prep(az_rf_recipe)

baked <- bake(prep, new_data = az_train)


# set up model and workflow
rf_mod <- rand_forest(mtry = 1, min_n = 22,
                      trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

az_rf_wf <- 
  workflow() %>%
  add_recipe(az_rf_recipe) %>%
  add_model(rf_mod)
# 
## set up a tuning grid
tuning_grid <-
  grid_regular(mtry(range = c(1,9)),
               min_n(),
               levels = 9)
# 
# ## split into folds
# folds <- vfold_cv(az_train, v = 5, repeats = 1)
# 
# # run cv
# 
# CV_results <-
#   az_rf_wf %>%
#   tune_grid(resamples = folds,
#             grid = tuning_grid,
#             metrics = metric_set(roc_auc))
# 
# # find best tuning parm values
# 
# best_tune <-
#   CV_results %>%
#   select_best("roc_auc")
# 
# # finalize wf and get preds

final_wf <-
  az_rf_wf %>%
  # finalize_workflow(best_tune) %>%
  fit(data = az_train)

rf_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

rf_output <- tibble(id = az_test$id, Action = rf_preds$.pred_1)

vroom_write(rf_output, "AmazonRFPreds.csv", delim = ",")


## try a BART classification model





my_mod <- bart(
  trees = 10
) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification") %>% 
  translate()

workflow_bart <- workflow() %>%
  add_recipe(az_rf_recipe) %>%
  add_model(my_mod) %>%
  fit(data = az_train) # fit the workflow

# 
# final_wf <-
#   workflow_bart %>%
#   # finalize_workflow(best_tune) %>%
#   fit(data = az_train)

bart_preds <-
  workflow_bart %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

bart_output <- tibble(id = az_test$id, Action = bart_preds$.pred_1)

vroom_write(bart_output, "AmazonBARTPreds.csv", delim = ",")


### try a boosting model


library(doParallel)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

# set up model and workflow
boost_mod <- boost_tree(tree_depth = tune(), learn_rate = tune(),
                        trees = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

boost_wf <- 
  workflow() %>%
  add_recipe(az_rf_recipe) %>%
  add_model(boost_mod)
# 
## set up a tuning grid
tuning_grid <-
  grid_regular(tree_depth(),
               learn_rate(),
               trees(),
               levels = 5)

## split into folds
folds <- vfold_cv(az_train, v = 4, repeats = 1)

# run cv

CV_results <-
  boost_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  boost_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

boost_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

boost_output <- tibble(id = az_test$id, Action = boost_preds$.pred_1)

vroom_write(boost_output, "AmazonBoostPreds.csv", delim = ",")




## use smote to balance the data

library(themis)

smote_recipe <-
  recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_dummy(all_nominal_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 5) %>%
  step_upsample()




# set up model and workflow
rf_mod <- rand_forest(mtry = tune(), min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


az_rf_wf <- 
  workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(rf_mod)

## set up a tuning grid
tuning_grid <-
  grid_regular(mtry(range = c(1,9)),
               min_n(),
               levels = 5)

## split into folds
folds <- vfold_cv(az_train, v = 3, repeats = 1)

# run cv

CV_results <-
  az_rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# find best tuning parm values

best_tune <-
  CV_results %>%
  select_best("roc_auc")

# finalize wf and get preds

final_wf <-
  az_rf_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = az_train)

rf_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

rf_output <- tibble(id = az_test$id, Action = rf_preds$.pred_1)

vroom_write(rf_output, "AmazonRFPreds.csv", delim = ",")

