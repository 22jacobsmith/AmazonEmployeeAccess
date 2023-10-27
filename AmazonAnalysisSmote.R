### using smote to re-run all models

library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)

### read in test and training data
az_train <- vroom("train.csv")

az_test <- vroom("test.csv")

## make action a factor

az_train$ACTION = factor(az_train$ACTION)


## smote recipe

library(themis)

smote_recipe <-
  recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 5)

# prep <- prep(smote_recipe)
# baked <- bake(prep, new_data = az_train)



### run each of the models

## logistic

logistic_model <- logistic_reg() %>%
  set_engine("glm")


# set up the workflow

logistic_wf <- workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(logistic_model) %>%
  fit(data = az_train)


logistic_preds <- predict(logistic_wf, new_data = az_test,
                          type = "prob")

# prepare for submission to kaggle

logistic_output <- tibble(id = az_test$id, Action = logistic_preds$.pred_1)

vroom_write(logistic_output, "logisticPreds.csv", delim = ",")


## penalized logistic


# set up model and workflow
plog_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

az_pen_wf <- 
  workflow() %>%
  add_recipe(smote_recipe) %>%
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


## random forests


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



###naive bayes


library(discrim)

nb_model <-
  naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")


nb_wf <-
  workflow() %>%
  add_recipe(smote_recipe) %>%
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


## KNN


## set up model and workflow
knn_model <-
  nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")


knn_wf <-
  workflow() %>%
  add_recipe(smote_recipe) %>%
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


### SVM



svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")


az_svm_wf <- 
  workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(svmRadial)

## set up a tuning grid
tuning_grid <-
  grid_regular(rbf_sigma(),
               cost(),
               levels = 5)

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


