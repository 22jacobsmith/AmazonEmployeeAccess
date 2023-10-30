
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
#  step_normalize(all_numeric_predictors()) %>%
  step_smote(all_outcomes())



library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)
# set up model and workflow
rf_mod <- rand_forest(mtry = tune(), min_n = tune(),
                      trees = 1000) %>%
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

stopCluster(cl)


# non-smote


az_rf_recipe <- recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))



# apply the recipe to the data
prep <- prep(az_rf_recipe)

baked <- bake(prep, new_data = az_train)


# set up model and workflow
rf_mod <- rand_forest(mtry = 1, min_n = 25,
                      trees = 1000) %>%
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
  #finalize_workflow(best_tune) %>%
  fit(data = az_train)

rf_preds <-
  final_wf %>%
  predict(new_data = az_test, type = "prob")

# prepare and export preds to csv for kaggle

rf_output <- tibble(id = az_test$id, Action = rf_preds$.pred_1)

vroom_write(rf_output, "AmazonRFPreds.csv", delim = ",")

