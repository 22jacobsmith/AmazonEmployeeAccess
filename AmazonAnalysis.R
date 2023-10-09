# Amazon Analysis file
### libraries
library(tidyverse)
library(vroom)
library(embed)
library(ggmosaic)
library(tidymodels)
### read in test and training data
az_train <- vroom("train.csv")
az_test <- vroom("test.csv")

## make action a factor

az_train$ACTION = factor(az_train$ACTION)

#head(az_train)

## explore the data
# proportion of 1's and 0's
ggplot(data=az_train) +
  geom_bar(aes(x = ACTION))

# number of 1's for job titles with over 500 1's
tibble(ROLE_TITLE = az_train$ROLE_TITLE,ACTION = az_train$ACTION) %>%
  group_by(ROLE_TITLE) %>% summarize(approved = sum(ACTION)) %>% filter(approved > 500) %>%
  ggplot() +
  geom_col(aes(x = as.factor(ROLE_TITLE), y = approved))



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
