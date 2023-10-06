# Amazon Analysis file
### libraries
library(tidyverse)
library(vroom)
library(embed)
library(ggmosaic)
### read in test and training data
az_train <- vroom("train.csv")
az_test <- vroom("test.csv")
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
  step_dummy(all_nominal_predictors()) # dummy variable encoding
   


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(az_recipe)
baked <- bake(prep, new_data = az_train)
head(baked) %>% View()


#112
