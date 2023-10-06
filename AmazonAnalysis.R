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

ggplot(data=az_train) +
  geom_bar(aes(x = ACTION))

ggplot(data = az_train) +
  geom_mosaic(aes(x = product(ACTION), fill = ROLE_FAMILY))



az_recipe <- recipe(ACTION~., data=az_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_other(all_nominal_predictors(), threshold = .01)# combines categorical values that occur <1% into an "other" value


# NOTE: some of these step functions are not appropriate to use together

# apply the recipe to your data
prep <- prep(az_recipe)
baked <- bake(prep, new_data = NULL)
head(baked)


#112
