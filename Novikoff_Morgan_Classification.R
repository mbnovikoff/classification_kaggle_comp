
  
# Load in Packages and Data; Set seed

library(tidyverse)
library(ggplot2)
library(naniar)
library(tidymodels)

train <- read_csv("data/train.csv") %>%
  janitor::clean_names() %>%
  mutate(hi_int_prncp_pd = factor(hi_int_prncp_pd))

test <- read_csv("data/test.csv") %>%
  janitor::clean_names()

set.seed(123)

# Data checks ----

# Missingness
# There is no missing data
vis_miss(train)
miss_var_table(train)


# Visualizing Outcome Skew
# Most of the repayments were made towards principal, not interest 
  train %>% 
  ggplot(aes(hi_int_prncp_pd)) +
  geom_histogram(stat = "count") 
  
# Looking at important numerical vairables 
  plot_corr <- train %>% 
    mutate(hi_int_prncp_pd = as.numeric(hi_int_prncp_pd)) %>% 
    select(c(hi_int_prncp_pd, 
             acc_now_delinq, 
             acc_open_past_24mths, 
             annual_inc, 
             avg_cur_bal, 
             bc_util, 
             delinq_2yrs, 
             delinq_amnt, 
             dti, 
             int_rate, 
             loan_amnt, 
             mort_acc, 
             num_sats, 
             out_prncp_inv, 
             tot_coll_amt, 
             tot_cur_bal, 
             total_rec_late_fee)) %>% 
    cor(use = "pairwise.complete.obs") %>% 
    corrplot::corrplot(method = "number", type = "upper", 
                       tl.col = "black", col = "black", 
                       tl.cex = 0.5) 
  
 
  
  
loan_folds <- vfold_cv(data = train, v = 5, repeats = 3, strata = hi_int_prncp_pd)
  
  #recipe
  loan_rf_recipe <- recipe(hi_int_prncp_pd ~ int_rate + loan_amnt + out_prncp_inv 
                           + application_type + grade + sub_grade + term, data = train) %>% 
    step_other(all_nominal(), -all_outcomes(), threshold = 0.1) %>% 
    step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
    step_normalize(all_predictors(), -all_outcomes()) %>% 
    step_zv(all_predictors(), -all_outcomes())
  
  prep(loan_rf_recipe) %>% 
    bake(new_data = NULL)
  
  #define model
  rf_model <- rand_forest(mode = "classification",
                          min_n = tune(),
                          mtry = tune()) %>% 
    set_engine("ranger")
  
  rf_workflow <- workflow() %>% 
    add_model(rf_model) %>% 
    add_recipe(loan_rf_recipe)
  
  #set up tuning grid
  rf_params <- parameters(rf_model) %>% 
    update(mtry = mtry(range = c(2, 10)))
  
  # define tuning grid
  rf_grid <- grid_regular(rf_params, levels = 3)
  
  # workflow ----
  rf_tuned <-  rf_workflow %>% 
    tune_grid(loan_folds, grid = rf_grid)
  write_rds(rf_tuned, "rf_results.rds")
  
  save(rf_tuned, rf_workflow, file = "~/Desktop/Stat_301-3/classification_kaggle_comp/data/rf_tuned.rds")
  
  #results
  rf_workflow_tuned <- rf_workflow %>% 
    finalize_workflow(select_best(rf_tuned, metric = "accuracy"))
  
  rf_results <- fit(rf_workflow_tuned, train)
  
  final_rf_results <- rf_results %>%
    predict(new_data = test) %>%
    bind_cols(test %>%
                select(id)) %>%
    mutate(Category = .pred_class,
           Id = id) %>%
    select(Id, Category)
  
  final_rf_results
  
  write_csv(final_rf_results, "rf_output.csv")
  