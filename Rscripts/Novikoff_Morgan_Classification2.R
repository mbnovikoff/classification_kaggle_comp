
# Load in Packages and Data; Set seed
library(tidyverse)
library(ggplot2)
library(naniar)
library(tidymodels)


train <- read_csv("~/Desktop/Stat_301-3/classification_kaggle_comp/data/train.csv") %>%
  janitor::clean_names() %>%
  mutate(hi_int_prncp_pd = factor(hi_int_prncp_pd))

test <- read_csv("~/Desktop/Stat_301-3/classification_kaggle_comp/data/test.csv") %>%
  janitor::clean_names()


set.seed(123)


# Data checks ----

# Missingness
# There is no missing data
vis_miss(train)
miss_var_table(train)


# Visualizing Outcome Distribution
# Most of the repayments were made towards principal, not interest 
train %>% 
  ggplot(aes(hi_int_prncp_pd)) +
  geom_histogram(stat = "count") 


# By state
train %>%
  group_by(addr_state) %>%
  ggplot(aes(hi_int_prncp_pd)) +
  geom_histogram(stat = "count") +
  facet_wrap(~ addr_state)

# Looking at important numerical vairables 
plot_corr <- train %>% 
  mutate(hi_int_prncp_pd = as.numeric(hi_int_prncp_pd)) %>% 
  select(c(hi_int_prncp_pd, acc_now_delinq, acc_open_past_24mths, 
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


# Looking at important categorical variables and only including the 
# ones where there is st

chisq.test(table(train$addr_state, train$hi_int_prncp_pd))
chisq.test(table(train$application_type, train$hi_int_prncp_pd))
chisq.test(table(train$emp_length, train$hi_int_prncp_pd))
chisq.test(table(train$emp_title, train$hi_int_prncp_pd))
chisq.test(table(train$grade, train$hi_int_prncp_pd))
chisq.test(table(train$home_ownership, train$hi_int_prncp_pd))
chisq.test(table(train$initial_list_status, train$hi_int_prncp_pd))
chisq.test(table(train$last_credit_pull_d, train$hi_int_prncp_pd))
chisq.test(table(train$purpose, train$hi_int_prncp_pd))
chisq.test(table(train$sub_grade, train$hi_int_prncp_pd))
chisq.test(table(train$term, train$hi_int_prncp_pd))
chisq.test(table(train$verification_status, train$hi_int_prncp_pd))





# Folds  
loan_folds <- vfold_cv(data = train, v = 5, repeats = 3, strata = hi_int_prncp_pd)

# Recipe
loan_rf_recipe <- recipe(hi_int_prncp_pd ~ int_rate + loan_amnt + out_prncp_inv 
                         + application_type + grade + initial_list_status + last_credit_pull_d +
                           purpose + sub_grade + term + verification_status, data = train) %>% 
  step_other(all_nominal(), -all_outcomes(), threshold = 0.1) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_predictors(), -all_outcomes()) %>% 
  step_zv(all_predictors(), -all_outcomes())

# Prep and Bake  
prep(loan_rf_recipe) %>% 
  bake(new_data = NULL)

# Define model
bt_model <- boost_tree(mode = "classification", 
                       mtry = tune(), 
                       min_n = tune(), 
                       learn_rate = tune()) %>%
  set_engine("xgboost")

bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(loan_rf_recipe)

# Set up tuning grid
bt_params <- parameters(bt_model) %>%
  update(mtry = mtry(range = c(2, 8)),
         learn_rate = learn_rate(range = c(0.01, 0.5),
                                 trans = scales::identity_trans()))

# Define tuning grid
bt_grid <- grid_regular(bt_params, levels = 3)

# Workflow 
bt_tuned <-  bt_workflow %>% 
  tune_grid(loan_folds, grid = bt_grid)
write_rds(bt_tuned, "bt_results.rds")

# Save objects
save(bt_tuned, bt_workflow, file = "~/Desktop/Stat_301-3/classification_kaggle_comp/model_info/bt_tuned.rds")

# Results

bt_workflow_tuned <- bt_workflow %>% 
  finalize_workflow(select_best(bt_tuned, metric = "accuracy"))

bt_results <- fit(bt_workflow_tuned, train)
save(bt_results, file = "~/Desktop/Stat_301-3/classification_kaggle_comp/model_info/bt_results.rds")


final_bt_results <- bt_results %>%
  predict(new_data = test) %>%
  bind_cols(test %>% select(id)) %>%
  mutate(Category = .pred_class,
         Id = id) %>%
  select(Id, Category)

# Save for Submission
write_csv(final_bt_results, "bt_output.csv")
