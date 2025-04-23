New todo:

1. get replicate of this table.

2. scaled difference with paper table.

# Triply Robust Panel Estimators (TROP)

This package contains the replication files and implementation of the TROP estimator.

## Dependencies

Install the `devtools` package from CRAN.

``` r
install.packages('devtools')
```

This package depends on the following packages, not available in CRAN: `MCPanel` and `synthdid`.

They can be installed with the following code:

``` r
devtools::install_github("susanathey/MCPanel")
devtools::install_github("synth-inference/synthdid")
```

## Installation

Once the non-CRAN dependencies are installed, install the package with

``` r
devtools::install_github('')
```
### Example execution
Before running the compute_DWCP functions, make sure to have run something like the following in order to have the requisite parameters. Change the column from $hours to something else as you see fit. Replace 'CPS.csv' with the local path on your machine. 
``` r
df <- read.csv('CPS.csv', sep = ';')
Y_true_full <- matrix(df$hours, nrow = 40, byrow = TRUE)
Y_true_full <- t(Y_true_full)
Y_true_full <- Y_true_full / sd(Y_true_full)
Y_true_full <- Y_true_full - mean(Y_true_full)
N_total <- nrow(Y_true_full)
T_total <- ncol(Y_true_full)
W_true_full <- matrix(0, N_total, T_total)

#creating the assignment vector for minimum wage
min_wage <- matrix(df$min_wage, nrow = 40, byrow = TRUE)
min_wage <- t(min_wage)
Ds <- which(min_wage == TRUE, arr.ind = TRUE)[, 1]
assignment_vector <- numeric(N_total)
assignment_vector[Ds] <- 1

valid <- cross_validation(Y_true_full, W_true_full, 1#10^seq(-4, 2, length.out = 10)
                          , num_runs = 2)#500)  
lambda_unit <- valid$best_lambda[1]
lambda_time <- valid$best_lambda[2]
lambda_nn <- valid$best_lambda[3]
```


## Table of Functions

| Function       | Inputs                         | Output                         |
|----------------|--------------------------------|--------------------------------|
| `ar2_correlation_matrix()`   | `ar_coef`: , `T_`:    | cor_matrix |
| `calculate_weights()`  | `lambda_time`: ,   `lambda_unit`:,   `Y`:,   `W`:,   `N_treat`:,   `T`:,   `T_treat`:            |         weights           |
| `compute_and_save_metrics()`  | `variables`: , `true_value`: default value is 0, `output_csv`: default is 'metrics_results.csv' (see data folder)|  final_results_wide  |
| `compute_DWCP()`   | `data`:,`Y`:,`W`:,`lambda_unit`:,`lambda_time`:,`lambda_nn`:,`exp_num`:     | baseline row |
| `compute_metrics()`   | `x`: , `true_value`: default value is 0    | c(bias = bias, rmse = rmse) |
| `cross_validation()`   | `Y`:, `W`:, `lambda_grid`:, `num_runs`: default value is 500  | list(best_lambda = best_lambda, best_std_dev = best_std_dev) |
| `decompose_Y()`   | `Y`:, `rank`: default value is 4|  list(F_ = F_, M = M, E = E, factor_unit_scaled = factor_unit * sqrt(N)) |
| `DIDp()`   | `M`:, `mask`:  | M_pred |
| `DIFP_TWFE()`   | `Y`:, `W`:, `treated_units`:, `treated_periods`: | treatment_effect |
| `dist_time()`   | `s`:, `T`:, `T_treat`:  |  |
| `dist_unit()`   | `j`:, `Y`:, `W`:, `N_treat`:, `T`:, `T_treat`:    |  |
| `do_CV()`   | `Y_obs`:  , `O`:, `lambdas`: default value is c(5, 10, 20, 40), `n_tries`: default valya is 10, `verbose`: default value is FALSE   | score |
| `DWCP_TWFE_average()`   | `Y`:, `W`:, `treated_units`:, `lambda_unit`:, `lambda_time`:, `lambda_nn`:, `treated_periods`: default value is 10  | result$getValue(tau) |
| `fit_ar2()`   | `E`:   | ar_coef |
| `generate_data()`   | `F_`:, `M`:, `cov_mat`:, `pi`:, `noise`: default is "norm", `treated_periods`: default is 10, `treated_units`: default is 10   | list(Y = Y, W = W, index = index) |
| `get_CV_score()`   | `Y_obs`:, `O`:,lambd:, `n_folds`: default is 4, `verbose`: default is FALSE    | mse / n_folds |
| `getPO()`   | `A`:, `O`:    | A_out |
| `getPOinv()`   | `A`:, `O`:     | A_out |
| `objective()`   | `params`:, `Y`:, `W`:, `weights`:, `placebo_units`:, `lambda_nn`:    |  |
| `run_MCNNM()`   | `ar_coef`: , `T_`:    | L_new |
| `shrink_lambda()`   | `Y_obs`:, `O`:, `lambd`: default value is 10, `threshold`: default value is 0.01, `print_every`: default value is NULL, `max_iters`: default value is 20000 |  |
| `table_generation()`   | `fixed_effects`: , `interactive_data`:, `cov_mat`:, `prob`:, `noise`: default value is "norm", `treatment_periods`: , `treatment_units`, `exp_num`, `ran_seed`: default value is 0  | list(estimate_sdid, estimate_sc, estimate_did, estimate_mc,  estimate_difp,  estimate_dwcp) |



## Authors

Romy Aran (romyaran1@gmail.com)

## Version History

-   0.1
    -   Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
