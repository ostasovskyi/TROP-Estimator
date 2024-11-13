# Doubly Weighted Causal Panel (DWCP) Estimators

This package contains the replication files and implementation of the DWCP estimator.

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

We simulate data for a four alternative problem. After identification for level and scale shifts, we have true parameters

$$
\begin{align}
  \beta &= 2
  \\
  \Sigma^{-1} &= 
    \begin{pmatrix}
      1 & 0.8 & 0.8 
      \\
      0.8 & 1 & 0.8 
      \\
      0.8 & 0.8 & 1
    \end{pmatrix}
\end{align}
$$

We generate the choices, latent utilities and covariates with

$$
\begin{align}
  X &\sim N(0, 1)
  \\
  Z &\sim MVN(X\beta, \Sigma^{-1})
  \\
  Y &= 
    \begin{cases}
      1 & Z < 0
      \\
      \arg \max(Z). & \text{else}
    \end{cases}
\end{align}
$$

``` r
library(fsprobit)
library(mvtnorm)
library(doParallel)

p = 1
n_obs = 2000
n_choices = 4

tol = .001
conv_metric = "precision"
relerr_tol = .1
max_iter = 500

# true parameters ----
Prec_iden = .2 * diag(n_choices-1) + .8 * rep(1, n_choices-1) %*% t(rep(1, n_choices-1))
Sigma_iden = solve(Prec_iden)

coef_true = as.matrix(c(2))

# initial parameters
Sigma_init = diag(n_choices-1)
coef_init = as.matrix(c(.2))

# covariate mean and sd
n_mean = 0
n_sd = 1

simdata = generate_custom_identified_choice_data(
  n_obs = n_obs,
  sampler = function(n) rnorm(n, mean=n_mean, sd=n_sd),
  coef_true = coef_true,
  Sigma_iden = solve(Prec_iden),
  seed = 1
)

# fit model ----
registerDoParallel(4)
probit_trace_iden = mnp_probit(
  X = simdata$X, Y = simdata$Y,
  beta_init = coef_init,
  Sigma_init = Sigma_init,
  E_method = "EP",
  E_sample_rate = 1,
  M_method = "Newton",
  n_choices = n_choices,
  true_trace = sum(diag(Prec_iden)),
  tol = tol,
  newton_tol = 1e-3,
  max_newton_iter = 50,
  max_iter = max_iter,
  conv_metric = conv_metric,
  shift_iden_method = "ref",
  scale_iden_method = "trace",
  verbose=5
)

# check result ---
probit_trace_iden$Precision
```

## Authors

Contributors names and contact info

## Version History

-   0.1
    -   Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
