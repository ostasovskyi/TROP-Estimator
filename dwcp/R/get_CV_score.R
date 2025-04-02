get_CV_score <- function(Y_obs, O, lambd, n_folds = 4, verbose = FALSE) {

  # Initialize K-fold cross-validation
  folds <- createFolds(1:nrow(O), k = n_folds, list = TRUE)

  mse <- 0
  for (i in seq_along(folds)) {
    Otr_idx <- folds[[i]]
    Otst_idx <- setdiff(1:nrow(O), Otr_idx)

    Otr <- O[Otr_idx, , drop = FALSE]
    Otst <- O[Otst_idx, , drop = FALSE]

    if (verbose) cat(".")

    L <- run_MCNNM(Y_obs, Otr, lambd, threshold = 1e-10, print_every = NULL, max_iters = 20000)

    # Calculate mean squared error for the test set
    mse <- mse + sum((Y_obs[Otst] - L[Otst]) ^ 2)
  }

  # Return the average MSE over all folds
  return(mse / n_folds)
}
