do_CV <- function(Y_obs, O, lambdas = c(5, 10, 20, 40), n_tries = 10, verbose = FALSE) {

  score <- list()

  for (t in seq_len(n_tries)) {
    run_score <- list()
    for (l in lambdas) {
      if (verbose) cat(sprintf("lambda %d:", l))
      run_score[[as.character(l)]] <- get_CV_score(Y_obs, O, l, n_folds = 4, verbose = verbose)
      if (verbose) cat(sprintf(" : %f\n", run_score[[as.character(l)]]))
    }
    score[[as.character(t)]] <- run_score
  }

  return(score)
}
