cross_validation <- function(Y, W, lambda_grid, num_runs = 500) {
  N <- nrow(Y)
  T_ <- ncol(Y)

  N_treat <- nrow(W) - 9
  T_treat <- ncol(W) - 9

  best_lambda <- NULL
  best_std_dev <- Inf

  for (lambda_time in lambda_grid) {
    for (lambda_unit in lambda_grid) {
      for (lambda_nn in lambda_grid) {
        weights <- calculate_weights(lambda_time, lambda_unit, Y, W, N_treat, T_, T_treat)
        ate_estimates <- numeric(num_runs)

        for (run in 1:num_runs) {
          control_units <- which(rowSums(W) == 0)
          placebo_units <- sample(control_units, N_treat, replace = FALSE)

          init_params <- c(0, 0, 0, 0, rep(0, length(Y)))

          print(class(init_params))
          print(class(Y))
          print(class(W))
          print(class(weights))
          print(class(placebo_units))



          result <- optim(init_params, fn = objective, Y = Y, W = W, weights = weights,
                          placebo_units = placebo_units, lambda_nn = lambda_nn)

          if (result$convergence == 0) {
            ate_estimates[run] <- result$value
          }
        }

        std_dev <- sd(ate_estimates)
        if (std_dev < best_std_dev) {
          best_std_dev <- std_dev
          best_lambda <- c(lambda_time, lambda_unit, lambda_nn)
        }
      }
    }
  }
  list(best_lambda = best_lambda, best_std_dev = best_std_dev)
}
