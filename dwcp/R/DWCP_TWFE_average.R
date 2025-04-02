DWCP_TWFE_average <- function(Y, W, treated_units, lambda_unit, lambda_time, lambda_nn, treated_periods = 10) {
  N <- nrow(Y)
  T_ <- ncol(Y)

  # dist_time
  dist_time <- abs(1:T_ - (T_ - treated_periods / 2))

  # dist_unit
  average_treated <- rowMeans(Y[treated_units, , drop = FALSE])
  mask <- matrix(1, N, T_)
  mask[, (T_ - treated_periods + 1):T_] <- 0

  #---
  #cat("Length of average_treated:", length(average_treated), "\n")
  #cat("N:", N, "T_:", T_, "\n")
  #cat("N * T_:", N * T_, "\n")
  #print(identical(length(average_treated), N * T_))  # sanity check

  #str(average_treated)
  #print(is.numeric(average_treated))
  #print(class(average_treated))
  #---

  A <- rowSums((Y - matrix(average_treated, nrow = N, ncol = T_, byrow = TRUE))^2 * mask)
  B <- rowSums(mask)
  dist_unit <- sqrt(A / B)

  # distance-based weights
  delta_unit <- exp(-lambda_unit * dist_unit)
  delta_time <- exp(-lambda_time * dist_time)
  delta <- outer(delta_unit, delta_time)

  # CVXR variables
  unit_effects <- Variable(1, N)
  time_effects <- Variable(1, T_)
  mu <- Variable()
  tau <- Variable()
  L <- Variable(N, T_)

  unit_factor <- t(kronecker(matrix(1, T_, 1), unit_effects))
  time_factor <- kronecker(matrix(1, N, 1), time_effects)

  # Objective function
  if (lambda_nn == Inf) {
    objective <- sum_squares((Y - mu - unit_factor - time_factor - L - W * tau) * delta)
  } else {
    objective <- sum_squares((Y - mu - unit_factor - time_factor - L - W * tau) * delta) + lambda_nn * cvxr_norm(L, "nuc")
  }

  # Constraints
  constraints <- list()

  # Problem setup and solve
  prob <- Problem(Minimize(objective), constraints)
  result <- tryCatch({
    solve(prob)
  }, error = function(e) {
    print("Solver error:")
    print(e)
    return(NULL)
  })

  if (!is.null(result)) {
    #print(result$status)  # Check if "infeasible" or "unbounded"
  } else {
    print("Solver did not return a result.")
  }
  #print(result)
  return(result$getValue(tau))
}
