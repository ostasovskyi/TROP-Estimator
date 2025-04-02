fit_ar2 <- function(E) {

  T_full <- ncol(E)
  E_ts <- E[, 3:T_full]
  E_lag_1 <- E[, 2:(T_full - 1)]
  E_lag_2 <- E[, 1:(T_full - 2)]

  a_1 <- sum(diag(crossprod(E_lag_1, E_lag_1)))
  a_2 <- sum(diag(crossprod(E_lag_2, E_lag_2)))
  a_3 <- sum(diag(crossprod(E_lag_1, E_lag_2)))

  matrix_factor <- matrix(c(a_1, a_3, a_3, a_2), nrow = 2, byrow = TRUE)

  b_1 <- sum(diag(crossprod(E_lag_1, E_ts)))
  b_2 <- sum(diag(crossprod(E_lag_2, E_ts)))

  ar_coef <- solve(matrix_factor) %*% c(b_1, b_2)

  return(ar_coef)
}
