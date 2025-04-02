decompose_Y <- function(Y, rank = 4) {
  N <- nrow(Y)
  T_ <- ncol(Y)

  # Perform Singular Value Decomposition
  svd_result <- svd(Y)
  u <- svd_result$u
  s <- svd_result$d
  v <- svd_result$v

  # Extract the rank components
  factor_unit <- u[, 1:rank]
  factor_time <- t(v)[1:rank, ]

  # Calculate the low-rank approximation matrix L
  L <- factor_unit %*% diag(s[1:rank]) %*% factor_time

  # Calculate the residual matrix E
  E <- Y - L

  # Calculate F and M matrices
  F_ <- outer(rowMeans(L), colMeans(L), `+`) - mean(L)
  M <- L - F_

  # Return the results
  list(F_ = F_, M = M, E = E, factor_unit_scaled = factor_unit * sqrt(N))
}
