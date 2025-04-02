shrink_lambda <- function(A, lambd) {
  # Perform Singular Value Decomposition
  svd_result <- svd(A)
  S <- svd_result$u
  Σ <- svd_result$d
  R <- t(svd_result$v)

  # Shrink the singular values by lambda
  Σ <- Σ - lambd
  Σ[Σ < 0] <- 0

  # Reconstruct the matrix with the modified singular values
  S %*% diag(Σ) %*% R
}
