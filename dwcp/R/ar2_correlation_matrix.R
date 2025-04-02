ar2_correlation_matrix <- function(ar_coef, T_) {

  result <- numeric(T_)
  result[1] <- 1
  result[2] <- ar_coef[1] / (1 - ar_coef[2])

  for (t in 3:T_) {
    result[t] <- ar_coef[1] * result[t - 1] + ar_coef[2] * result[t - 2]
  }

  index_matrix <- abs(outer(1:T_, 1:T_, "-"))
  cor_matrix <- matrix(result[index_matrix + 1], nrow = T_, ncol = T_)

  return(cor_matrix)
}
