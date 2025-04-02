generate_data <- function(F_, M, cov_mat, pi, noise = "norm", treated_periods = 10, treated_units = 10) {

  N <- nrow(F_)
  T_total <- ncol(F_)

  # Y = F + M + multivariate normal noise
  if (noise == "norm"){
    Y <- F_ + M + MASS::mvrnorm(n = N, mu = rep(0, T_total), Sigma = cov_mat)
  }else if (noise == "none"){
    Y <- F_ + M
  }else if (noise == "noise"){
    Y <- MASS::mvrnorm(n = N, mu = rep(0, T_total), Sigma = cov_mat)
  }


  # Initialize W matrix
  W <- matrix(0, nrow = N, ncol = T_total)

  # Generate treatment candidates
  candidates <- rbinom(N, 1, pi)

  treated_number <- sum(candidates)

  if (treated_number == 0) {
    index <- sample(1:N, 1)
  } else {
    index <- which(candidates == 1)
    if (treated_number > treated_units) {
      index <- sample(index, treated_units, replace = FALSE)
    }
  }

  # Set treatment in last treated_periods columns for treated units
  W[index, (T_total - treated_periods + 1):T_total] <- 1

  return(list(Y = Y, W = W, index = index))
}
