calculate_weights <- function(lambda_time, lambda_unit, Y, W, N_treat, T, T_treat) {



  weights <- matrix(0, nrow = nrow(Y), ncol = ncol(Y))
  for (j in 1:nrow(Y)) {
    for (s in 1:ncol(Y)) {
      if (W[j, s] == 0) {
        time_dist <- dist_time(s, T, T_treat)
        unit_dist <- dist_unit(j, Y, W, N_treat, T, T_treat)
        weights[j, s] <- exp(-lambda_time * time_dist - lambda_unit * unit_dist)
      }
    }
  }
  weights
}
