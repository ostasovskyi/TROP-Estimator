dist_unit <- function(j, Y, W, N_treat, T, T_treat) {

  N <- nrow(Y)

  untreated_time <- 1:(T - T_treat)
  Y_j_untreated <- mean(Y[j, untreated_time])
  Y_treated_avg <- mean(Y[(N - N_treat + 1):N, untreated_time])

  squared_diffs <- sum((Y[j, untreated_time] - Y_treated_avg)^2)
  sqrt(squared_diffs / (T - T_treat))
}
