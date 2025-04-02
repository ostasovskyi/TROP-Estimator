dist_time <- function(s, T, T_treat) {
  abs(s - (T - T_treat / 2))
}
