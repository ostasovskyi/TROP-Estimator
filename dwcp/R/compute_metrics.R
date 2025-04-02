compute_metrics <- function(x, true_value = 0) {
  bias <- mean(x - true_value)
  rmse <- sqrt(mean((x - true_value)^2))
  return(c(bias = bias, rmse = rmse))
}
