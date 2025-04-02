objective <- function(params, Y, W, weights, placebo_units, lambda_nn) {
  tau <- params[1]
  mu <- params[2]
  alpha <- params[3]
  beta <- params[4]
  L <- matrix(params[-(1:4)], nrow = nrow(Y), ncol = ncol(Y))

  T_treat <- ncol(W) - 9

  W_placebo <- W
  W_placebo[placebo_units, (ncol(W) - T_treat + 1):ncol(W)] <- 1

  residuals <- weights * ((Y - (mu + alpha + beta + L - W_placebo * tau))^2)
  penalty <- lambda_nn * nuclear(L)
  sum(residuals) + penalty
}
