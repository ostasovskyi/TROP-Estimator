compute_DWCP <- function(data, Y, W, lambda_unit, lambda_time, lambda_nn, exp_num) {
  # Required packages

  pck <- c("Matrix", "CVXR", "data.table", "ggplot2", "parallel", "caret", "MASS", "npmr", "synthdid")
  lapply(pck, require, character.only = TRUE)

  # Check and fix Y orientation
  #if (nrow(Y) > ncol(Y)) {
  #  warning("Transposing Y to match (units x time) format.")
  #  Y <- t(Y)
  #}

  # SVD decomposition
  result <- decompose_Y(Y, rank = 4)

  # Extract matrices
  F_ <- result$F_
  M <- result$M
  E <- result$E
  unit_factors <- result$factor_unit_scaled

  # Dimensions
  N_total <- nrow(Y)
  T_total <- ncol(Y)

  # === Construct AR(2) correlation matrix and scaled cov ===
  ar_coef <- fit_ar2(E)
  cor_matrix <- ar2_correlation_matrix(ar_coef, T_total)
  scaled_sd <- norm(crossprod(t(E)) / N_total, type = "F") / norm(cor_matrix, type = "F")
  cov_mat <- cor_matrix * scaled_sd
  # === Construct assignment vector from min_wage ===
  #if (length(data$min_wage) != N_total * T_total) {
  #  stop("min_wage in data has incorrect length: expected N_total × T_total.")
  #}
  #print("🔥 compute_DWCP updated 🔥")
  min_wage <- matrix(data$min_wage, nrow = N_total, byrow = TRUE)
  min_wage <- t(min_wage)

  Ds <- which(min_wage == TRUE, arr.ind = TRUE)[, 1]
  assignment_vector <- numeric(N_total)
  assignment_vector[Ds] <- 1
  # === Safety checks ===
  stopifnot(length(assignment_vector) == nrow(unit_factors))
  stopifnot(!any(is.na(assignment_vector)))
  stopifnot(!any(is.na(unit_factors)))
  stopifnot(!any(is.infinite(unit_factors)))

  # === Fit logistic regression for propensity scores ===
  model <- glm(assignment_vector ~ unit_factors - 1, family = binomial)
  p <- predict(model, newdata = as.data.frame(unit_factors), type = "response")

  # Calculate conditional variance
  cond_var <- cov_mat[-1, -1] - (cov_mat[-1, (ncol(cov_mat)-2):(ncol(cov_mat)-1)] %*%
                                   solve(cov_mat[(ncol(cov_mat)-2):(ncol(cov_mat)-1), (ncol(cov_mat)-2):(ncol(cov_mat)-1)]) %*%
                                   cov_mat[(ncol(cov_mat)-2):(ncol(cov_mat)-1), -1])

  estimate_sdid <- rep(0,exp_num)
  estimate_dwcp <- rep(0,exp_num)
  estimate_mc <- rep(0,exp_num)
  estimate_sc <- rep(0,exp_num)
  estimate_difp <- rep(0,exp_num)
  estimate_did <- rep(0,exp_num)

  # === Estimate treatment effects using various methods ===
  baseline <- table_generation(F_, M, cov_mat, p, "norm", 10, 10, exp_num, 0)

  # Extract estimates

  estimate_sdid <- baseline[[1]]
  estimate_dwcp <- baseline[[6]]
  estimate_mc   <- baseline[[4]]
  estimate_sc   <- baseline[[2]]
  estimate_difp <- baseline[[5]]
  estimate_did  <- baseline[[3]]

  estimate_vector <- list(estimate_sdid, estimate_dwcp, estimate_mc, estimate_sc, estimate_difp, estimate_did)
  name_vector <- c("estimate_sdid", "estimate_dwcp", "estimate_mc", "estimate_sc", "estimate_difp", "estimate_did")
  error_vector <- numeric(length(estimate_vector))

  for (i in seq_along(error_vector)) {
    error_vector[i] <- sqrt(mean(unlist(estimate_vector[[i]])^2))
    cat("The RMSE of", name_vector[i], "is", error_vector[i], "\n")
  }

  bias_vector <- numeric(length(estimate_vector))
  for (i in seq_along(bias_vector)) {
    bias_vector[i] <- mean(unlist(estimate_vector[[i]]))
    cat("The bias of", name_vector[i], "is", bias_vector[i], "\n")
  }

  # === Diagnostics ===
  cat("This is AR(2): ", ar_coef, "\n")
  cat("This is scaled_sd: ", scaled_sd, "\n")
  cat("Frobenius norm of F_ / sqrt(N*T): ", norm(F_, "F") / sqrt(N_total * T_total), "\n")
  cat("Frobenius norm of M / sqrt(N*T): ", norm(M, "F") / sqrt(N_total * T_total), "\n")
  cat("Trace of cov_mat / T_total: ", sqrt(sum(diag(cov_mat)) / T_total), "\n")

  return(invisible(list(
    estimates = setNames(estimate_vector, name_vector),
    rmse = setNames(error_vector, name_vector),
    bias = setNames(bias_vector, name_vector),
    ar_coef = ar_coef,
    scaled_sd = scaled_sd
  )))
}
