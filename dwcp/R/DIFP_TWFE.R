DIFP_TWFE <- function(Y, W, treated_units, treated_periods) {

  # Define indices for treated and control periods
  T_post <- seq((ncol(Y) - treated_periods + 1), ncol(Y))  # Treated periods (last 10 columns)
  T_pre <- seq_len(ncol(Y) - treated_periods)              # Pre-treatment periods

  # Remove treated units and periods for control group matrix (X)
  X <- t(Y[-treated_units, T_pre])  # Control group: pre-treatment data only

  # Mean of treated units in the pre-treatment period
  y <- colMeans(Y[treated_units, T_pre, drop = FALSE])

  # Number of control units
  control_units <- ncol(X)

  # CVXR variables for optimization
  unit_weights <- CVXR::Variable(control_units, nonneg = TRUE)
  intercept <- CVXR::Variable()

  # Constraints for optimization
  constraints <- list(sum(unit_weights) == 1)  # Weights sum to 1

  # Objective function: minimize squared differences between treated and control
  objective <- sum_squares(y - X %*% unit_weights - intercept)

  # Solve the optimization problem
  prob <- CVXR::Problem(CVXR::Minimize(objective), constraints)
  result <- CVXR::solve(prob, solver = "ECOS")

  # Extract weights and intercept
  estimated_weights <- result$getValue(unit_weights)
  estimated_intercept <- result$getValue(intercept)

  # Predict counterfactual outcomes for treated units in treated periods
  X_predict <- t(Y[-treated_units, T_post])  # Control group data in post-treatment period
  y_predict <- X_predict %*% estimated_weights + estimated_intercept  # Counterfactual predictions

  # Observed outcomes of treated units in treated periods
  y_actual <- colMeans(Y[treated_units, T_post, drop = FALSE])

  # Compute treatment effect
  treatment_effect <- mean(y_actual - y_predict)

  # Return treatment effect
  return(treatment_effect)
}
