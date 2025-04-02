run_MCNNM <- function(Y_obs, O, lambd = 10, threshold = 0.01, print_every = NULL, max_iters = 20000) {

  # Initialize L_prev with values from Y_obs outside the observed entries
  L_prev <- getPOinv(Y_obs, O)
  change <- 1000
  iters <- 0

  while ((change > threshold) && (iters < max_iters)) {

    # Get observed and unobserved parts
    PO <- getPO(Y_obs, O)
    PO_inv <- getPOinv(L_prev, O)

    # Update L_star and shrink the matrix
    L_star <- PO + PO_inv
    L_new <- shrink_lambda(L_star, lambd)

    # Calculate the change
    change <- norm(L_prev - L_new, type = "F")

    # Update L_prev and iteration count
    L_prev <- L_new
    iters <- iters + 1

    # Print progress if requested
    if (!is.null(print_every) && (iters %% print_every == 0)) {
      cat("Iteration:", iters, "Change:", change, "\n")
    }
  }

  return(L_new)
}
