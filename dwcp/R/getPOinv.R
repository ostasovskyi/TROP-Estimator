getPOinv <- function(A, O) {
  # Create a copy of A to modify
  A_out <- A

  # Set elements at specified indices in O to 0
  A_out[cbind(O[,1], O[,2])] <- 0

  return(A_out)
}
