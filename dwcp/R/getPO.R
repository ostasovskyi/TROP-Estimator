getPO <- function(A, O) {
  # Create an output matrix initialized to zeros with the same dimensions as A
  A_out <- matrix(0, nrow = nrow(A), ncol = ncol(A))

  # Use the indices in O to copy elements from A to A_out
  A_out[cbind(O[,1], O[,2])] <- A[cbind(O[,1], O[,2])]

  return(A_out)
}
