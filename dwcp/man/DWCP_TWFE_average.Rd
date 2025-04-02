\name{DWCP_TWFE_average}
\alias{DWCP_TWFE_average}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
DWCP_TWFE_average
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
DWCP_TWFE_average(x)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
%%     ~~Describe \code{x} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or standard data sets, see data().

## The function is currently defined as
DWCP_TWFE_average <- function(Y, W, treated_units, lambda_unit, lambda_time, lambda_nn, treated_periods = 10) {

  N <- nrow(Y)
  T_ <- ncol(Y)

  # dist_time
  dist_time <- abs(1:T_ - (T_ - treated_periods / 2))

  # dist_unit
  average_treated <- rowMeans(Y[treated_units, , drop = FALSE])
  mask <- matrix(1, N, T_)
  mask[, (T_ - treated_periods + 1):T_] <- 0
  A <- rowSums((Y - matrix(average_treated, nrow = N, ncol = T_, byrow = TRUE))^2 * mask)
  B <- rowSums(mask)
  dist_unit <- sqrt(A / B)

  # distance-based weights
  delta_unit <- exp(-lambda_unit * dist_unit)
  delta_time <- exp(-lambda_time * dist_time)
  delta <- outer(delta_unit, delta_time)

  # CVXR variables
  unit_effects <- Variable(1, N)
  time_effects <- Variable(1, T_)
  mu <- Variable()
  tau <- Variable()
  L <- Variable(N, T_)

  unit_factor <- t(kronecker(matrix(1, T_, 1), unit_effects))
  time_factor <- kronecker(matrix(1, N, 1), time_effects)

  # Objective function
  if (lambda_nn == Inf) {
    objective <- sum_squares((Y - mu - unit_factor - time_factor - L - W * tau) * delta)
  } else {
    objective <- sum_squares((Y - mu - unit_factor - time_factor - L - W * tau) * delta) + lambda_nn * cvxr_norm(L, "nuc")
  }

  # Constraints
  constraints <- list()

  # Problem setup and solve
  prob <- Problem(Minimize(objective), constraints)
  result <- solve(prob)

  return(result$getValue(tau))
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
