\name{cross_validation}
\alias{cross_validation}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
A Capitalized Title (ideally limited to 65 characters)
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
cross_validation(Y, W, lambda_grid, num_runs = 500)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
%%     ~~Describe \code{Y} here~~
}
  \item{W}{
%%     ~~Describe \code{W} here~~
}
  \item{lambda_grid}{
%%     ~~Describe \code{lambda_grid} here~~
}
  \item{num_runs}{
%%     ~~Describe \code{num_runs} here~~
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
function (Y, W, lambda_grid, num_runs = 500) 
{
    N <- nrow(Y)
    T_ <- ncol(Y)
    N_treat <- nrow(W) - 9
    T_treat <- ncol(W) - 9
    best_lambda <- NULL
    best_std_dev <- Inf
    for (lambda_time in lambda_grid) {
        for (lambda_unit in lambda_grid) {
            for (lambda_nn in lambda_grid) {
                weights <- calculate_weights(lambda_time, lambda_unit, 
                  Y, W, N_treat, T_, T_treat)
                ate_estimates <- numeric(num_runs)
                for (run in 1:num_runs) {
                  control_units <- which(rowSums(W) == 0)
                  placebo_units <- sample(control_units, N_treat, 
                    replace = FALSE)
                  init_params <- c(0, 0, 0, 0, rep(0, length(Y)))
                  print(class(init_params))
                  print(class(Y))
                  print(class(W))
                  print(class(weights))
                  print(class(placebo_units))
                  result <- optim(init_params, fn = objective, 
                    Y = Y, W = W, weights = weights, placebo_units = placebo_units, 
                    lambda_nn = lambda_nn)
                  if (result$convergence == 0) {
                    ate_estimates[run] <- result$value
                  }
                }
                std_dev <- sd(ate_estimates)
                if (std_dev < best_std_dev) {
                  best_std_dev <- std_dev
                  best_lambda <- c(lambda_time, lambda_unit, 
                    lambda_nn)
                }
            }
        }
    }
    list(best_lambda = best_lambda, best_std_dev = best_std_dev)
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
