\name{fit_ar2}
\alias{fit_ar2}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
fit_ar2
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
fit_ar2(x)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{E}{
    Residual matrix from the decompose_Y function
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
fit_ar2 <- function(E) {

  T_full <- ncol(E)
  E_ts <- E[, 3:T_full]
  E_lag_1 <- E[, 2:(T_full - 1)]
  E_lag_2 <- E[, 1:(T_full - 2)]

  a_1 <- sum(diag(crossprod(E_lag_1, E_lag_1)))
  a_2 <- sum(diag(crossprod(E_lag_2, E_lag_2)))
  a_3 <- sum(diag(crossprod(E_lag_1, E_lag_2)))

  matrix_factor <- matrix(c(a_1, a_3, a_3, a_2), nrow = 2, byrow = TRUE)

  b_1 <- sum(diag(crossprod(E_lag_1, E_ts)))
  b_2 <- sum(diag(crossprod(E_lag_2, E_ts)))

  ar_coef <- solve(matrix_factor) %*% c(b_1, b_2)

  return(ar_coef)
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
