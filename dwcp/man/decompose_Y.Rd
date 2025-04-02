\name{decompose_Y}
\alias{decompose_Y}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
decompose_Y
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
decompose_Y(x)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
        Input data
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
decompose_Y <- function(Y, rank = 4) {
  N <- nrow(Y)
  T_ <- ncol(Y)

  # Perform Singular Value Decomposition
  svd_result <- svd(Y)
  u <- svd_result$u
  s <- svd_result$d
  v <- svd_result$v

  # Extract the rank components
  factor_unit <- u[, 1:rank]
  factor_time <- t(v)[1:rank, ]

  # Calculate the low-rank approximation matrix L
  L <- factor_unit %*% diag(s[1:rank]) %*% factor_time

  # Calculate the residual matrix E
  E <- Y - L

  # Calculate F and M matrices
  F_ <- outer(rowMeans(L), colMeans(L), `+`) - mean(L)
  M <- L - F_

  # Return the results
  list(F_ = F_, M = M, E = E, factor_unit_scaled = factor_unit * sqrt(N))
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
