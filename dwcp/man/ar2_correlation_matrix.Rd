\name{ar2_correlation_matrix}
\alias{ar2_correlation_matrix}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
ar2_correlation_matrix
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
ar2_correlation_matrix(x)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{ar_coef}{
    Output of the fit_ar2 function
}
  \item{T_}{

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
ar2_correlation_matrix <- function(ar_coef, T_) {

  result <- numeric(T_)
  result[1] <- 1
  result[2] <- ar_coef[1] / (1 - ar_coef[2])

  for (t in 3:T_) {
    result[t] <- ar_coef[1] * result[t - 1] + ar_coef[2] * result[t - 2]
  }

  index_matrix <- abs(outer(1:T_, 1:T_, "-"))
  cor_matrix <- matrix(result[index_matrix + 1], nrow = T_, ncol = T_)

  return(cor_matrix)
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
