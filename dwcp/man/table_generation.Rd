\name{table_generation}
\alias{table_generation}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
table_generation
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
table_generation(x)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{fixed_effects}{
%%     ~~Describe \code{x} here~~
}
\item{interactive_data}{
%%     ~~Describe \code{x} here~~
}
\item{cov_mat}{
%%     ~~Describe \code{x} here~~
}
\item{noise}{
%%     ~~Describe \code{x} here~~
}
\item{treatment_periods}{
%%     ~~Describe \code{x} here~~
}
\item{treatment_units}{
%%     ~~Describe \code{x} here~~
}
\item{exp_num}{
%%     ~~Describe \code{x} here~~

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

table_generation <- function(fixed_effects, interactive_data, cov_mat, prob, noise = "norm", treatment_periods, treatment_units, exp_num, ran_seed=0){

  #Setting random seed for reproducibility
  set.seed(ran_seed)


  #Initializing the vectors for each of the estimators
  estimate_sdid <- rep(0,exp_num) #Romy: maybe should be <-?
  estimate_dwcp = rep(0,exp_num)
  estimate_mc = rep(0,exp_num)
  estimate_sc = rep(0,exp_num)
  estimate_difp = rep(0,exp_num)
  estimate_did = rep(0,exp_num)

  #Running the experiment exp_num times for robustness default is 1000 as used in SDID paper
  for (expirement in 1:exp_num){

    #Generating data based on simulation requirements
    data <-generate_data(fixed_effects, interactive_data, cov_mat, prob, noise, treatment_periods, treatment_units)
    print(class(data))
    print(class(data$Y))
    print(class(data$W))
    print(class(data$index))

    Y_true <- data$Y
    W_true <- data$W
    treated_units <- data$index

    if (length(Y_true) %% nrow(Y_true) != 0) {
  stop("Y_true dimensions are inconsistent with expected operations.")
}
if (length(W_true) %% nrow(W_true) != 0) {
  stop("W_true dimensions are inconsistent with expected operations.")
}
    #print(str(Y_true))  # Check structure of Y_true
    #print(str(W_true))  # Check structure of W_true
    #storing the estimates of the different methods
    estimate_did[expirement] <- did_estimate(Y_true,
                                             nrow(Y_true) - treatment_units,
                                             ncol(Y_true)-treatment_periods
                                            )


    estimate_mc[expirement] <- DWCP_TWFE_average(Y_true, W_true, treatment_units, lambda_unit = 0, lambda_time=0, lambda_nn=.6, treatment_periods)

    estimate_sdid[expirement] <- synthdid_estimate(Y_true,
                                                   nrow(Y_true) - treatment_units,
                                                   ncol(Y_true)- treatment_periods
                                                   )

    estimate_sc[expirement] <- sc_estimate(Y_true,
                                           nrow(Y_true) - treatment_units,
                                           ncol(Y_true)-treatment_periods
                                           )

    estimate_difp[expirement] <- DIFP_TWFE(Y_true, W_true, treatment_units, treatment_periods)
    estimate_dwcp[expirement] <- DWCP_TWFE_average(Y_true, W_true, treatment_units, lambda_unit = 1.5, lambda_time=.25, lambda_nn=.176)
  }

return(list(estimate_sdid, estimate_sc, estimate_did, estimate_mc,  estimate_difp,  estimate_dwcp))
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
