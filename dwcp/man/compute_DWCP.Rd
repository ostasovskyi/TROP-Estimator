\name{compute_DWCP}
\alias{compute_DWCP}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
A Capitalized Title (ideally limited to 65 characters)
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
compute_DWCP(x)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{
  Input dataframe
  }
  \item{Y}{
  Subset of data that is of interest
      }
  \item{W}{
    Treatment indicator matrix
      }
  \item{lambda_unit}{
      }
  \item{lambda_time}{
      }
  \item{lambda_nn}{
      }
  \item{exp_num}{
  }
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
 Random seed automatically set to 0
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

compute_DWCP <- function(data,Y,W,lambda_unit,lambda_time,lambda_nn,exp_num) {

  #df <- read.csv('CPS.csv', sep = ';')
  pck <- c("Matrix", "CVXR", "data.table", "ggplot2", "parallel", "caret", "MASS", "npmr", "synthdid") #"MCPanel")
  #Deleted MCPanel package since not available in this version of R
  lapply(pck, library, character.only = TRUE)

  result <- decompose_Y(Y, rank = 4)
  #print(names(result))

  N_total <- nrow(Y)
  assignment_vector <- numeric(N_total)

  F_ <- result$F_
  M <- result$M
  E <- result$E
  unit_factors <- result$factor_unit_scaled
  ar_coef <- fit_ar2(E)
  T_total <- ncol(F_)
  cor_matrix <- ar2_correlation_matrix(ar_coef, T_total)

  # Calculate scaled_sd
  scaled_sd <- norm(crossprod(t(E)) / N_total, type = "F") / norm(cor_matrix, type = "F")

  # Calculate cov_mat
  cov_mat <- cor_matrix * scaled_sd
  #creating the assignment vector for minimum wage
  min_wage <- matrix(data$min_wage, nrow = 40, byrow = TRUE)
  min_wage <- t(min_wage)
  Ds <- which(min_wage == TRUE, arr.ind = TRUE)[, 1]
  assignment_vector <- numeric(N_total)
  assignment_vector[Ds] <- 1

  # Fit logistic regression model
  #Creating probabilities for each assignment models
  model <- glm(assignment_vector ~ unit_factors - 1, family = binomial)

  str(unit_factors)
  summary(unit_factors)
  any(is.na(unit_factors))  # Check for missing values
  any(is.infinite(unit_factors))  # Check for infinite values
  pi <- predict(model, newdata = as.data.frame(unit_factors), type = "response")

  #model_g <- glm(assignment_vector_guns ~ unit_factors - 1, family = binomial)
  #pi_g <- predict(model_g, newdata = as.data.frame(unit_factors), type = "response")

  #model_a <- glm(assignment_vector_abort ~ unit_factors - 1, family = binomial)
  #pi_a <- predict(model_a, newdata = as.data.frame(unit_factors), type = "response")

  # Calculate conditional variance
  cond_var <- cov_mat[-1, -1] - (cov_mat[-1, (ncol(cov_mat)-2):(ncol(cov_mat)-1)] %*%
                                   solve(cov_mat[(ncol(cov_mat)-2):(ncol(cov_mat)-1), (ncol(cov_mat)-2):(ncol(cov_mat)-1)]) %*%
                                   cov_mat[(ncol(cov_mat)-2):(ncol(cov_mat)-1), -1])
  print(class(F_))
  print(class(M))
  print(class(cov_mat))
  print(class(pi))

  baseline <- table_generation(F_, M, cov_mat, pi, "norm", 10, 10, exp_num, 0)

  estimate_sdid <- baseline[1]
  estimate_dwcp <- baseline[6]
  estimate_mc <- baseline[4]
  estimate_sc <- baseline[2]
  estimate_difp <- baseline[5]
  estimate_did <- baseline[3]

  estimate_vector <- list(estimate_sdid, estimate_dwcp, estimate_mc, estimate_sc, estimate_difp, estimate_did)
  name_vector <- c("estimate_sdid", "estimate_dwcp", "estimate_mc", "estimate_sc", "estimate_difp", "estimate_did")
  error_vector <- rep(0,length(estimate_vector))

  for(i in 1:length(error_vector)){
    error_vector[i] <- sqrt(mean(unlist(estimate_vector[[i]])^2))
  }

  for(i in 1:length(error_vector)){
    cat("The RMSE of", name_vector[i], "is", error_vector[i], "\n")
  }

  cat("This is AR(2): ", ar_coef, "\n")
  cat("This is scaled_sd: ", scaled_sd, "\n")
  cat("Frobenius norm of F divided by the square root of N_total * T_total: ",norm(F_, type = "F") / sqrt(N_total * T_total), "\n")
  cat("Frobenius norm of M, divided by the square root of N_total * T_total: ", norm(M, type = "F") / sqrt(N_total * T_total), "\n")

  cat("This is trace of cov_mat divided by T_total: ", sqrt(sum(diag(cov_mat)) / T_total), "\n")
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
