table_generation <- function(fixed_effects, interactive_data, cov_mat, prob, noise = "norm", treatment_periods, treatment_units, exp_num, ran_seed=0){
  #Setting random seed for reproducibility
  set.seed(ran_seed)


  #Initializing the vectors for each of the estimators
  estimate_sdid <<- rep(0,exp_num) #Romy: maybe should be <-?
  estimate_dwcp <<- rep(0,exp_num)
  estimate_mc <<- rep(0,exp_num)
  estimate_sc <<- rep(0,exp_num)
  estimate_difp <<- rep(0,exp_num)
  estimate_did <<- rep(0,exp_num)

  #Running the experiment exp_num times for robustness default is 1000 as used in SDID paper
  for (experiment in 1:exp_num){

    #Generating data based on simulation requirements
    data <-generate_data(fixed_effects, interactive_data, cov_mat, prob, noise, treatment_periods, treatment_units)
    Y_true <<- data$Y
    W_true <<- data$W
    treated_units <<- data$index

    if (length(Y_true) %% nrow(Y_true) != 0) {
      stop("Y_true dimensions are inconsistent with expected operations.")
}
if (length(W_true) %% nrow(W_true) != 0) {
      stop("W_true dimensions are inconsistent with expected operations.")
}
    #print(str(Y_true))  # Check structure of Y_true
    #print(str(W_true))  # Check structure of W_true
    #storing the estimates of the different methods
    estimate_did[experiment] <- did_estimate(Y_true,
                                             nrow(Y_true) - treatment_units,
                                             ncol(Y_true)-treatment_periods
                                             )

    #print(Y_true)
    estimate_mc[experiment] <- DWCP_TWFE_average(Y_true, W_true, treatment_units, lambda_unit = 0, lambda_time=0, lambda_nn=.6, treatment_periods)

    estimate_sdid[experiment] <- synthdid_estimate(Y_true,
                                                   nrow(Y_true) - treatment_units,
                                                   ncol(Y_true)- treatment_periods
    )

    estimate_sc[experiment] <- sc_estimate(Y_true,
                                           nrow(Y_true) - treatment_units,
                                           ncol(Y_true)-treatment_periods
    )

    estimate_difp[experiment] <- DIFP_TWFE(Y_true, W_true, treatment_units, treatment_periods)
    estimate_dwcp[experiment] <- DWCP_TWFE_average(Y_true, W_true, treatment_units, lambda_unit = 1.5, lambda_time=.25, lambda_nn=.176)
  }
  return(list(estimate_sdid, estimate_sc, estimate_did, estimate_mc,  estimate_difp,  estimate_dwcp))

}
