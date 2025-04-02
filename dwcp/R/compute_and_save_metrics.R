compute_and_save_metrics <- function(variables, true_value = 0, output_csv = "metrics_results.csv") {
  # Initialize an empty list to store results
  results_list <- list()

  # Loop through each variable (list of vectors)
  for (var_name in names(variables)) {
    variable <- variables[[var_name]]

    # Apply metrics computation to each vector in the variable
    metrics <- do.call(rbind, lapply(variable, compute_metrics, true_value = true_value))

    # Convert metrics to a data frame and add method/variable labels
    metrics_df <- as.data.frame(metrics)
    metrics_df$method <- seq_len(nrow(metrics_df))  # Method identifier
    metrics_df$variable <- var_name

    # Append to results list
    results_list[[var_name]] <- metrics_df
  }

  # Combine all results into a single data frame
  final_results <- do.call(rbind, results_list)

  # Reshape to wide format for CSV
  final_results_wide <- reshape(
    final_results,
    timevar = "variable",
    idvar = "method",
    direction = "wide"
  )

  # Save to CSV
  write.csv(final_results_wide, output_csv, row.names = FALSE)

  return(final_results_wide) # Return results for inspection
}
