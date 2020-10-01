# Comparison of gradient and Newton boosting

This is the code to reproduce the results of Sigrist (2018) "Gradient and Newton Boosting for Classification and Regression". We compare gradient and Newton boosting, as well as hybrid gradient-Newton boosting with trees as base learners using various datasets and loss functions. See https://arxiv.org/abs/1808.03064 for more details.

Run the file 'Compare_boosting.py'. Running all experiments with all settings for real-world and simulated datasets takes some time. The parameter 'which_data' specifies which experiments are run. The file(s) 'results_summary_simulation=.csv' in the results folder contains the summary of the results.