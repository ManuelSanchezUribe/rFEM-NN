#########################################################################################################
# An r-adaptive finite element method using neural networks for parametric self-adjoint elliptic problems
# Author: Danilo Aballay, Federico Fuentes, Vicente Iligaray, Ángel J. Omella,
#         David Pardo, Manuel A. Sánchez, Ignacio Tapia, Carlos Uriarte
#########################################################################################################

# Import libraries
import numpy as np
import pandas as pd
import os
np.random.seed(40)

# Data set
n_samples     = 1000
n_70          = int(0.7 * n_samples)
data_set      = np.logspace(-4, 4, n_samples)

# Mandatory training set
mandatory_train   = {0, 1, n_samples - 1, n_samples - 2}
mandatory_test    = {83, 467, 868}
remaining_indices = set(range(0, n_samples)) - mandatory_train - mandatory_test
n_mandatory_train = len(mandatory_train)

# Generate random train/test split
train_indices     = set(np.random.choice(list(remaining_indices), size= n_70-n_mandatory_train, replace=False))
train_indices    |= mandatory_train
test_indices      = set(range(n_samples)) - train_indices

train_indices = np.array(sorted(train_indices))
test_indices  = np.array(sorted(test_indices))

# Selection of 7 random indices and the mandatory test indices
test_subset       = np.array(test_indices[np.random.choice(len(test_indices), 10, replace=False)])
alpha_test_subset = data_set[test_subset]
alpha_special     = data_set[list(mandatory_test)]

# Create DataFrames
dataset_df           = pd.DataFrame({'data': data_set})
indices_train_df     = pd.DataFrame({'train_indices': train_indices})
indices_test_df      = pd.DataFrame({'test_indices': test_indices})
alpha_test_subset_df = pd.DataFrame({'test_subset': alpha_test_subset})
alpha_special_df     = pd.DataFrame({'special_test': alpha_special})

# Save to CSV files
folder = 'Dataset/'
dataset_df.to_csv(folder + 'dataset.csv', index=False)
indices_train_df.to_csv(folder + 'train_indices.csv', index=False)
indices_test_df.to_csv(folder + 'test_indices.csv', index=False)
alpha_test_subset_df.to_csv(folder + 'test_subset.csv', index=False)
alpha_special_df.to_csv(folder + 'special_test.csv', index=False)
