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
n_samples = 100
alpha_min, alpha_max = 1, 50  
a, b                 = np.log2(alpha_min), np.log2(alpha_max)  
alpha                = (alpha_max+alpha_min) - np.logspace(a, b, num=n_samples, base=2)
s                    = np.linspace(0.2, 0.8, n_samples) 

data_s1, data_s2 =  np.meshgrid(alpha, s)
data = np.array([data_s1.ravel(), data_s2.ravel()]).T
n_70          = int(0.7 * len(data))

# Mandatory training set
mandatory_train   = {0, 99, 9900, 9999} # corners
mandatory_test    = {3609, 5887, 9294}  # ploted points
remaining_indices = set(range(0, len(data))) - mandatory_train - mandatory_test
n_mandatory_train = len(mandatory_train)

# Generate random train/test split
train_indices     = set(np.random.choice(list(remaining_indices), size= n_70-n_mandatory_train, replace=False))
train_indices    |= mandatory_train
test_indices      = set(range(len(data))) - train_indices

train_indices = np.array(sorted(train_indices))
test_indices  = np.array(sorted(test_indices))

# Selection of 7 random indices and the mandatory test indices
test_subset       = np.array(test_indices[np.random.choice(len(test_indices), 10, replace=False)])
alpha_test_subset = data[test_subset]
alpha_special     = data[list(mandatory_test)]

# Create DataFrames
dataset_df = pd.DataFrame(data, columns=['alpha', 's'])
indices_train_df = pd.DataFrame({'train_indices': train_indices})
indices_test_df  = pd.DataFrame({'test_indices': test_indices})
set_test_subset_df = pd.DataFrame(alpha_test_subset, columns=['alpha', 's'])
set_special_df = pd.DataFrame(alpha_special, columns=['alpha', 's'])

# Save to CSV files
folder = 'Dataset/'
os.makedirs(folder, exist_ok=True)
dataset_df.to_csv(folder + 'dataset.csv', index=False)
indices_train_df.to_csv(folder + 'train_indices.csv', index=False)
indices_test_df.to_csv(folder + 'test_indices.csv', index=False)
set_test_subset_df.to_csv(folder + 'test_subset.csv', index=False)
set_special_df.to_csv(folder + 'special_test.csv', index=False)
