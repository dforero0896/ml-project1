#!/usr/bin/env python
import numpy as np
def standardize_features(x):
    '''Standardize matrix x by feature (column)'''
    mean = np.mean(x, axis=0)
    zero_mean=x-mean
    variance = np.std(zero_mean, axis=0)
    unit_variance=zero_mean/variance
    return unit_variance, mean, variance
def clean_data(y, tx, nan_value = -999):
    '''Return clean data.'''
    tx[tx == nan_value] = np.nan
    mask = ~np.isnan(tx).any(axis=1)  #identify rows containing nan_value
    tx_nonan = tx[mask]
    tx_nonan_std = standardize_features(tx_nonan)
    means = tx_nonan_std[1] 
    mean_matrix = np.array([means for _ in range(tx.shape[0])])
    tx[np.isnan(tx)]=mean_matrix[np.isnan(tx)]
    return y, tx
def pca(x, max_comp=30):
    covariance_matrix = np.cov(x.T)
    eigenvals, eigenvect = np.linalg.eig(covariance_matrix)
    rank_eigenvals = sorted(eigenvals, reverse=True)
    diagonal2original = np.vstack(eigenvect[:max_comp])
    new_x = (np.linalg.inv(diagonal2original).dot(x.T)).T
    return new_x, diagonal2original

def preprocess(x, y, clean=True, dopca=True, max_comp = 30):
    if clean:
        y_clean, x_clean = clean_data(y, x)
    else:
        y_clean, x_clean = y, x
    x_clean, x_mean, x_var = standardize_features(x_clean)
    if dopca:
        x_clean, transform = pca(x_clean, max_comp=max_comp)
    return y_clean, x_clean, x_mean, x_var, transform

