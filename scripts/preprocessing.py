#!/usr/bin/env python
import numpy as np
def standardize_features(x):
    """Standardize matrix x by feature (column).

    Returns zero-mean and unit-variance design matrix x, original mean of x and original variance of x."""

    mean = np.mean(x, axis=0)
    zero_mean=x-mean
    variance = np.std(zero_mean, axis=0)
    unit_variance=zero_mean/variance
    return unit_variance, mean, variance
def clean_data(y, tx, nan_value = -999):
    """Return clean data.

    Replaces invalid values (defaults to -999) with the mean values of the column."""

    tx[tx == nan_value] = np.nan
    mask = ~np.isnan(tx).any(axis=1)  #identify rows containing nan_value
    tx_nonan = tx[mask]
    tx_nonan_std = standardize_features(tx_nonan)
    means = tx_nonan_std[1]
    mean_matrix = np.array([means for _ in range(tx.shape[0])])
    tx[np.isnan(tx)]=mean_matrix[np.isnan(tx)]
    return np.copy(y), tx
def pca(x, max_comp=30):
    """PCA on the design matrix x.

    Returns the new design matrix with vectors transformed to eigenspace of covariance matrix and the transformation matrix."""

    covariance_matrix = np.cov(x.T)
    eigenvals, eigenvect = np.linalg.eig(covariance_matrix)
    rank_eigenvals = sorted(eigenvals, reverse=True)
    val_vect_couples = {val:vect for val, vect in zip(eigenvals, eigenvect)}
    rank_eigenvects = [val_vect_couples[val] for val in rank_eigenvals]
    diagonal2original = np.vstack(rank_eigenvects)[:,:max_comp]
    new_x = x.dot(diagonal2original)
    return new_x, diagonal2original, rank_eigenvals
def preprocess(x, y, clean=True, dopca=True, max_comp = 30, remove_cols = False, cols = None):
    """Preprocess raw data.

    Standardizes data and optionally cleans and/or does pca and/or removes columns (cols). Returns target, design matrix, original mean and standard deviation of x, transformation matrix and eigenvalues of the covariance matrix (if PCA was done else None)."""
    work_x = np.copy(x)
    work_y = np.copy(y)
    if remove_cols:
        if cols != None:
            work_x = np.delete(work_x, cols[1:], axis=1)
        else:
            raise TypeError("No column-to-remove list provided. Please do so or set remove_cols=False")
    if clean:
        y_clean, x_clean = clean_data(work_y, work_x)
    else:
        y_clean, x_clean = work_y, work_x
    x_clean, x_mean, x_var = standardize_features(x_clean)
    if dopca:
        x_clean, transform, eigenvals = pca(x_clean, max_comp=max_comp)
    else:
        transform = None
        eigenvals = None
    return y_clean, x_clean, x_mean, x_var, transform, eigenvals

def minus_one_2_zero(y):
    """Transform '-1' labels into '0' for logistic regression."""

    y_log = np.copy(y)
    y_log[y == -1] = 0
    return y_log
