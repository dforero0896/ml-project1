#!/usr/bin/env python
import numpy as np
'''
Implementation functions for various ML algorithms
'''
#Miscellaneous functions
def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred
def clean_data(y, tx, nan_value = -999):
    '''Return clean data.'''
    tx[tx == nan_value] = np.nan
    mask = ~np.isnan(tx).any(axis=1)  #identify rows containing nan_value
    tx = tx[mask]
    y = y[mask]
    return y, tx
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x
def standardize_features(x):
    '''Standardize matrix x by feature (column)'''
    mean = np.mean(x, axis=0)
    zero_mean=x-mean
    variance = np.std(zero_mean, axis=0)
    unit_variance=zero_mean/variance
    return unit_variance, mean, variance
def compute_loss(y, tx, w, kind='mse'):
    error = y - tx.dot(w)
    if kind == 'mse':
        return error.dot(error)/(2*len(y))
    elif kind == 'mae':
        return sum(np.abs(error))/len(y)
    else:
        raise NotImplementedError
def compute_gradient(y, tx, w, kind='mse'):
    """Compute the gradient."""
    error = y - tx.dot(w)
    if kind == 'mse':
        return -tx.T.dot(error)/len(y)
    elif kind == 'mae':
        return -np.sign(error).dot(tx)/len(y) #Sum rows
    else:
        raise NotImplementedError

def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio.
    Returns: x_train, y_train, x_test, y_test
    """
    # Maximum number of elements to include in the train set
    max_ind = int(ratio * len(x))
    # set seed
    np.random.seed(seed)
    # Generate an ordered list of indices
    ids = np.arange(len(x))
    # Suffle indices
    np.random.shuffle(ids)
    #Get shuffled arrays
    x_shuf = x[ids]
    y_shuf = y[ids]
    # Return both subsets
    return x_shuf[:max_ind], y_shuf[:max_ind], x_shuf[max_ind:], y_shuf[max_ind:]
def accuracy_ratio(prediction, labels):
    mask = (prediction==labels)
    count = np.zeros(len(prediction))
    count[mask]=1
    correct = float(sum(count))
    return correct/len(prediction)
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = [np.array(x)**j for j in range(degree+1)]
    return phi.T

#ML Implementations
def least_squares_GD(y, tx, initial_w, max_iters, gamma, kind='mse', adapt_gamma = False, pr = False):
    """Linear regression using Gradient descent algorithm."""
    w = initial_w.astype(float)
    gamma_0 = gamma
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w, kind=kind)
        loss = compute_loss(y, tx, w, kind=kind)
        # update w by gradient
        if adapt_gamma:
            gamma = gamma_0/(n_iter + 1)
        w = w - gamma * gradient
        if pr == True and n_iter%100 == 0:
            print("GD ({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma, kind='mse', adapt_gamma = False, pr = False):
    """Stochastic gradient descent algorithm."""
    # implement stochastic gradient descent
    # Define parameters to store w and loss
    w=initial_w
    gamma_0 = gamma
    for n_iter in range(max_iters):
        for new_y, new_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute gradient and loss
            gradient = compute_gradient(new_y, new_tx, w, kind=kind)
            loss = compute_loss(new_y, new_tx, w, kind=kind)
            if adapt_gamma and gamma > 1e-4:
                gamma = gamma_0/(n_iter + 1)
            # update w by gradient
            w = w - gamma * gradient
            if pr == True and n_iter%100 == 0:
                print("SGD ({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss
def least_squares(y, tx):
    """calculate the least squares solution."""
    gram_matrix = tx.T.dot(tx)
    w = np.linalg.solve(gram_matrix, tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    gram_matrix = tx.T.dot(tx)
    reg_term = 2*len(y)*lambda_*np.identity(gram_matrix.shape[0])
    w = np.linalg.solve(gram_matrix + reg_term, tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    id_test = k_indices[k]
    id_train = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
    x_test = x[id_test]
    x_train = x[id_train]
    y_test = y[id_test]
    y_train = y[id_train]
    tx_train = x_train
    tx_test = x_test
    # form data with polynomial degree
    #tx_train = build_poly(x_train, degree)
    #tx_test = build_poly(x_test, degree)
    # ridge regression
    weight, loss_tr = ridge_regression(y_train, tx_train, lambda_)
    # calculate the loss for train and test data
    loss_te = compute_loss(y_test, tx_test, weight)
    accuracy = accuracy_ratio(predict_labels(weight, tx_test), y_test)

    return loss_tr, loss_te, accuracy