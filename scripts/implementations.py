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
    tx_nonan = tx[mask]
    tx_nonan_std = standardize_features(tx_nonan)
    means = tx_nonan_std[1] 
    mean_matrix = np.array([means for _ in range(tx.shape[0])])
    tx[np.isnan(tx)]=mean_matrix[np.isnan(tx)]
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
    '''Computes the accuracy ratio between the prediction and the test labels.'''
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
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.array([np.array(x[:,i])**j for j in range(1,degree+1) for i in range(x.shape[1])])
    phi_off = np.array(np.c_[np.ones(x.shape[0]), phi.T])
    return phi_off
def cross_validation(y, x, k_indices, k, lambda_, degree, clean=True):
    """Returns the train and test loss as well as the test accuracy for ridge regression."""
    # get k'th subgroup in test, others in train
    id_test = k_indices[k]
    id_train = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
    x_test = x[id_test]
    x_train = x[id_train]
    y_test = y[id_test]
    y_train = y[id_train]
    #Clean
    if clean:
        y_train, x_train = clean_data(y_train, x_train)
        y_test, x_test = clean_data(y_test, x_test)
    # Standardize
    x_train_std = standardize(x_train)[0]
    x_test_std = standardize(x_test)[0]
    # Define feature matrix
    tx_train = build_poly(x_train_std, degree)
    tx_test = build_poly(x_test_std, degree)
    # ridge regression
    weight, loss_tr = ridge_regression(y_train, tx_train, lambda_)
    # calculate the loss for train and test data
    loss_te = compute_loss(y_test, tx_test, weight)
    accuracy = accuracy_ratio(predict_labels(weight, tx_test), y_test)
    return loss_tr, loss_te, accuracy

def compute_loss_logistic(y, tx, w):
    '''Compute the loss for logistic regression.'''
    loss = sum(np.log(1 + np.exp(tx.dot(w)))) - y.T.dot(tx.dot(w))
    return loss
def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1 + np.exp(t))
def compute_gradient_logistic(y, tx, w):
    """Compute the gradient for logistic regression loss."""
    return tx.T.dot(sigmoid(tx.dot(w))) - tx.T.dot(y)
def calculate_hessian(y, tx, w):
    """Return the hessian of the loss function."""
    # calculate hessian
    Xw = tx.dot(w)
    sigma_Xw = sigmoid(Xw).reshape(-1)
    S = np.diag(sigma_Xw*(1 - sigma_Xw))
    return tx.T.dot(S).dot(tx)

def add_offset_column(x):
    '''Add offset column to the data x.'''
    return np.c_[np.ones(x.shape[0]), x]

#ML Implementations
def least_squares_GD(y, tx, initial_w, max_iters, gamma, kind='mse', adapt_gamma = False, pr = False, accel = False):
    """Linear regression using Gradient descent algorithm."""
    w = initial_w.astype(float)
    gamma_0 = gamma
    ws = []
    ws.append(w)
    w_bar = w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w, kind=kind)
        loss = compute_loss(y, tx, w, kind=kind)
        # update w by gradient
        if adapt_gamma:
            gamma = gamma_0/(n_iter + 1)
        if accel:
            w = w_bar - gamma * compute_gradient(y, tx, w_bar, kind=kind)
            w_bar = w + ((n_iter)/(n_iter + 1)) * (w - ws[-1])
        else:
            w = w - gamma * gradient
        ws.append(w)
        if pr == True and n_iter%100 == 0:
            print("GD ({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma, kind='mse', adapt_gamma = False, pr = False, choose_best = False):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    w=initial_w
    gamma_0 = gamma
    ws=[]
    losses = []
    for n_iter in range(max_iters):
        for new_y, new_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # Compute gradient and loss
            gradient = compute_gradient(new_y, new_tx, w, kind=kind)
            loss = compute_loss(new_y, new_tx, w, kind=kind)
            if adapt_gamma and gamma > 1e-4:
                gamma = gamma_0/(n_iter + 1)
            # Update w by gradient
            w = w - gamma * gradient
            ws.append(w)
            losses.append(compute_loss(y, tx, w))
            if pr == True and n_iter%100 == 0:
                print("SGD ({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))
    if choose_best:
        w = ws[np.argmin(losses)]
        loss = min(losses)
    return w, loss
    
def least_squares(y, tx):
    """Calculate the least squares solution."""
    gram_matrix = tx.T.dot(tx)
    w = np.linalg.solve(gram_matrix, tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss
def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""
    gram_matrix = tx.T.dot(tx)
    reg_term = 2*len(y)*lambda_*np.identity(gram_matrix.shape[0])
    w = np.linalg.solve(gram_matrix + reg_term, tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss
def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold = 1e-8, adapt_gamma = False, pr = False, accel=False):
    """Returns the weights and loss for the regularized logistic regression using GD."""
    w = initial_w
    gamma_0 = gamma
    losses = []
    ws = []
    ws.append(w)
    w_bar = w
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient_logistic(y, tx, w)
        loss = compute_loss_logistic(y, tx, w)
        # update w by gradient
        if adapt_gamma:
            gamma = gamma_0/(n_iter + 1)
        if accel:
            w = w_bar - gamma * compute_gradient_logistic(y, tx, w_bar)
            w_bar = w + ((n_iter)/(n_iter + 1)) * (w - ws[-1])
        else:
            w = w - gamma * gradient        
        if pr == True and n_iter%100 == 0:
            print("Logistic Regression GD ({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

def logistic_regression_SGD(y, tx, initial_w, batch_size, max_iters, gamma, adapt_gamma = False, pr = False):
    """Returns the weights and loss for the regularized logistic regression using SGD."""
    w = initial_w
    gamma_0 = gamma
    for n_iter in range(max_iters):
        for new_y, new_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute gradient, loss and hessian
            gradient = compute_gradient_logistic(new_y, new_tx, w)
            loss = compute_loss_logistic(y, tx, w)
            if adapt_gamma and gamma > 1e-4:
                gamma = gamma_0/(n_iter + 1)
            # update w by gradient
            w = w - gamma * gradient
            if pr == True and n_iter%100 == 0:
                print("Logistic Regression SGD ({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold = 1e-8, adapt_gamma = False, pr = False, accel=False):
    """Returns the weights and loss for the regularized logistic regression using GD."""
    w = initial_w
    gamma_0 = gamma
    losses = []
    ws = []
    ws.append(w)
    w_bar = w
    for n_iter in range(max_iters):
        # compute gradient and loss
        loss = compute_loss_logistic(y, tx, w) + lambda_ * sum(w*w)/2
        gradient = compute_gradient_logistic(y, tx, w) + lambda_ * w
        # update w by gradient
        if adapt_gamma:
            gamma = gamma_0/(n_iter + 1)
        if accel:
            w = w_bar - gamma * compute_gradient_logistic(y, tx, w_bar) + lambda_ * w
            w_bar = w + ((n_iter)/(n_iter + 1)) * (w - ws[-1])
        else:
            w = w - gamma * gradient
        if pr == True and n_iter%100 == 0:
            print(" Regularized Logistic Regression GD ({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss