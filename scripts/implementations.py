#!/usr/bin/env python
import numpy as np
from proj1_helpers import *
'''
Implementation functions for various ML algorithms
'''
#Miscellaneous functions

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
def compute_loss_logistic_new(y, tx, w):
    loss = np.sum(2*np.log(1 + np.exp(tx.dot(w)))-tx.dot(w)) - y.T.dot(tx.dot(w))
    return loss

def compute_gradient_logistic_new(y, tx, w):
    """Compute the gradient."""
    return 2*tx.T.dot(sigmoid(tx.dot(w))) - tx.T.dot(y) - tx.T.dot(np.ones(len(y)))

def compute_loss_logistic(y, tx, w):
    loss = np.sum(np.log(1 + np.exp(tx.dot(w)))) - y.T.dot(tx.dot(w))
    return loss
def sigmoid(t):
    """apply sigmoid function on t."""
    return 1./(1. + np.exp(-t))
def compute_gradient_logistic(y, tx, w):
    """Compute the gradient."""
    return tx.T.dot(sigmoid(tx.dot(w))) - tx.T.dot(y)
def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # calculate hessian
    Xw = tx.dot(w)
    sigma_Xw = sigmoid(Xw).reshape(-1)
    S = np.diag(sigma_Xw*(1 - sigma_Xw))
    return tx.T.dot(S).dot(tx)
def add_offset_column(x):
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
    # implement stochastic gradient descent
    # Define parameters to store w and loss
    w=initial_w
    gamma_0 = gamma
    ws=[]
    losses = []
    for n_iter in range(max_iters):
        for new_y, new_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute gradient and loss
            gradient = compute_gradient(new_y, new_tx, w, kind=kind)
            loss = compute_loss(new_y, new_tx, w, kind=kind)
            if adapt_gamma and gamma > 1e-4:
                gamma = gamma_0/(n_iter + 1)
            # update w by gradient
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
def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold = 1e-8, adapt_gamma = False, pr = False, accel=False, new = False):
    """return the loss, gradient, and hessian."""
    w = initial_w
    gamma_0 = gamma
    losses = []
    ws = []
    ws.append(w)
    w_bar = w
    for n_iter in range(max_iters):
        # compute gradient, loss and hessian
        if new:
            gradient = compute_gradient_logistic_new(y, tx, w)
            loss = compute_loss_logistic_new(y, tx, w)
        else:
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
def logistic_regression_SGD(y, tx, initial_w, batch_size, max_iters, gamma, adapt_gamma = False, pr = False, new=False):
    """return the loss, gradient, and hessian."""
    w = initial_w
    gamma_0 = gamma
    for n_iter in range(max_iters):
        for new_y, new_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute gradient, loss and hessian
            if new:
                gradient = compute_gradient_logistic_new(new_y, new_tx, w)
                loss = compute_loss_logistic_new(y, tx, w)
            else:
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


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold = 1e-8, adapt_gamma = False, pr = False, accel=False, new=False):
    """return the loss, gradient, and hessian."""
    w = initial_w
    gamma_0 = gamma
    losses = []
    ws = []
    ws.append(w)
    w_bar = w
    for n_iter in range(max_iters):
        # compute gradient, loss and hessian
        if new:
            loss = compute_loss_logistic_new(y, tx, w)   + lambda_ * sum(w*w)/2
            gradient = compute_gradient_logistic_new(y, tx, w) + lambda_ * w
        else:
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
