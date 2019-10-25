#!/usr/bin/env python
import numpy as np
from proj1_helpers import *
from preprocessing import *
"""
Implementation functions for various ML algorithms
"""
#Miscellaneous functions

def compute_loss(y, tx, w, kind='mse'):
    """MSE or MAE loss functions.

    Return Mean Squared Error (mse) or Mean Absolute Error (mae) loss
    function given labels y, design matrix tx and model weights w. """

    error = y - tx.dot(w)
    if kind == 'mse':
        return error.dot(error)/(2*len(y))
    elif kind == 'mae':
        return sum(np.abs(error))/len(y)
    else:
        raise NotImplementedError
def compute_gradient(y, tx, w, kind='mse'):
    """MSE or MAE loss' gradients.

    Return Mean Squared Error (mse) or Mean Absolute Error (mae) gradients
    given labels y, design matrix tx and model weights w."""

    error = y - tx.dot(w)
    if kind == 'mse':
        return -tx.T.dot(error)/len(y)
    elif kind == 'mae':
        return -np.sign(error).dot(tx)/len(y) #Sum rows
    else:
        raise NotImplementedError
def compute_loss_logistic_new(y, tx, w):
    """Logistic regression loss function.

    Return logistic regression loss (-loglikelihood) function given
    labels y, design matrix tx and model weights w. If labels are 1 and -1. """

    loss = np.sum(2*np.log(1 + np.exp(tx.dot(w)))-tx.dot(w)) - y.T.dot(tx.dot(w))
    return loss

def compute_gradient_logistic_new(y, tx, w):
    """Logistic regression loss' gradient.

    Return logistic regression gradient given labels y,
    design matrix tx and model weights w. If labels are 1 and -1."""

    return 2*tx.T.dot(sigmoid(tx.dot(w))) - tx.T.dot(y) - tx.T.dot(np.ones(len(y)))

def compute_loss_logistic(y, tx, w):
    """Logistic regression loss function.

    Return logistic regression loss (-loglikelihood) function given
    labels y, design matrix tx and model weights w. If labels are 1 and 0."""
    loss = np.sum(np.log(1 + np.exp(tx.dot(w)))) - y.T.dot(tx.dot(w))
    return loss
def sigmoid(t):
    """Apply sigmoid function on t."""
    return 1./(1. + np.exp(-t))
def compute_gradient_logistic(y, tx, w):
    """Logistic regression loss' gradient.

    Return logistic regression gradient given labels y,
    design matrix tx and model weights w. If labels are 1 and -1."""
    return tx.T.dot(sigmoid(tx.dot(w))) - tx.T.dot(y)
def calculate_hessian(y, tx, w):
    """Calculate Hessian matrix for logistic regression loss.

    Return logistic regression hessian given labels y,
    design matrix tx and model weights w. If labels are 1 and 0."""
    Xw = tx.dot(w)
    sigma_Xw = sigmoid(Xw).reshape(-1)
    S = np.diag(sigma_Xw*(1 - sigma_Xw))
    return tx.T.dot(S).dot(tx)
def add_offset_column(x):
    """Add a column of ones to the left of the design matrix x."""
    return np.c_[np.ones(x.shape[0]), x]

#ML Implementations
def least_squares_GD(y, tx, initial_w, max_iters, gamma, kind='mse', adapt_gamma = False, pr = False, accel = False):
    """Linear regression using Gradient descent algorithm.

    Iteratively compute the model weights given data y, a design matrix tx, an initial condition (on w), a maximum of iterations and a step size (gamma); with either MSE or MAE loss. Additional parameters allow for an adapting step size, output printing each 100 epochs and use of accelerated gradient descent algorithm.
    Returns weights and loss."""

    w = initial_w.astype(float)
    gamma_0 = gamma
    ws = []
    ws.append(w)
    w_bar = w
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w, kind=kind)
        loss = compute_loss(y, tx, w, kind=kind)
        if np.isinf(loss):
            raise ValueError("Infinite loss, exiting.")
            break
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
    """Linear regression using Stochastic Gradient descent algorithm.

    Iteratively compute the model weights given data y, a design matrix tx, an initial condition (on w), a batch size, a maximum of iterations and a step size (gamma); with either MSE or MAE loss. Additional parameters allow for an adapting step size, output printing each 100 epochs and choosing the weights that registered the smallest loss in the path taken.
    Returns weights and loss."""

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
            if np.isinf(loss):
                raise ValueError("Infinite loss, exiting.")
                break
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
    """Calculate the least squares solution.

    Returns weights and loss for the linear regression with least squares given data y and design matrix tx."""

    gram_matrix = tx.T.dot(tx)
    w = np.linalg.solve(gram_matrix, tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Return weights and loss for the linear regression with least squares plus a regularization term (Ridge Regression) given data y and design matrix tx."""
    gram_matrix = tx.T.dot(tx)
    reg_term = 2*len(y)*lambda_*np.identity(gram_matrix.shape[0])
    w = np.linalg.solve(gram_matrix + reg_term, tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss
def logistic_regression(y, tx, initial_w, max_iters, gamma, threshold = 1e-8, adapt_gamma = False, pr = False, accel=False, new = False):
    """Linear regression using Gradient Descent on the Logistic Regression objective.

    Iteratively compute the model weights given data y, a design matrix tx, an initial condition (on w), a batch size, a maximum of iterations and a step size (gamma); with logistic regression loss. Additional parameters allow for early stopping of iterations when convergence threshold is reached, an adapting step size, output printing each 100 epochs and use of accelerated gradient descent algorithm.
    Returns weights and loss."""
    # TODO: Check documentation for use of new parameter.

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
        if np.isinf(loss):
            raise ValueError("Infinite loss, exiting.")
            break
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
    """Linear regression using Stochastic Gradient Descent on the Logistic Regression objective.

    Iteratively compute the model weights given data y, a design matrix tx, an initial condition (on w), a batch size, a maximum of iterations and a step size (gamma); with logistic regression loss. Additional parameters allow for an adapting step size and output printing each 100 epochs."""
    # TODO: Check documentation for use of new parameter.

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
            if np.isinf(loss):
                raise ValueError("Infinite loss, exiting.")
                break
            if adapt_gamma and gamma > 1e-4:
                gamma = gamma_0/(n_iter + 1)
            # update w by gradient
            w = w - gamma * gradient
            if pr == True and n_iter%100 == 0:
                print("Logistic Regression SGD ({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, threshold = 1e-8, adapt_gamma = False, pr = False, accel=False, new=False):
    """Linear regression using Gradient Descent on the Logistic Regression + regularization term objective.

    Iteratively compute the model weights given data y, a design matrix tx, an initial condition (on w), a batch size, a maximum of iterations and a step size (gamma); with logistic regression loss with regularization term. Additional parameters allow for early stopping of iterations when convergence threshold is reached, an adapting step size, output printing each 100 epochs and use of accelerated gradient descent algorithm.
    Returns weights and loss."""
    # TODO: Chech documentation for use of new.

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
        if np.isinf(loss):
            raise ValueError("Infinite loss, exiting.")
            break
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
