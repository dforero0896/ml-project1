import matplotlib
matplotlib.use("Agg")
from proj1_helpers import *
from implementations import *
from preprocessing import *
import numpy as np
import matplotlib.pyplot as plt

clean = True
dopca = False
remove_cols = True
cols = (4, 5, 6, 12, 26, 27, 28)
max_comp = 30  # For cleaning, and no removing cols

max_iter_gd = 500
gamma_gd = 1e-3
max_iter_sgd = 500
gamma_sgd = 1e-5
max_iter_lrgd = 500
gamma_lrgd = 1e-8
lambda_rlrgd = 50
gamma_rlrgd = 1e-8
max_iter_rlrgd = 500
lambda_rr = 2e-4
w_init = None
batch_size=1
args_gd = {
    'initial_w': w_init,
    'max_iters': max_iter_gd,
    'gamma': gamma_gd,
    'kind': 'mse',
    'adapt_gamma': False,
    'pr': False,
    'accel': False
}
args_sgd = {
    'initial_w':w_init,
    'batch_size':batch_size,
    'max_iters':max_iter_sgd,
    'gamma':gamma_sgd,
    'pr':False,
    'adapt_gamma':False,
    'choose_best':True
}
args_lsq = {
 }
args_rr = {
    'lambda_':lambda_rr
}

args_lrgd = {
    'initial_w': w_init,
    'max_iters': max_iter_lrgd,
    'gamma': gamma_lrgd,
    'adapt_gamma': False,
    'pr': False,
    'accel': False
}
args_rlrgd = {
    'lambda_':lambda_rlrgd,
    'initial_w': w_init,
    'max_iters': max_iter_rlrgd,
    'gamma': gamma_rlrgd,
    'adapt_gamma': False,
    'pr': False,
    'accel': False
}

def cross_validation(y,
                     x,
                     k_indices,
                     k,
                     degree,
                     method,
                     method_args,
                     clean=clean,
                     dopca=dopca,
                     remove_cols=remove_cols):
    """Perform k-fold cross validation on the set using given method.

    Given a RAW dataset of targets y and predictors x, a list k_indices of
    index partitions, index k to use as test, a method and its necessary
    arguments, performs k-fold cross validation. Accepts keyword arguments to preprocessing.preprocess function."""

    # get k'th subgroup in test, others in train
    id_test = k_indices[k]
    id_train = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)
    x_test = x[id_test]
    x_train = x[id_train]
    y_test = y[id_test]
    y_train = y[id_train]
    y_train, x_train, _, _, _, _ = preprocess(x_train,
                                              y_train,
                                              clean=clean,
                                              dopca=dopca,
                                              max_comp=max_comp,
                                              remove_cols=remove_cols,
                                              cols=cols)
    y_test, x_test, _, _, _, _ = preprocess(x_test,
                                            y_test,
                                            clean=clean,
                                            dopca=dopca,
                                            max_comp=max_comp,
                                            remove_cols=remove_cols,
                                            cols=cols)
    # form data with polynomial degree
    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)
    if 'initial_w' in method_args:
        method_args['initial_w'] = np.zeros(tx_test.shape[1])
    # ridge regression
    if method.__name__ == 'logistic_regression' or method.__name__ == 'reg_logistic_regression':
        y_train = minus_one_2_zero(y_train)
        y_test_loss = minus_one_2_zero(y_test)
        loss_func = compute_loss_logistic
    else:
        loss_func = compute_loss
        y_test_loss = y_test
    weight, loss_tr = method(y_train, tx_train, **method_args)
    # calculate the loss for train and test data
    loss_te = loss_func(y_test_loss, tx_test, weight)
    accuracy = accuracy_ratio(predict_labels(weight, tx_test), y_test)
    return loss_tr, loss_te, accuracy


def cross_validation_visualization(results_fname):
    """Visualization the curves of mse_tr and mse_te."""

    results = np.loadtxt(results_fname, usecols=(0, 1, 2, 4, 5))
    lambds = results[:, 0]
    mse_tr = results[:, 1]
    mse_te = results[:, 2]
    std_tr = results[:, 3]
    std_te = results[:, 4]

    best_l_err = lambds[np.argmin(mse_te)]
    print('Best lambda from error: %.2e' % best_l_err)
    plt.xscale('log')
    plt.errorbar(lambds,
                 mse_tr,
                 yerr=std_tr,
                 marker=".",
                 color='b',
                 label='train error')
    plt.errorbar(lambds,
                 mse_te,
                 yerr=std_te,
                 marker=".",
                 color='r',
                 label='test error')
    plt.axvline(best_l_err,
                c='g',
                label='$\lambda^*_{rmse}=%.1e$' % best_l_err,
                ls=':')
    plt.xlabel("lambda", fontsize=15)
    plt.ylabel("mse", fontsize=15)
    plt.title("Cross validation", fontsize=15)
    plt.legend(loc=0, fontsize=10)
    plt.grid(True)
    plt.savefig(results_fname.replace('.dat', '_loss.png'), dpi=200)


def cross_validation_visualization_accuracy(results_fname):
    """Visualization the accuracy curve."""

    results = np.loadtxt(results_fname, usecols=(0, 3, 6))
    lambdas = results[:, 0]
    accuracies = results[:, 1]
    accuracies_std = results[:, 2]
    plt.xscale('log')
    plt.errorbar(lambdas,
                 accuracies,
                 yerr=accuracies_std,
                 lw=2,
                 marker='*',
                 label='Accuracy ratio')
    best_l_acc = lambdas[np.argmax(accuracies)]
    plt.axvline(best_l_acc,
                c='k',
                label='$\lambda^*_{acc}=%.1e$' % best_l_acc,
                ls=':')
    print('Best lambda from accuracy: %.2e' % best_l_acc)
    plt.xlabel("lambda", fontsize=15)
    plt.ylabel("accuracy", fontsize=15)
    plt.title("Cross validation", fontsize=15)
    plt.legend(loc=0, fontsize=10)
    plt.grid(True)
    plt.savefig(results_fname.replace('.dat', '_acc.png'), dpi=200)


def cross_validation_demo(x,
                          y,
                          method,
                          method_args,
                          seed=42,
                          degree=1,
                          k_fold=4,
                          clean=clean,
                          dopca=dopca,
                          remove_cols=remove_cols):
    """Perform cross validation on the raw data, given a method and its arguments.

    Iterates over 10 lambdas 1e-7 to 1e-3 and computes the cross validation for the given method. Plots are displayed if the selected method takes a lambda_ argument. Else just does cross validation once."""
    print('Using method %s' % method.__name__)
    lambdas = np.logspace(-7, 0, 10)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    x_validations_mean = []
    x_validations_std = []
    # cross validation
    if 'lambda_' in method_args:
        for ind, lambda_ in enumerate(lambdas):
            method_args['lambda_'] = lambda_
            print('Using lambda = %.1e' % lambda_)
            x_validation = np.array([
                cross_validation(y,
                                 x,
                                 k_indices,
                                 k,
                                 degree,
                                 method,
                                 method_args,
                                 clean=clean,
                                 dopca=dopca,
                                 remove_cols=remove_cols) for k in range(k_fold)
            ])
            x_validations_mean.append(np.mean(x_validation, axis=0))
            x_validations_std.append(np.std(x_validation, axis=0))
        x_validations_mean = np.array(x_validations_mean)
        x_validations_std = np.array(x_validations_std)
        output = np.c_[lambdas, x_validations_mean, x_validations_std]
    else:
        x_validation = np.array([
            cross_validation(y,
                             x,
                             k_indices,
                             k,
                             degree,
                             method,
                             method_args,
                             clean=clean,
                             dopca=dopca,
                             remove_cols=remove_cols) for k in range(k_fold)
        ])
        x_validations_mean=np.mean(x_validation, axis=0)
        x_validations_std=np.std(x_validation, axis=0)
        output =[ np.concatenate((np.zeros(1),x_validations_mean,x_validations_std), axis = None)]


    outname = '../results/cv_%s_d%i_cl%i_pca%i_rmcols%i.dat' % (
                method.__name__, degree, int(clean), int(dopca), int(remove_cols))
    np.savetxt(outname, output)
    if 'lambda_' in method_args:
        cross_validation_visualization(outname)
        plt.figure()
        cross_validation_visualization_accuracy(outname)


if __name__ == '__main__':
    subsamp = False
    y, x, id_ = load_csv_data('../data/train.csv', sub_sample=subsamp)
    methods = [least_squares, least_squares_GD, least_squares_SGD, ridge_regression, reg_logistic_regression, logistic_regression]
    args = [args_lsq, args_gd, args_sgd, args_rr, args_rlrgd, args_lrgd]
    bools = [False, True]
    for pc in bools:
        for cl in bools:
            for rmcols in bools:
                for i_method, method in enumerate(methods):
                    cross_validation_demo(x, y, method, args[i_method], seed=42,
                        degree=2,
                        k_fold=4,
                        clean=cl,
                        dopca=pc,
                        remove_cols=rmcols)
