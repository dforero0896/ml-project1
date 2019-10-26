#!/usr/bin/env python
from proj1_helpers import *
from preprocessing import *
from implementations import *
subsamp = False

y, x, id_ = load_csv_data('../data/train.csv', sub_sample=subsamp)
y_out_test, x_out_test, id_out_test = load_csv_data('../data/test.csv', sub_sample=subsamp)
clean = True # Clean de data
dopca = False # Do PCA
remove_cols = False # Remove columns cols
stdafter=False #Re-standardize after feature expansion
cols = (4, 5, 6, 12, 26, 27, 28) # Columns to remove (NaN fraction > 0.7)
max_comp = 30 # Principal components kept when doing PCA (use max_comp < 30 for dimensionality reduction)
y_train, x_train, x_train_mean, x_train_var, transform_train, eigenvals_train = preprocess(
    x,
    y,
    clean=clean,
    dopca=dopca,
    max_comp=max_comp,
    remove_cols=remove_cols,
    cols=cols)
y_test, x_test, x_test_mean, x_test_var, transform_test, eigenvals_test = preprocess(
    x_out_test,
    y_out_test,
    clean=clean,
    dopca=dopca,
    max_comp=max_comp,
    remove_cols=remove_cols,
    cols=cols)
print(x_test.shape, x_train.shape)
degree = 10
# Build data matrix with feature expansion
tx_train = build_poly(x_train, degree)
tx_test = build_poly(x_test, degree)
if stdafter:
    tx_train[:,1:], _, _ = standardize_features(tx_train[:,1:])
    tx_test[:,1:], _, _ = standardize_features(tx_test[:,1:])
lambda_rr = 1e-4
w_rr, loss_rr = ridge_regression(y_train, tx_train, lambda_rr)
rr_prediction = predict_labels(w_rr, tx_test)
outname = '../results/rr_pred_deg%i_cl%i_pc%i_rmcol%i_stdafter%i.csv'%(degree, clean, dopca, remove_cols, stdafter)
create_csv_submission(id_out_test, predict_labels(w_rr, tx_test) , outname)
print('Done. Predictions for test data saved in file\n%s'%outname)