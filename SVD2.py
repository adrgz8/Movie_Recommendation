"""
PROJECT 1
Methods of Classification and Dimensionality Reduction
---------
submitted by:
    Adrian Rodriguez - 332045
    Pranav Kelkar    - 334722
"""
import argparse
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error

def preprocess_csv(train_filepath, test_filepath, weight=0.4):
    """
    Takes in train and test filepaths and returns
    a train matrix (with NaNs filled with weighted means),
    the original train matrix,
    the test matrix,
    NaN indices of original train matrix (required for SVD2),
    NaN indices of test matrix
    Parameters
    ----------
        train_filepath : str
            Path of the train file
        test_filepath : str
            Path of the test file
        weight : float, OPTIONAL
            NaNs will be replaced with
                weight * mean_of_row + (1 - weight) * mean_of_column
            (default is 0.4)
    """
    train_ratings_original = pd.read_csv(train_filepath)
    test_ratings_original = pd.read_csv(test_filepath)

    train_indices = list(range(0, len(train_ratings_original)))
    test_indices = list(range(
                        len(train_ratings_original),
                        len(train_ratings_original) + len(test_ratings_original)
                    ))

    # Create the complete dataframe first (to make `z_est_svd2` and `Z_test` the same size)
    train_ratings = pd.concat([train_ratings_original, test_ratings_original], ignore_index=True)
    test_ratings = train_ratings.copy()

    # Set NaNs in the alternate DataFrame
    train_ratings.loc[test_indices, 'rating'] = np.nan
    test_ratings.loc[train_indices, 'rating'] = np.nan

    # Change train and test to matrices with users as rows and movies as columns
    train_ratings.drop(columns='timestamp', inplace=True)
    train_df = train_ratings.set_index(['userId', 'movieId'], drop=True).unstack('movieId')
    test_ratings.drop(columns='timestamp', inplace=True)
    test_df = test_ratings.set_index(['userId', 'movieId'], drop=True).unstack('movieId')
    Z_train = train_df.to_numpy()
    Z_test = test_df.to_numpy()

    # Indices for NaN Values
    nan_train = np.where(np.isnan(Z_train))

    Z_wmeans = Z_train.copy()

    col_mean = np.nanmean(Z_wmeans, axis = 0)
    col_mean[np.where(np.isnan(col_mean))] = np.nanmean(col_mean)
    row_mean = np.nanmean(Z_wmeans, axis = 1)
    Z_wmeans[nan_train] = weight * np.take(row_mean, nan_train[0])
    Z_wmeans[nan_train] += (1 - weight) * np.take(col_mean, nan_train[1])

    return Z_wmeans, Z_train, Z_test

def predictions(W, H):
    """
    Outputs z_est_svd2 given W and H
    Parameters
    ----------
        W : numpy.ndarray
            W matrix (must be n x n_features)
        H : numpy.ndarray
            H matrix (must be n_features x p)
    """
    return W.T @ H

def rmse(real, pred):
    """
    Calculates RMSE for a prediction, compared to real values.
    Parameters
    ----------
        real : numpy.ndarray
            Matrix with actual values
        pred : numpy.ndarray
            Matrix with predicted values
    """
    real = real.flatten()
    pred = pred.flatten()
    nan_indices = np.where(np.isnan(real))
    real = np.delete(real, nan_indices)
    pred = np.delete(pred, nan_indices)

    return np.sqrt(mean_squared_error(pred, real))

def svd2(Z, nan_indices=[], num_features=10):
    """
    Performs SVD2 on a given Z matrix, and returns the estimate.
    Parameters
    ----------
        Z : numpy.ndarray
            Matrix with train values
        nan_indices : list
            list of indices where original train values are NaNs
        num_features : int, OPTIONAL
            Number of features to reduce to (default is 10)
    """
    # Step 1: Perform SVD
    svd = TruncatedSVD(n_components=num_features)
    svd.fit(Z)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    svd_W = svd.transform(Z) / svd.singular_values_
    svd_H = Sigma2 @ VT
    z_est_svd2 = svd_W @ svd_H

    # Step 2: Change NaNs to values from z_est_svd2
    Z_svd2 = Z.copy()
    Z_svd2[nan_indices] = z_est_svd2[nan_indices]

    svd = TruncatedSVD(n_components=num_features)
    svd.fit(Z_svd2)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    svd_W = svd.transform(Z) / svd.singular_values_
    svd_H = Sigma2 @ VT
    svd_est = svd_W @ svd_H

    return svd_est

def sgd(Z, Z_test, epochs=50, features=25, lr=0.001, alpha=0.01, epsilon=0.001):
    """
    Performs SGD on a given Z matrix, and returns the estimate.
    Parameters
    ----------
        Z : numpy.ndarray
            Matrix with train values
        Z_test : numpy.ndarray
            Matrix with test values
        epochs : int, OPTIONAL
            Number of iterations for gradient descent (default is 50)
        features : int, OPTIONAL
            Number of features to reduce to (default is 25)
        lr : float, OPTIONAL
            Learning rate of SGD (default is 0.01)
        alpha : float, OPTIONAL
            L2 regularisation constant (default is 0.01)
        epsilon : float, OPTIONAL
            Stopping criteria (default is 0.001)
    """
    m, n = Z.shape
    Z[np.isnan(Z)] = 0

    # Random Start
    W = 0.5 * np.random.rand(features, m)
    H = 0.5 * np.random.rand(features, n)

    min_model = predictions(W, H)
    min_loss = np.inf

    loss_test_errors = []
    users, movies = Z.nonzero()
    for epoch in range(epochs):
        for u, i in zip(users, movies):
            error = Z[u, i] - predictions(W[:, u], H[:, i])
            W[:, u] += lr * (error * 2*H[:, i] - 2*alpha * W[:, u])
            H[:, i] += lr * (error * 2*W[:, u] - 2*alpha * H[:, i])
        loss_test = rmse(Z_test, predictions(W, H))
        loss_test_errors.append(loss_test)

        if loss_test < min_loss: # Save only the best W and H
            min_loss = loss_test
            min_model = predictions(W, H)

        if epoch > 1 and loss_test_errors[-2] - loss_test_errors[-1] < epsilon:
            # Updating Learning Rate when reaching a Plateau
            lr = lr * 0.1
            # Breaking when Learning Rate is close to 0
            if lr < epsilon * 0.0001:
                break

        

    return min_model

# Ignore warning for np.nanmean on columns with all nans
simplefilter("ignore", category=RuntimeWarning)
# Ignore warnings for convergence and deprecation (caused by NMF)
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=FutureWarning)

# Getting the train and test matrix
Z_wmeans, Z_train, Z_test = preprocess_csv('train_ratings.csv', 'test_ratings.csv')

# Estimating with SVD2
Z_est_svd2 = svd2(Z_wmeans, nan_indices=np.where(np.isnan(Z_train)))
test_rmse_svd2 = rmse(Z_test, Z_est_svd2)

# Estimating with SGD
Z_est_sgd = sgd(Z_train, Z_test)
test_rmse_sgd = rmse(Z_test, Z_est_sgd)

# Results
print(f'RMSE using SVD2 is {test_rmse_svd2} and using SGD is {test_rmse_sgd}')
