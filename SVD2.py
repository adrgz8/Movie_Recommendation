"""
PROJECT 1
Methods of Classification and Dimensionality Reduction
---------
submitted by:
    Adrian Rodriguez - 332045
    Pranav Kelkar    - 334722
"""
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

    # Create the complete dataframe first (to make `Z_est` and `Z_test` the same size)
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
    Outputs Z_est given W and H
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
    Z_est = svd_W @ svd_H

    # Step 2: Change NaNs to values from Z_est
    Z_svd2 = Z.copy()
    Z_svd2[nan_indices] = Z_est[nan_indices]

    svd = TruncatedSVD(n_components=num_features)
    svd.fit(Z_svd2)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    svd_W = svd.transform(Z) / svd.singular_values_
    svd_H = Sigma2 @ VT
    svd_est = svd_W @ svd_H

    return svd_est