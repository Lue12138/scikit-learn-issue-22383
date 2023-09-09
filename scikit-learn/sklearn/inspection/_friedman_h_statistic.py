import numpy as np
import itertools
from ._partial_dependence import partial_dependence

def friedman_h(estimator, x, y, grid_resolution = 50):
    """Compute Friedman's H Statistic.
    Calculation of the H Statistic which determines how strongly does two features
    interact with each other.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator object implementing predict, predict_proba, or decision_function.
        Multioutput-multiclass classifiers are not supported.
    x : array-like of shape (sample_num,feature_n)
        The input data.
    y : array-like of shape (sample_num,)
        The target values.
    grid_resolution: int
        Number of grid points to use for plotting. The default value is 50.


    Returns
    -------
    friedman_h : ndarray of shape (feature_num,feature_n)
        The H statistic for the dataset

    References
    --------
    1. https://www.firmai.org/bit/interaction.html#theory-friedmans-h-statistic
    """
    univariate_partial_dependence = {}
    for i in range(x.shape[1]):
        univariate_partial_dependence[i] = partial_dependence(
            estimator, x, features=[i], kind='average', grid_resolution=grid_resolution)['average']

    bivariate_partial_dependence = {}
    for i, j in itertools.combinations(range(x.shape[1]), 2):
        bivariate_partial_dependence[(i, j)] = partial_dependence(
            estimator, x, features=[i, j], kind='average', grid_resolution=grid_resolution)['average']

    friedman_h = np.zeros((x.shape[1], x.shape[1]))
    for i, j in itertools.combinations(range(x.shape[1]), 2):
        numerator = ((bivariate_partial_dependence[(i, j)] - univariate_partial_dependence[i].reshape(1, -1, 1) - univariate_partial_dependence[j].reshape(1, 1, -1) + y.mean()) ** 2).sum()
        denominator = ((bivariate_partial_dependence[(i, j)] - y.mean()) ** 2).sum()
        friedman_h[i, j] = numerator / denominator

    return friedman_h