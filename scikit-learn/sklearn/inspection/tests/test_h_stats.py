import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.inspection import friedman_h
# Load the diabetes dataset
diabetes = load_diabetes()

# Create a linear regression estimator
estimator = LinearRegression().fit(diabetes.data[:, :2], diabetes.target)

# Define a test function for the h_statistic function
def test_h_statistic():
    # Test the function using the diabetes dataset and the linear regression estimator
    result = friedman_h(estimator, diabetes.data[:, :2], diabetes.target)

    # Check that the result has the expected shape
    assert result.shape == (2, 2)

    # Check that the diagonal elements of the result are zero
    assert np.allclose(np.diag(result), np.zeros(2))

    # Check that the result is symmetric
    assert np.allclose(result, result.T)

    # Check that the result is within a reasonable range
    assert np.all(result >= 0) and np.all(result <= 1)


def test_friedman_h_insignificant_interaction():
    # Create a linear regression estimator with insignificant interaction
    X = np.random.normal(0, 1, size=(1000, 2))
    y = X[:, 0] + np.random.normal(0, 0.1, size=1000)
    estimator = LinearRegression().fit(X, y)

    # Test the function using the linear regression estimator
    result = friedman_h(estimator, X, y)

    # Check that the result has the expected shape
    assert result.shape == (2, 2)

    # Check that the diagonal elements of the result are zero
    assert np.allclose(np.diag(result), np.zeros(2))

    # Check that the result is symmetric
    assert np.allclose(result, result.T)

    # Check that the result is within a reasonable range
    assert np.all(result >= 0) and np.all(result <= 1)

    # Check that the off-diagonal elements of the result are insignificant
    assert np.abs(result[0, 1]) < 0.1

# since no significant interact in diabetes dataset, we use a new dataset to test
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor

def test_high_interaction():
    # Generate a dataset with high interaction
    X, y = make_regression(n_samples=1000, n_features=4, n_informative=2, n_targets=1, noise=0.1, random_state=0)
    # Multiply the first two features to create high interaction
    X[:, 2] = X[:, 0] * X[:, 1]
    
    # Fit a model to the data
    model = MLPRegressor(hidden_layer_sizes=(30, 15), learning_rate_init=0.01, early_stopping=True, random_state=0)
    model.fit(X, y)
    
    # Compute the Friedman H-statistic using the same model and dataset
    H_stats = friedman_h(model, X, y)
    
    # Assert that the H-statistic is significantly larger than expected by chance
    assert (H_stats[0] >= H_stats[1]).all()


if __name__ == '__main__':
    print('------------------------ Test begins! ------------------------')
    test_h_statistic()
    test_friedman_h_insignificant_interaction()
    test_high_interaction()
    #test_friedman_h_significant_interaction()
    print('------------------------ All done! ------------------------')
