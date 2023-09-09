# Add friedman's H statistic #22383

## Table of Content

- [Issue Link](#issue-link)
- [Implementation Details](#implementation-details)
- [How the Implementation Addresses the Issue](#how-the-implementation-addresses-the-issue)
- [File Changes](#file-changes)
  - [File Modified](#files-modified)
  - [File Added](#files-added)

## Issue Link
[Add friedman's H statistic #22383](https://github.com/scikit-learn/scikit-learn/issues/22383)


## Proposed Solution
Given the marginal and pairwise partial dependence, Friedman's H statistics provide a simple way to determine whether there is substantial interaction between variables in a model. 
It will be convenient for the users if this feature can be implemented into a existing module.

## How the Implementation Addresses the Issue
A new API friedman_h(estimator, X, y, grid_solution) is created in the script `sklearn/inspection/_friedman_h_statistic.py`. The function takes an estimator, feature matrix, and target vector as input, and computes the Friedman's H Statistic and the corresponding 
p-value of the test.  

The script first computes the univariate partial dependence for each feature in the dataset. This is done using the partial_dependence() function in 
scikit-learn, which computes the partial dependence of the estimator on each feature while holding all other features constant. 
The kind parameter is set to 'average' to compute the average partial dependence across all samples in the dataset, 
and the grid_resolution parameter controls the density of the grid used to compute the partial dependence.  

Next, the script computes the bivariate partial dependence for each pair of features in the dataset. This is done using the same partial_dependence() 
function as before, but with the features parameter set to a tuple of two features. The resulting bivariate partial dependence is stored in a dictionary
keyed by the indices of the two features.  

Finally, the script computes the Friedman's H Statistic for each pair of features. The numerator of the statistic is the sum of the squared differences 
between the bivariate partial dependence and the sum of the univariate partial dependences for each of the two features, 
plus the mean of the target variable. The denominator is the sum of the squared differences between the bivariate partial 
dependence and the mean of the target variable. The resulting statistic is stored in a matrix, 
where each element corresponds to the pair of features used to compute the statistic.

## File Changes

Here we list the changes that were made to the design and/or codebase of the chosen system as a result of the implementation.
### Files Modified
`sklearn/inspection/__init__.py`

The friedman_h_statistics function is imported here so the user can import this feature similar to the way importing other feature.

### Files Added

`sklearn/inspection/_friedman_h_statistic.py`  

This is the file for the friedman_h_statistics feature functionality. The logic of the implementation is explained above.

`sklearn/inspection/tests/test_h_stats.py`

This is the unit testing file for the friedman_h_statistics feature.

`sklearn/inspection/tests/user_case_22383.py`

This is the user acceptence file for the friedman_h_statistics. It's using bike sharing retrieved from sklearn.dataset and computing a friedman H statistics matric that analyzing the interaction between each 2 features.
