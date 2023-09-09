# Friedman's H Statistic and Partial Dependence Plots for Regression Estimators

- [Overview](#overview) 
- [Mathematical Definition](#mathematical-definition)
- [Example of Use](#example-of-use)
- [Relevant Links](#relevant-links)

## Overview
This user guide will give a description of what Friedman's H Statistic and how Scikit-Learn provide support for computing Friedman's H Statistic 
and utilising it to produce partial dependence charts for regression estimators. 

## Mathematical Definition
A prediction from a machine learning model that is based on two features can be broken down into four terms: a constant term,
a term for the first feature, a term for the second feature, and a term for the interaction between the two features.   
After weighing the impacts of each feature separately, the interaction between two features refers to the change in
prediction that results from changing the features.  


The H-statistic that Friedman and Popescu proposed mathematically for the interaction between feature j and feature k is:

$$
H^2_{jk} = \frac{\sum_{i=1}^n\left[PD_{jk}(x_{j}^{(i)},x_k^{(i)})-PD_j(x_j^{(i)}) - PD_k(x_{k}^{(i)})\right]^2}{\sum_{i=1}^n{PD}^2_{jk}(x_j^{(i)},x_k^{(i)})}
$$

But in Scikit Learn,  the partial dependence on the mean is acting different and should not be taken into account. 
To make the statistic independent of the scale of the target variable, the mean must be added to the numerator. 
Without the adding the mean, the statistic could be sensitive to the target variable's scale selection, 
making it challenging to assess the effectiveness of various regression models. 
The statistic is centred at 0 by adding the mean, ensuring that it is independent of the scale selection for the target variable.
So in the API, the Friedman's H statistic of a given estimator, feature matrix, and target vector is calculated by
```
numerator = ((bivariate_partial_dependence[(i, j)] - univariate_partial_dependence[i].reshape(1, -1, 1) 
             - univariate_partial_dependence[j].reshape(1, 1, -1) + y.mean()) ** 2).sum()
denominator = ((bivariate_partial_dependence[(i, j)] - y.mean()) ** 2).sum()
friedman_h[i, j] = numerator / denominator
```

## Example of Use

```
import numpy as np
from sklearn.inspection import friedman_h
from sklearn.ensemble import GradientBoostingClassifier

X = np.array([[0, 0, 2], [1, 0, 0]])
y = np.array([0, 1])

gb = GradientBoostingClassifier(random_state=0).fit(X, y)
H_stats = friedman_h(gb, X, y)
```

## Relevant Links
https://christophm.github.io/interpretable-ml-book/interaction.html
