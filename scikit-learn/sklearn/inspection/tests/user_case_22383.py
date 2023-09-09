import sys
import os
SCRIPT_DIR = os.path.split(os.path.split(os.path.dirname(__file__))[0])[0]
sys.path.append(os.path.dirname(SCRIPT_DIR[0]))

from sklearn.inspection import friedman_h
from sklearn.datasets import fetch_openml

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder

from time import time
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

bikes = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas")
# Make an explicit copy to avoid "SettingWithCopyWarning" from pandas
X, y = bikes.data.copy(), bikes.target

# Because of this rare category, we collapse it into "rain".
X["weather"].replace(to_replace="heavy_rain", value="rain", inplace=True)

# use neural network model
numerical_features = [
    "temp",
    "feel_temp",
    "humidity",
    "windspeed",
]
categorical_features = X.columns.drop(numerical_features)

mlp_preprocessor = ColumnTransformer(
    transformers=[
        ("num", QuantileTransformer(n_quantiles=100), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
mlp_preprocessor

print("Training MLPRegressor...")
tic = time()
mlp_model = make_pipeline(
    mlp_preprocessor,
    MLPRegressor(
        hidden_layer_sizes=(30, 15),
        learning_rate_init=0.01,
        early_stopping=True,
        random_state=0,
    ),
)
mlp_model.fit(X, y)
print(f"done in {time() - tic:.3f}s")
print(f"Training R2 score: {mlp_model.score(X, y):.2f}")

# use only first 500 samples to measure the H_stats to save time since we already have a neural network model
H_stats = friedman_h(mlp_model, X.head(500), y.head(500))

print(H_stats)