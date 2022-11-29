from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext.sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]
    scaler = StandardScaler()

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name", "wind_dir"]
    numerical_cols = ['site_id', 'latitude', 'longitude', 'Temperature (C)', 'wind_speed',
                      'Humidity', 'Visibility', 'pressure1', "Precipitation"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
            ("scaler", scaler, numerical_cols)
        ]
    )
    params = {'base_score': 0.5, 'booster': 'gbtree', 'callbacks': None,
              'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 1.0,
              'early_stopping_rounds': None, 'enable_categorical': False, 'eta': 0.3,
              'eval_metric': 'rmse', 'feature_types': None, 'gamma': 0.3, 'gpu_id': -1,
              'grow_policy': 'depthwise', 'importance_type': None,
              'learning_rate': 0.2500000012, 'max_bin': 286,
              'max_cat_threshold': 64, 'max_cat_to_onehot': 4, 'max_delta_step': 0,
              'max_depth': 10, 'max_leaves': 0, 'min_child_weight': 4,
              'monotone_constraints': '()', 'n_estimators': 1500, 'n_jobs': 0,
              'num_parallel_tree': 1, 'predictor': 'auto',  'tree_method': 'gpu_hist'}

    Boost = XGBRegressor(**params)

    pipe = make_pipeline(FunctionTransformer(
        _merge_external_data, validate=False), date_encoder, preprocessor, Boost)

    return pipe
