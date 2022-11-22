from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import xgboost as xgb



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
    file_path = Path(__file__).parent / "custom_external_data.csv"
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
    params = {'objective': 'reg:squarederror', 'base_score': 0.5, 
    'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1,
    'colsample_bytree': 1.0, 'eval_metric': 'rmse', 'gamma': 0.3, 'learning_rate': 0.300000012,
    'max_bin': 256, 'max_cat_threshold': 64, 'max_cat_to_onehot': 4, 'max_delta_step': 0,
    'max_depth': 7, 'max_leaves': 0, 'min_child_weight': 4,
    'n_estimators': 500, 'num_parallel_tree': 1, 'predictor': 'auto', 'reg_alpha': 0,
    'reg_lambda': 1, 'sampling_method': 'uniform', 'scale_pos_weight': 1, 'subsample': 0.9,
    'tree_method': 'exact', 'validate_parameters': 1, 'eta': 0.3}
    

    Boost = XGBRegressor(**params)

    pipe =make_pipeline(date_encoder, preprocessor, Boost)
    
    return pipe

