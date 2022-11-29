from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures



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
    file_path = "external_data1.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X_comb = X.join(df_ext.set_index("date"), on="date", rsuffix="right") 
    X_comb.fillna(method="ffill", inplace=True)
    
    return X_comb.drop(["Unnamed: 0", "date_only"], axis=1)


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['year', 'month', 'day', 'weekday', 'hour']
    scaler = StandardScaler()

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name", "wind_dir"]
    numerical_cols = ['site_id', 'latitude', 'longitude', 'Temperature (C)', 'wind_speed',
                    'Humidity', 'Visibility', 'Precipitation', 'pressure1', 'sunshine_time',
                    'suntime', 'new_cases', 'holidays2']

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
            #("scaler", scaler, numerical_cols)
        ]
    )

    regressor = Ridge()

    pipe = make_pipeline(FunctionTransformer(
        _merge_external_data, validate=False), date_encoder, preprocessor, regressor)
    
    return pipe
