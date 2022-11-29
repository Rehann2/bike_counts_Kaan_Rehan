import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
import datetime as dt
import requests
import numpy as np


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



def get_weather_data(dates):

    df_weather = pd.DataFrame()
    missing_dates = []

    for date in dates:

        url = f'https://prevision-meteo.ch/climat/horaire/paris-montsouris/{date}'
        html = requests.get(url).content

        try:
            _, df_hour, df_day = pd.read_html(html)
        except:
            missing_dates.append(date)
            continue

        df_hour.insert(0, ("Heure UTC1", "date"), date)
        date = pd.to_datetime(df_hour["Heure UTC1", "date"])
        time = pd.to_datetime(df_hour["Heure UTC1", "Heure UTC1"],
                              format="%H:%M") - dt.datetime(1900, 1, 1)
        df_hour.drop(columns=df_hour.columns[0:2], inplace=True)
        df_hour.insert(0, "date", date+time)
        df_weather = pd.concat([df_weather, df_hour], ignore_index=True)

        return df_weather


def merge_external_data(main_data, external_data):
    data = main_data.join(external_data.set_index("date"), on="date")

    return data



def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    regressor = Ridge()

    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe
