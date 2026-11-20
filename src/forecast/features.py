"""Feature groups and the two preprocessors (previously notebook cells 51-56)."""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# 1. Historical lag feature engineering

categorical_features = ["Product", "Day_of_Week", "Month", "Quarter"]
revenue_features = ["Revenue_Lag_364", "Revenue_Same_Date_Last_Year"]
other_numerical_features = ["Day_of_Month", "Year", "Week_of_Year",
                            "Quantity_Lag_364", "Quantity_Same_Date_Last_Year"]
binary_features = ["Is_Weekend", "Is_Month_Start", "Is_Month_End", "Is_Publication_Day", "COVID_Intervention"]
future_numerical_features = ["Day_of_Month", "Year", "Week_of_Year"]

# history window sees every engineered feature; the forecast window only calendar facts known in advance
history_input_features = revenue_features + other_numerical_features + categorical_features + binary_features
future_features = categorical_features + future_numerical_features + binary_features


def make_preprocessors():
    """Fresh, unfitted (history_preprocessor, future_preprocessor). Always fit on training rows only."""
    history = ColumnTransformer(
        transformers=[
            ("revenue", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("log_transform", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                  ("scaler", StandardScaler())]), revenue_features),
            ("numerical", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler())]), other_numerical_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
            ("binary", "passthrough", binary_features),
        ], remainder="drop")
    future = ColumnTransformer(
        transformers=[
            ("numerical", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler())]), future_numerical_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
            ("binary", "passthrough", binary_features),
        ], remainder="drop")
    return history, future