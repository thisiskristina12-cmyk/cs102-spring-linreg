import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import GDRegressor

FEATURE_COLUMNS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
TARGET_COLUMN = "apparent_temperature"

def train_model(data: pd.DataFrame):
    if data.empty:
        raise ValueError("данных нет")

    X = data[FEATURE_COLUMNS].values
    y = data[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    reg = GDRegressor(alpha=0.01, n_iter=3000, progress=False)
    reg.fit(X_train_scaled, y_train)

    y_pred = reg.predict(X_test_scaled)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    test_data = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
    test_data["apparent_temperature"] = y_test
    test_data["prediction"] = y_pred
    test_data["absolute_error"] = np.abs(
        test_data["apparent_temperature"] - test_data["prediction"]
    )
    test_data = test_data.sort_values("absolute_error", ascending=False).reset_index(drop=True)

    return reg, {"MAE": mae, "RMSE": rmse, "R2": r2}, test_data, scaler