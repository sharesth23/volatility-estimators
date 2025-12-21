import pandas as pd
import statsmodels.api as sm


def lead_lag_regression(vol_series, corr_series, lags=5):
    df = pd.concat([vol_series, corr_series], axis=1)
    df.columns = ["vol", "corr"]

    for i in range(1, lags + 1):
        df[f"vol_lag_{i}"] = df["vol"].shift(i)

    df = df.dropna()

    X = sm.add_constant(df[[f"vol_lag_{i}" for i in range(1, lags + 1)]])
    y = df["corr"]

    model = sm.OLS(y, X).fit()
    return model
