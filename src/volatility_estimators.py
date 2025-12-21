import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH


def rolling_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling realized volatility (standard deviation).
    """
    return returns.rolling(window).std()


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    var = pd.Series(index=returns.index, dtype=float)
    var.iloc[0] = returns.iloc[0] ** 2

    for t in range(1, len(returns)):
        var.iloc[t] = lam * var.iloc[t - 1] + (1 - lam) * returns.iloc[t] ** 2

    return np.sqrt(var)


def garch_volatility(returns: pd.Series) -> pd.Series:
    am = ConstantMean(returns)
    am.volatility = GARCH(1, 1)
    am.distribution = "normal"

    res = am.fit(disp="off")
    vol = pd.Series(
        np.sqrt(res.conditional_volatility),
        index=returns.index
    )
    return vol


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 30
) -> pd.Series:

    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2.0))

    return np.sqrt(
        factor * log_hl.rolling(window).mean()
    )
