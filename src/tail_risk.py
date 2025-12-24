import numpy as np
from scipy.stats import norm
import pandas as pd


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    return np.quantile(returns.dropna(), alpha)


def historical_es(returns: pd.Series, alpha: float = 0.05) -> float:
    var = historical_var(returns, alpha)
    return returns[returns <= var].mean()


def gaussian_var(returns: pd.Series, alpha: float = 0.05) -> float:
    mu, sigma = returns.mean(), returns.std()
    return mu + sigma * norm.ppf(alpha)


def gaussian_es(returns: pd.Series, alpha: float = 0.05) -> float:
    mu, sigma = returns.mean(), returns.std()
    z = norm.ppf(alpha)
    return mu - sigma * norm.pdf(z) / alpha
