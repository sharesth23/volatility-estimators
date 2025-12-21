from data.tickers import ASSETS
from src.data_loader import load_data
from src.volatility_estimators import garch_volatility
from src.correlation import rolling_correlation
from src.systemic_metrics import average_correlation
from src.lead_lag import lead_lag_regression

_, returns = load_data(ASSETS)

vol = garch_volatility(returns["SP500"])
rolling_corrs = rolling_correlation(returns)

avg_corr = {
    d: average_correlation(c)
    for d, c in rolling_corrs.items()
}

model = lead_lag_regression(vol, avg_corr)
print(model.summary())
