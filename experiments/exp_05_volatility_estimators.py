"""
Experiment 05
-------------
Volatility estimator comparison.

Reproduces Figure 5 in the research paper.
"""

import matplotlib.pyplot as plt

from data.tickers import ASSETS
from src.data_loader import load_data
from src.volatility_estimators import (
    rolling_volatility,
    ewma_volatility,
    garch_volatility
)

# Load data
prices, returns = load_data(ASSETS)

asset = "SP500"
r = returns[asset]

# Estimate volatility
vol_rolling = rolling_volatility(r)
vol_ewma = ewma_volatility(r)
vol_garch = garch_volatility(r)

# Plot comparison
plt.figure(figsize=(10, 5))
vol_rolling.plot(label="Rolling Volatility", alpha=0.7)
vol_ewma.plot(label="EWMA Volatility", alpha=0.7)
vol_garch.plot(label="GARCH(1,1) Volatility", alpha=0.7)

plt.legend()
plt.title(f"Volatility Estimator Comparison — {asset}")
plt.tight_layout()

plt.savefig(
    "paper/figures/fig_volatility_estimators.png",
    dpi=300
)
plt.show()
