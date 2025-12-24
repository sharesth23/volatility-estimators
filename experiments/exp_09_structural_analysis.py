import pandas as pd
import matplotlib.pyplot as plt

from data.tickers import ASSETS
from src.data_loader import load_data
from src.correlation import rolling_correlation
from src.systemic_metrics import eigenvalue_concentration
from src.structural_analysis import detect_structural_breaks

# =========================
# 1. Load data
# =========================
prices, returns = load_data(ASSETS)

# =========================
# 2. Rolling correlation
# =========================
rolling_corrs = rolling_correlation(returns, window=60)

# =========================
# 3. Systemic risk series (PC1)
# =========================
pc1_series = pd.Series(
    {
        date: eigenvalue_concentration(corr)
        for date, corr in rolling_corrs.items()
    }
)

pc1_series.name = "Systemic_Risk_PC1"