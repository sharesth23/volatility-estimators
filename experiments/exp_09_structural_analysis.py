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

# =========================
# 4. Structural break detection
# =========================
break_indices = detect_structural_breaks(pc1_series, penalty=10)

break_dates = pc1_series.index[break_indices[:-1]]  # exclude last index

print("\nDetected structural breaks:")
for d in break_dates:
    print(d.date())


# =========================
# 5. Visualization
# =========================
plt.figure(figsize=(10, 5))
pc1_series.plot(label="Systemic Risk (PC1)")

for d in break_dates:
    plt.axvline(d, color="red", linestyle="--", alpha=0.7)

plt.title("Structural Breaks in Cross-Asset Systemic Risk")
plt.ylabel("PC1 Explained Variance")
plt.legend()
plt.tight_layout()

plt.savefig(
    "paper/figures/fig_structural_breaks.png",
    dpi=300
)
plt.show()

# =========================
# 6. Pre/Post regime comparison
# =========================
if len(break_dates) > 0:
    split_date = break_dates[0]

    pre_regime = pc1_series.loc[:split_date]
    post_regime = pc1_series.loc[split_date:]

    print("\nRegime statistics:")
    print(f"Pre-break mean PC1  : {pre_regime.mean():.3f}")
    print(f"Post-break mean PC1 : {post_regime.mean():.3f}")
