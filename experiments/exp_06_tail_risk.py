import matplotlib.pyplot as plt
from data.tickers import ASSETS
from src.data_loader import load_data
from src.tail_risk import historical_var, historical_es

_, returns = load_data(ASSETS)

asset = "SP500"
r = returns[asset]

var_95 = historical_var(r)
es_95 = historical_es(r)

plt.figure(figsize=(9, 5))
plt.hist(r, bins=100, alpha=0.7)
plt.axvline(var_95, color="red", label="VaR 5%")
plt.axvline(es_95, color="black", label="ES 5%")
plt.legend()
plt.title("Tail Risk Estimation (VaR / ES)")
plt.savefig("paper/figures/fig_tail_risk.png", dpi=300)
plt.show()
