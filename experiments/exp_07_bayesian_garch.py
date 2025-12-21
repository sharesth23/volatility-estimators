import arviz as az
from data.tickers import ASSETS
from src.data_loader import load_data
from src.bayesian_garch import fit_bayesian_garch

_, returns = load_data(ASSETS)

trace = fit_bayesian_garch(returns["SP500"])
az.plot_posterior(trace, var_names=["omega", "alpha", "beta"])
