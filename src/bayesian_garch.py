import pymc as pm
import numpy as np


def fit_bayesian_garch(returns):
    r = returns.values

    with pm.Model() as model:
        omega = pm.Exponential("omega", 1.0)
        alpha = pm.Beta("alpha", 2, 2)
        beta = pm.Beta("beta", 2, 2)

        sigma2 = pm.GARCH(
            "sigma2",
            omega=omega,
            alpha=alpha,
            beta=beta,
            shape=len(r)
        )

        pm.Normal("obs", mu=0, sigma=pm.math.sqrt(sigma2), observed=r)

        trace = pm.sample(
            1000,
            tune=1000,
            chains=2,
            target_accept=0.9,
            progressbar=False
        )

    return trace
