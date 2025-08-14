# optimizer.py
import cvxpy as cp
import numpy as np
import pandas as pd

def optimize_portfolio(returns: pd.DataFrame):
    """
    Optimize a portfolio using mean-variance minimization.
    
    Parameters:
        returns (pd.DataFrame): Daily returns of assets (columns = tickers)
    
    Returns:
        weights (np.ndarray): Optimal weights for each asset
        expected_return (float): Expected portfolio return
        risk (float): Portfolio variance
    """
    mu = returns.mean().values          # Expected returns
    cov = returns.cov().values         # Covariance matrix
    n = len(mu)

    # Decision variable: portfolio weights
    w = cp.Variable(n)

    # Portfolio risk (variance) and expected return
    risk = cp.quad_form(w, cov)
    ret = mu @ w

    # Constraints: weights sum to 1, no shorting (weights >= 0)
    problem = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])
    problem.solve()

    return w.value, ret.value, risk.value
