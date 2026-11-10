"""Model factory and target transforms."""
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet, HuberRegressor


def _lin(est):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def make_model(spec, seed=0):
    t, p = spec["type"], dict(spec.get("params", {}))
    if t == "ols":
        return _lin(LinearRegression())
    if t == "ridge":
        return _lin(Ridge(**p))
    if t == "lasso":
        return _lin(Lasso(max_iter=50000, **p))
    if t == "enet":
        return _lin(ElasticNet(max_iter=50000, **p))
    if t == "huber":
        return _lin(HuberRegressor(max_iter=5000, **p))
    if t == "svr":
        from sklearn.svm import SVR
        return _lin(SVR(**p))
    if t == "kernel_ridge":
        from sklearn.kernel_ridge import KernelRidge
        return _lin(KernelRidge(**p))
    if t == "pls":
        from sklearn.cross_decomposition import PLSRegression
        return _lin(PLSRegression(**p))
    if t == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(random_state=seed, n_jobs=2, **p))
    if t == "extratrees":
        from sklearn.ensemble import ExtraTreesRegressor
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(random_state=seed, n_jobs=2, **p))
    if t == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=seed, **p)
    if t == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=seed, verbose=-1, n_jobs=2, **p)
    if t == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(random_state=seed, n_jobs=2, **p)
    if t == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=seed, verbose=0, thread_count=2, **p)
    if t == "mlp":
        from sklearn.neural_network import MLPRegressor
        return _lin(MLPRegressor(random_state=seed, max_iter=2000, **p))
    if t == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
        return _lin(GaussianProcessRegressor(kernel=DotProduct() + RBF(length_scale=10.0) + WhiteKernel(1.0), normalize_y=True, random_state=seed))
    raise ValueError(t)


def transform_target(y, how):
    """Fitted on the training part of each fold only."""
    y = np.asarray(y, dtype=float)
    if how in (None, "none"):
        return y
    if how == "rank":       # uniform ranks in (-0.5, 0.5)
        return (rankdata(y) - 0.5) / len(y) - 0.5
    if how == "gauss":      # normal scores of ranks
        return norm.ppf((rankdata(y) - 0.5) / len(y))
    if how.startswith("clip"):
        q = float(how[4:]) / 100
        lo, hi = np.quantile(y, [q, 1 - q])
        return np.clip(y, lo, hi)
    if how == "signlog":
        return np.sign(y) * np.log1p(np.abs(y))
    raise ValueError(how)
