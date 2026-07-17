from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from joblib import Parallel, delayed

from .estimator import TROP_TWFE_average
from .cv import adaptive_TROP_cv
from .baselines import DID_TWFE, SC_TWFE, DIFP_TWFE, SDID_TWFE


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
EstimatorFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int], float]

_VALID_OUTCOME_MODELS = {"full", "no_factor", "no_fixed", "noise_only"}
_VALID_NOISE_MODELS = {"ar2", "independent", "none"}
_VALID_ASSIGNMENTS = {"fitted", "random"}


# ---------------------------------------------------------------------------
# Fitting realistic synthetic-data components from a real panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationComponents:
    """
    Realistic synthetic-data generator fitted from a real outcome panel.

    Attributes
    ----------
    F : ndarray of shape (N, T)
        Additive unit/time fixed-effects component.
    M : ndarray of shape (N, T)
        Low-rank factor component (mean-zero across both unit and time margins).
    cov_time : ndarray of shape (T, T)
        AR(2)-implied covariance matrix of the idiosyncratic residual noise.
    propensity : ndarray of shape (N,)
        Fitted per-unit probability of being selected for placebo treatment.
    """

    F: np.ndarray
    M: np.ndarray
    cov_time: np.ndarray
    propensity: np.ndarray

    @property
    def shape(self) -> Tuple[int, int]:
        return self.F.shape


def decompose_panel(
    Y: ArrayLike, rank: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose an outcome panel into additive fixed effects and a low-rank
    factor component via truncated SVD (see the SDID and TROP papers).

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Outcome panel.
    rank : int, default=4
        Rank of the truncated SVD factor decomposition.

    Returns
    -------
    F : ndarray of shape (N, T)
        Additive unit/time fixed-effects component.
    M : ndarray of shape (N, T)
        Low-rank factor component (mean-zero across both margins).
    E : ndarray of shape (N, T)
        Idiosyncratic residual, ``Y - (F + M + E's low-rank fit)``.
    unit_factors : ndarray of shape (N, rank)
        Rescaled left singular vectors, used to fit treatment propensities.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"Y must be a 2D array of shape (N, T). Got Y.ndim={Y.ndim}.")
    N, T = Y.shape
    if not (1 <= rank <= min(N, T)):
        raise ValueError(f"rank must be in [1, min(N, T)]={min(N, T)}. Got rank={rank}.")

    u, s, v = np.linalg.svd(Y, full_matrices=False)
    factor_unit = u[:, :rank]
    factor_time = v[:rank, :]
    L = (factor_unit * s[:rank]) @ factor_time
    E = Y - L
    F = np.add.outer(L.mean(axis=1), L.mean(axis=0)) - L.mean()
    M = L - F
    unit_factors = factor_unit * np.sqrt(N)

    return F, M, E, unit_factors


def _fit_ar2_coefficients(E: np.ndarray) -> np.ndarray:
    """Fit an AR(2) process to panel residuals E via least squares (per-row lags, pooled)."""
    E_ts = E[:, 2:]
    E_lag_1 = E[:, 1:-1]
    E_lag_2 = E[:, :-2]

    a_1 = np.sum(np.diag(E_lag_1 @ E_lag_1.T))
    a_2 = np.sum(np.diag(E_lag_2 @ E_lag_2.T))
    a_3 = np.sum(np.diag(E_lag_1 @ E_lag_2.T))
    matrix_factor = np.array([[a_1, a_3], [a_3, a_2]])

    b_1 = np.sum(np.diag(E_lag_1 @ E_ts.T))
    b_2 = np.sum(np.diag(E_lag_2 @ E_ts.T))

    return np.linalg.inv(matrix_factor).dot(np.array([b_1, b_2]))


def _ar2_correlation_matrix(ar_coef: np.ndarray, T: int) -> np.ndarray:
    result = np.zeros(T)
    result[0] = 1
    result[1] = ar_coef[0] / (1 - ar_coef[1])
    for t in range(2, T):
        result[t] = ar_coef[0] * result[t - 1] + ar_coef[1] * result[t - 2]

    index_matrix = np.abs(np.arange(T)[:, None] - np.arange(T))
    return result[index_matrix].reshape(T, T)


def estimate_ar2_covariance(E: ArrayLike) -> np.ndarray:
    """
    Fit an AR(2) process to panel residuals and return the implied T x T
    time-series covariance matrix, rescaled to match the residuals' overall
    (Frobenius) magnitude.

    Parameters
    ----------
    E : array_like of shape (N, T)
        Idiosyncratic residual panel, e.g. from ``decompose_panel``.

    Returns
    -------
    ndarray of shape (T, T)
    """
    E = np.asarray(E, dtype=float)
    if E.ndim != 2:
        raise ValueError(f"E must be a 2D array of shape (N, T). Got E.ndim={E.ndim}.")
    N, T = E.shape
    if T < 3:
        raise ValueError(f"estimate_ar2_covariance requires T >= 3. Got T={T}.")

    ar_coef = _fit_ar2_coefficients(E)
    cor_matrix = _ar2_correlation_matrix(ar_coef, T)

    scaled_sd = np.linalg.norm(E.T.dot(E) / N, ord="fro") / np.linalg.norm(cor_matrix, ord="fro")
    return cor_matrix * scaled_sd


def estimate_propensity_scores(unit_factors: ArrayLike, treated_flags: ArrayLike) -> np.ndarray:
    """
    Estimate per-unit treatment-assignment propensity via logistic regression
    on unit factor loadings.

    Parameters
    ----------
    unit_factors : array_like of shape (N, rank)
        Unit factor loadings, e.g. from ``decompose_panel``.
    treated_flags : array_like of shape (N,)
        Binary indicator of whether each unit was ever treated.

    Returns
    -------
    ndarray of shape (N,)
        Fitted probability of treatment for each unit.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise ImportError(
            "estimate_propensity_scores requires scikit-learn. "
            "Install it with `pip install scikit-learn` or `pip install trop[notebook]`, "
            "or pass a custom `propensity_fn` to `build_simulation_components`."
        ) from e

    unit_factors = np.asarray(unit_factors, dtype=float)
    treated_flags = np.asarray(treated_flags, dtype=float)
    # C=np.inf disables regularization (equivalent to the deprecated penalty=None).
    model = LogisticRegression(C=np.inf).fit(unit_factors, treated_flags)
    return model.predict_proba(unit_factors)[:, 1]


def build_simulation_components(
    Y: ArrayLike,
    treated_flags: ArrayLike,
    rank: int = 4,
    propensity_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> SimulationComponents:
    """
    Fit realistic synthetic-data components from a real outcome panel, so that
    synthetic panels generated from them mirror the real panel's fixed
    effects, factor structure, time-series correlation, and treatment
    assignment mechanism (see the SDID and TROP papers).

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Real outcome panel.
    treated_flags : array_like of shape (N,)
        Binary indicator of whether each unit was ever treated in the real
        data, used to fit a realistic treatment-assignment propensity model.
    rank : int, default=4
        Rank of the SVD factor decomposition of ``Y``.
    propensity_fn : callable or None, default=None
        Optional ``(unit_factors, treated_flags) -> propensity`` override for
        estimating per-unit treatment probability. Defaults to
        ``estimate_propensity_scores`` (logistic regression; requires
        scikit-learn).

    Returns
    -------
    SimulationComponents
    """
    Y = np.asarray(Y, dtype=float)
    treated_flags = np.asarray(treated_flags, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"Y must be a 2D array of shape (N, T). Got Y.ndim={Y.ndim}.")
    N, T = Y.shape
    if treated_flags.shape != (N,):
        raise ValueError(f"treated_flags must have shape (N,)=({N},). Got {treated_flags.shape}.")

    F, M, E, unit_factors = decompose_panel(Y, rank=rank)
    cov_time = estimate_ar2_covariance(E)
    pi = (propensity_fn or estimate_propensity_scores)(unit_factors, treated_flags)
    pi = np.asarray(pi, dtype=float)
    if pi.shape != (N,):
        raise ValueError(f"propensity_fn must return shape (N,)=({N},). Got {pi.shape}.")

    return SimulationComponents(F=F, M=M, cov_time=cov_time, propensity=pi)


# ---------------------------------------------------------------------------
# Drawing synthetic placebo panels
# ---------------------------------------------------------------------------


def generate_synthetic_panel(
    components: SimulationComponents,
    treated_units: int = 10,
    treated_periods: int = 10,
    outcome_model: str = "full",
    noise_model: str = "ar2",
    assignment: str = "fitted",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Draw one synthetic outcome panel with a known, zero true treatment effect
    (placebo), mirroring the fitted structure in ``components``.

    Parameters
    ----------
    components : SimulationComponents
        Fitted synthetic-data components, e.g. from ``build_simulation_components``.
    treated_units : int, default=10
        Number of placebo-treated units to draw.
    treated_periods : int, default=10
        Number of final periods treated as the placebo post-treatment block.
    outcome_model : {"full", "no_factor", "no_fixed", "noise_only"}, default="full"
        Which fitted outcome components to include: "full" uses fixed effects
        and factor structure; "no_factor"/"no_fixed" ablate one; "noise_only"
        ablates both (useful for sensitivity analysis).
    noise_model : {"ar2", "independent", "none"}, default="ar2"
        Noise covariance structure used to draw residuals: fitted AR(2)
        correlation, its diagonal only (independent), or no noise.
    assignment : {"fitted", "random"}, default="fitted"
        Whether placebo-treated units are drawn from the fitted propensity
        model or uniformly at random.
    rng : numpy.random.Generator or None, default=None
        Random generator to use. If None, a fresh default generator is created.

    Returns
    -------
    Y : ndarray of shape (N, T)
        Synthetic outcome panel.
    W : ndarray of shape (N, T)
        Placebo treatment indicator.
    treated_idx : ndarray of int
        Row indices selected as placebo-treated.
    treated_periods : int
        Echoed back for convenience.
    """
    if outcome_model not in _VALID_OUTCOME_MODELS:
        raise ValueError(f"outcome_model must be one of {sorted(_VALID_OUTCOME_MODELS)}.")
    if noise_model not in _VALID_NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {sorted(_VALID_NOISE_MODELS)}.")
    if assignment not in _VALID_ASSIGNMENTS:
        raise ValueError(f"assignment must be one of {sorted(_VALID_ASSIGNMENTS)}.")

    rng = rng if rng is not None else np.random.default_rng()
    N, T = components.shape

    if treated_periods <= 0 or treated_periods >= T:
        raise ValueError(f"treated_periods must be in [1, T-1]. Got treated_periods={treated_periods}, T={T}.")
    if treated_units <= 0 or treated_units >= N:
        raise ValueError(f"treated_units must be in [1, N-1]. Got treated_units={treated_units}, N={N}.")

    if outcome_model == "no_factor":
        fixed, factor = components.F, np.zeros_like(components.M)
    elif outcome_model == "no_fixed":
        fixed, factor = np.zeros_like(components.F), components.M
    elif outcome_model == "noise_only":
        fixed, factor = np.zeros_like(components.F), np.zeros_like(components.M)
    else:
        fixed, factor = components.F, components.M

    if noise_model == "none":
        noise = np.zeros((N, T))
    elif noise_model == "independent":
        noise = rng.multivariate_normal(
            mean=np.zeros(T), cov=np.diag(np.diag(components.cov_time)), size=N
        )
    else:
        noise = rng.multivariate_normal(mean=np.zeros(T), cov=components.cov_time, size=N)

    Y_sim = fixed + factor + noise

    if assignment == "random":
        treated_idx = rng.choice(N, size=treated_units, replace=False)
    else:
        candidates = rng.binomial(n=1, p=components.propensity)
        selected = np.flatnonzero(candidates)
        if selected.size == 0:
            treated_idx = rng.choice(N, size=1, replace=False)
        elif selected.size > treated_units:
            treated_idx = rng.choice(selected, size=treated_units, replace=False)
        else:
            treated_idx = selected

    treated_idx = np.asarray(treated_idx, dtype=int)
    W_sim = np.zeros((N, T))
    W_sim[treated_idx, -treated_periods:] = 1.0

    return Y_sim, W_sim, treated_idx, treated_periods


# ---------------------------------------------------------------------------
# Benchmarking estimators via repeated placebo simulation
# ---------------------------------------------------------------------------


def _call_did(Y: np.ndarray, W: np.ndarray, treated_units: np.ndarray, treated_periods: int) -> float:
    return DID_TWFE(Y, W)


def _call_sc(Y: np.ndarray, W: np.ndarray, treated_units: np.ndarray, treated_periods: int) -> float:
    return SC_TWFE(Y, W, treated_units, treated_periods=treated_periods)


def _call_sdid(Y: np.ndarray, W: np.ndarray, treated_units: np.ndarray, treated_periods: int) -> float:
    return SDID_TWFE(Y, W, treated_units, treated_periods=treated_periods)


def _call_difp(Y: np.ndarray, W: np.ndarray, treated_units: np.ndarray, treated_periods: int) -> float:
    return DIFP_TWFE(Y, W, treated_units, treated_periods=treated_periods)


def _call_trop(
    Y: np.ndarray,
    W: np.ndarray,
    treated_units: np.ndarray,
    treated_periods: int,
    *,
    lambda_unit: float,
    lambda_time: float,
    lambda_nn: float,
) -> float:
    return TROP_TWFE_average(
        Y,
        W,
        treated_units,
        lambda_unit=lambda_unit,
        lambda_time=lambda_time,
        lambda_nn=lambda_nn,
        treated_periods=treated_periods,
    )


def default_estimator_suite(
    treated_periods: int,
    trop_lambdas: Optional[Tuple[float, float, float]] = None,
    tuning_data: Optional[ArrayLike] = None,
    tune_kwargs: Optional[Mapping] = None,
) -> Dict[str, EstimatorFn]:
    """
    Build the package's default estimator suite for benchmarking: TROP (tuned
    via placebo cross-validation), MC (TROP with ``lambda_unit=lambda_time=0``),
    plain DID, Synthetic Control (SC), SDID, and DIFP.

    Exactly one of ``trop_lambdas`` or ``tuning_data`` must be given, to fix
    the ``(lambda_unit, lambda_time, lambda_nn)`` used for the TROP and MC
    entries: ``trop_lambdas`` supplies them directly, or ``tuning_data`` is a
    panel passed to ``adaptive_TROP_cv`` to select them.

    Parameters
    ----------
    treated_periods : int
        Number of placebo-treated (post) periods each estimator will be
        called with.
    trop_lambdas : tuple of float or None, default=None
        ``(lambda_unit, lambda_time, lambda_nn)`` to use directly.
    tuning_data : array_like of shape (N, T) or None, default=None
        Panel passed to ``adaptive_TROP_cv(tuning_data, treated_periods, ...)``
        to select ``trop_lambdas``.
    tune_kwargs : mapping or None, default=None
        Extra keyword arguments forwarded to ``adaptive_TROP_cv``.

    Returns
    -------
    dict of str to callable
        Each callable accepts ``(Y, W, treated_units, treated_periods)``.
    """
    if (trop_lambdas is None) == (tuning_data is None):
        raise ValueError("Provide exactly one of trop_lambdas or tuning_data.")

    if trop_lambdas is None:
        trop_lambdas = adaptive_TROP_cv(tuning_data, treated_periods, **dict(tune_kwargs or {}))

    lambda_unit, lambda_time, lambda_nn = (float(v) for v in trop_lambdas)

    return {
        "TROP": partial(_call_trop, lambda_unit=lambda_unit, lambda_time=lambda_time, lambda_nn=lambda_nn),
        "MC": partial(_call_trop, lambda_unit=0.0, lambda_time=0.0, lambda_nn=lambda_nn),
        "DID": _call_did,
        "SC": _call_sc,
        "SDID": _call_sdid,
        "DIFP": _call_difp,
    }


def _run_one_trial(
    components: SimulationComponents,
    estimators: Mapping[str, EstimatorFn],
    treated_units: int,
    treated_periods: int,
    outcome_model: str,
    noise_model: str,
    assignment: str,
    seed: np.random.SeedSequence,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    Y_sim, W_sim, treated_idx, tp = generate_synthetic_panel(
        components,
        treated_units=treated_units,
        treated_periods=treated_periods,
        outcome_model=outcome_model,
        noise_model=noise_model,
        assignment=assignment,
        rng=rng,
    )
    return {name: float(fn(Y_sim, W_sim, treated_idx, tp)) for name, fn in estimators.items()}


def evaluate_estimators(
    components: SimulationComponents,
    estimators: Mapping[str, EstimatorFn],
    num_experiments: int = 200,
    treated_units: int = 10,
    treated_periods: int = 10,
    outcome_model: str = "full",
    noise_model: str = "ar2",
    assignment: str = "fitted",
    n_jobs: int = -1,
    prefer: str = "threads",
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark candidate estimators via repeated placebo simulation.

    Draws ``num_experiments`` independent synthetic panels from ``components``
    (each with a known, zero true treatment effect by construction), evaluates
    every estimator in ``estimators`` on each draw, and reports the RMSE and
    bias of each estimator's placebo estimates against the known zero effect.
    The estimator with the lowest RMSE is the most reliable choice for a panel
    with the fitted structure.

    Parameters
    ----------
    components : SimulationComponents
        Fitted synthetic-data components, e.g. from ``build_simulation_components``.
    estimators : mapping of str to callable
        Candidate estimators to benchmark. Each callable must accept
        ``(Y, W, treated_units, treated_periods)`` positionally and return a
        scalar treatment-effect estimate.
    num_experiments : int, default=200
        Number of independent synthetic placebo panels to draw.
    treated_units : int, default=10
        Number of placebo-treated units per synthetic panel.
    treated_periods : int, default=10
        Number of placebo-treated (post) periods per synthetic panel.
    outcome_model : {"full", "no_factor", "no_fixed", "noise_only"}, default="full"
        Which components of the fitted outcome model to include; useful for
        sensitivity analysis.
    noise_model : {"ar2", "independent", "none"}, default="ar2"
        Noise covariance structure used to draw residuals.
    assignment : {"fitted", "random"}, default="fitted"
        Whether placebo-treated units are drawn from the fitted propensity
        model or uniformly at random.
    n_jobs : int, default=-1
        Number of parallel jobs. ``-1`` uses all available cores.
    prefer : {"threads", "processes"}, default="threads"
        Joblib backend preference.
    random_state : int or None, default=None
        Seed for reproducible synthetic-panel draws.

    Returns
    -------
    dict
        ``{estimator_name: {"rmse": float, "bias": float, "n_experiments": int}}``.

    Raises
    ------
    ValueError
        If ``estimators`` is empty or ``num_experiments`` is not positive.
    """
    if not estimators:
        raise ValueError("estimators must be a non-empty mapping of name -> callable.")
    if num_experiments <= 0:
        raise ValueError("num_experiments must be a positive integer.")

    seeds = np.random.SeedSequence(random_state).spawn(num_experiments)

    trial_results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_run_one_trial)(
            components,
            estimators,
            treated_units,
            treated_periods,
            outcome_model,
            noise_model,
            assignment,
            seed,
        )
        for seed in seeds
    )

    summary: Dict[str, Dict[str, float]] = {}
    for name in estimators:
        vals = np.array([r[name] for r in trial_results], dtype=float)
        summary[name] = {
            "rmse": float(np.sqrt(np.mean(vals**2))),
            "bias": float(np.mean(vals)),
            "n_experiments": int(vals.size),
        }
    return summary


def assess_estimators(
    Y: ArrayLike,
    treated_flags: ArrayLike,
    treated_periods: int,
    treated_units: int = 10,
    rank: int = 4,
    estimators: Optional[Mapping[str, EstimatorFn]] = None,
    include_default_estimators: bool = True,
    trop_lambdas: Optional[Tuple[float, float, float]] = None,
    tune_trop_kwargs: Optional[Mapping] = None,
    outcome_model: str = "full",
    noise_model: str = "ar2",
    assignment: str = "fitted",
    num_experiments: int = 200,
    propensity_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    n_jobs: int = -1,
    prefer: str = "threads",
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    End-to-end estimator assessment for a user-supplied panel dataset.

    Fits realistic synthetic-data components from ``Y``/``treated_flags``
    (see ``build_simulation_components``), then benchmarks candidate
    estimators via repeated placebo simulation (see ``evaluate_estimators``),
    returning per-estimator RMSE and bias against the known zero effect.
    Lower RMSE indicates a more reliable estimator for a panel with this
    dataset's structure.

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Real outcome panel.
    treated_flags : array_like of shape (N,)
        Binary indicator of whether each unit was ever treated in the real
        data.
    treated_periods : int
        Number of placebo-treated (post) periods to simulate.
    treated_units : int, default=10
        Number of placebo-treated units to simulate.
    rank : int, default=4
        Rank of the SVD factor decomposition used to fit synthetic-data
        components.
    estimators : mapping of str to callable or None, default=None
        Additional/custom estimators to benchmark alongside (or instead of)
        the built-in suite. Each callable must accept
        ``(Y, W, treated_units, treated_periods)`` positionally.
    include_default_estimators : bool, default=True
        Whether to include the package's built-in suite (TROP, MC, DID, SC,
        SDID, DIFP; see ``default_estimator_suite``).
    trop_lambdas : tuple of float or None, default=None
        ``(lambda_unit, lambda_time, lambda_nn)`` to use for the built-in
        TROP/MC estimators. If None, they are selected via
        ``adaptive_TROP_cv`` on ``Y``.
    tune_trop_kwargs : mapping or None, default=None
        Extra keyword arguments forwarded to ``adaptive_TROP_cv`` when tuning
        ``trop_lambdas``.
    outcome_model, noise_model, assignment : str
        Forwarded to ``evaluate_estimators`` / ``generate_synthetic_panel``.
    num_experiments : int, default=200
        Number of independent synthetic placebo panels to draw.
    propensity_fn : callable or None, default=None
        Forwarded to ``build_simulation_components``.
    n_jobs, prefer, random_state
        Forwarded to ``evaluate_estimators``.

    Returns
    -------
    dict
        ``{estimator_name: {"rmse": float, "bias": float, "n_experiments": int}}``.
    """
    components = build_simulation_components(Y, treated_flags, rank=rank, propensity_fn=propensity_fn)

    all_estimators: Dict[str, EstimatorFn] = {}
    if include_default_estimators:
        all_estimators.update(
            default_estimator_suite(
                treated_periods,
                trop_lambdas=trop_lambdas,
                tuning_data=None if trop_lambdas is not None else np.asarray(Y, dtype=float),
                tune_kwargs=tune_trop_kwargs,
            )
        )
    if estimators:
        all_estimators.update(estimators)

    if not all_estimators:
        raise ValueError("No estimators to evaluate: set include_default_estimators=True or pass estimators.")

    return evaluate_estimators(
        components,
        all_estimators,
        num_experiments=num_experiments,
        treated_units=treated_units,
        treated_periods=treated_periods,
        outcome_model=outcome_model,
        noise_model=noise_model,
        assignment=assignment,
        n_jobs=n_jobs,
        prefer=prefer,
        random_state=random_state,
    )


def summarize(results: Mapping[str, Mapping[str, float]]):
    """
    Render ``evaluate_estimators``/``assess_estimators`` results as a pandas
    DataFrame sorted by RMSE (best estimator first). Requires pandas.

    Parameters
    ----------
    results : mapping
        Output of ``evaluate_estimators`` or ``assess_estimators``.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("summarize requires pandas. Install it with `pip install pandas`.") from e

    return pd.DataFrame.from_dict(results, orient="index").sort_values("rmse")
