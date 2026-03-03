from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union, List

import numpy as np
from joblib import Parallel, delayed

from .estimator import TROP_TWFE_average


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _validate_panel(Y: np.ndarray, treated_periods: int) -> Tuple[int, int]:
    """Validate panel dimensions and treated_periods. Returns (N, T)."""
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array of shape (N, T).")
    N, T = Y.shape
    if treated_periods <= 0 or treated_periods >= T:
        raise ValueError(
            f"treated_periods must be in [1, T-1]. Got treated_periods={treated_periods}, T={T}."
        )
    if N < 2:
        raise ValueError(f"Y must have at least 2 units (N>=2). Got N={N}.")
    return int(N), int(T)


def _as_list(grid: Iterable[float], *, name: str = "lambda_grid") -> List[float]:
    """Convert an iterable of grid values to a non-empty list of floats."""
    grid_list = list(grid)
    if len(grid_list) == 0:
        raise ValueError(f"{name} must be non-empty.")
    return [float(x) for x in grid_list]


def _validate_and_normalize_cv_sampling(
    N: int,
    cv_sampling_method: str,
    *,
    n_treated_units: Optional[int],
    n_trials: Optional[int],
    K: Optional[int],
) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    """
    Validate placebo-set sampling arguments and return normalized (method, n_treated_units, n_trials, K).

    Rules
    -----
    - resample: requires n_treated_units and n_trials; enforces 1 <= n_treated_units <= N-1, n_trials > 0
    - kfold   : requires K; enforces 2 <= K <= N (K>N would create empty folds)
    """
    method = str(cv_sampling_method).lower().strip()
    if method not in {"resample", "kfold"}:
        raise ValueError("cv_sampling_method must be one of {'resample', 'kfold'}.")

    def _as_int(name: str, x: Optional[int]) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, bool):
            raise TypeError(f"{name} must be an integer, not bool.")
        try:
            xi = int(x)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{name} must be an integer.") from e
        if xi != x:
            raise TypeError(f"{name} must be an integer.")
        return xi

    n_treated_units_i = _as_int("n_treated_units", n_treated_units)
    n_trials_i = _as_int("n_trials", n_trials)
    K_i = _as_int("K", K)

    if method == "resample":
        if n_treated_units_i is None or n_trials_i is None:
            raise ValueError("resample requires both n_treated_units and n_trials.")
        if n_trials_i <= 0:
            raise ValueError(f"n_trials must be positive. Got n_trials={n_trials_i}.")
        if n_treated_units_i <= 0 or n_treated_units_i >= N:
            raise ValueError(
                f"n_treated_units must be in [1, N-1]. Got n_treated_units={n_treated_units_i}, N={N}."
            )
        return method, n_treated_units_i, n_trials_i, None

    if K_i is None:
        raise ValueError("kfold requires K.")
    if K_i < 2:
        raise ValueError(f"K must be at least 2 for kfold. Got K={K_i}.")
    if K_i > N:
        raise ValueError(f"K cannot exceed N for kfold (would create empty folds). Got K={K_i}, N={N}.")
    return method, None, None, K_i


def _generate_placebo_sets(
    N: int,
    cv_sampling_method: str = "resample",
    *,
    n_treated_units: Optional[int] = None,
    n_trials: Optional[int] = None,
    K: Optional[int] = None,
    random_state: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate placebo treatment sets for cross-validation.

    - "resample": random placebo assignment repeated `n_trials` times
    - "kfold": K-fold cross-validation (each fold treated once)
    """
    method, n_treated_units_i, n_trials_i, K_i = _validate_and_normalize_cv_sampling(
        N, cv_sampling_method, n_treated_units=n_treated_units, n_trials=n_trials, K=K
    )

    rng = np.random.default_rng(random_state)
    units = np.arange(N)

    if method == "resample":
        assert n_treated_units_i is not None and n_trials_i is not None
        return [
            rng.choice(units, size=n_treated_units_i, replace=False).astype(int, copy=False)
            for _ in range(n_trials_i)
        ]

    assert K_i is not None
    shuffled = rng.permutation(units)
    folds = np.array_split(shuffled, K_i)
    return [fold.astype(int, copy=False) for fold in folds]


def _simulate_ate(
    Y: np.ndarray,
    treated_units: np.ndarray,
    treated_periods: int,
    lambda_unit: float,
    lambda_time: float,
    lambda_nn: float,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> float:
    """Simulate a single placebo ATE given indices of placebo-treated units."""
    W = np.zeros_like(Y, dtype=float)
    W[treated_units, -treated_periods:] = 1.0

    return TROP_TWFE_average(
        Y=Y,
        W=W,
        treated_units=treated_units,
        lambda_unit=lambda_unit,
        lambda_time=lambda_time,
        lambda_nn=lambda_nn,
        treated_periods=treated_periods,
        solver=solver,
        verbose=verbose,
    )


def _placebo_rmse_for_lambdas(
    *,
    Y: np.ndarray,
    placebo_sets: Sequence[np.ndarray],
    treated_periods: int,
    lambda_unit: float,
    lambda_time: float,
    lambda_nn: float,
    n_jobs: int,
    prefer: str,
    solver: Optional[str],
    verbose: bool,
) -> Optional[float]:
    """
    Compute placebo RMSE for a given (lambda_unit, lambda_time, lambda_nn).
    Returns None if *all* placebo trials are non-finite (e.g., solver failures).
    """
    ates = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_simulate_ate)(
            Y,
            treated_units,
            treated_periods,
            lambda_unit,
            lambda_time,
            lambda_nn,
            solver,
            verbose,
        )
        for treated_units in placebo_sets
    )

    ates_arr = np.asarray(ates, dtype=float)
    ates_arr = ates_arr[np.isfinite(ates_arr)]
    if ates_arr.size == 0:
        return None
    return float(np.sqrt(np.mean(ates_arr ** 2)))


def TROP_cv_single(
    Y_control: ArrayLike,
    treated_periods: int,
    fixed_lambdas: Tuple[float, float] = (0.0, 0.0),
    lambda_grid: Optional[Iterable[float]] = None,
    lambda_cv: str = "unit",
    *,
    cv_sampling_method: str = "resample",
    n_trials: Optional[int] = 200,
    n_treated_units: Optional[int] = 1,
    K: Optional[int] = None,
    n_jobs: int = -1,
    prefer: str = "threads",
    random_seed: int = 0,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> float:
    """
    Tune one TROP tuning parameter by placebo cross-validation on a control-only panel.

    For each candidate value in ``lambda_grid``, this routine assigns placebo treatment
    in the final ``treated_periods`` columns, computes the corresponding TROP estimate,
    and selects the lambda minimizing the RMSE of placebo ATEs (targeting zero).

    Parameters
    ----------
    Y_control : array-like of shape (n_units, n_periods)
        Control-only outcome panel used for placebo cross-validation.
    treated_periods : int
        Number of placebo treated (post) periods, taken as the final columns.
        Must satisfy ``1 <= treated_periods <= n_periods - 1``.
    fixed_lambdas : tuple of float, default=(0.0, 0.0)
        Values held fixed for the two lambdas not being tuned. Interpretation depends on
        ``lambda_cv``:

        - if ``lambda_cv="unit"``: ``(lambda_time, lambda_nn)``
        - if ``lambda_cv="time"``: ``(lambda_unit, lambda_nn)``
        - if ``lambda_cv="nn"``  : ``(lambda_unit, lambda_time)``
    lambda_grid : iterable of float or None, default=None
        Candidate values for the lambda being tuned. If None, uses
        ``np.arange(0.0, 2.0, 0.2)``.
    lambda_cv : {"unit", "time", "nn"}, default="unit"
        Which lambda to tune.

    cv_sampling_method : {"resample", "kfold"}, default="resample"
        Method used to construct placebo treated sets.

        - ``"resample"``: draw ``n_trials`` placebo sets, each containing
          ``n_treated_units`` units sampled without replacement.
        - ``"kfold"``: split units into ``K`` folds (after shuffling); treat each
          fold once.

        Placebo sets are generated once using ``random_seed`` and reused across all
        candidate lambdas.
    n_trials : int, default=200
        Number of placebo trials when ``cv_sampling_method="resample"``.
        Required for ``"resample"``. Ignored for ``"kfold"``.
    n_treated_units : int, default=1
        Number of placebo treated units per trial when ``cv_sampling_method="resample"``.
        Required for ``"resample"``. Ignored for ``"kfold"``.
    K : int, default=None
        Number of folds when ``cv_sampling_method="kfold"``. Required for ``"kfold"``.
        Ignored for ``"resample"``.

    n_jobs : int, default=-1
        Number of parallel jobs. ``-1`` uses all available cores.
    prefer : {"threads", "processes"}, default="threads"
        Joblib backend preference.
    random_seed : int, default=0
        Random seed used to generate placebo sets (and fold shuffling for kfold).
    solver : str or None, default=None
        CVXPY solver passed to ``TROP_TWFE_average``.
    verbose : bool, default=False
        Verbosity flag passed to ``TROP_TWFE_average``.

    Returns
    -------
    best_lambda : float
        Selected lambda value minimizing the placebo RMSE.

    Raises
    ------
    ValueError
        If inputs are invalid (panel shape, treated_periods bounds, empty grid, or
        inconsistent CV sampling arguments).
    RuntimeError
        If all placebo trials/folds fail or return non-finite ATEs for a candidate
        lambda (e.g., solver failures).
    """
    Y = np.asarray(Y_control, dtype=float)
    N, _ = _validate_panel(Y, treated_periods)

    if lambda_cv not in {"unit", "time", "nn"}:
        raise ValueError("lambda_cv must be one of {'unit','time','nn'}.")

    if lambda_grid is None:
        lambda_grid_list = _as_list(np.arange(0.0, 2.0, 0.2), name="lambda_grid")
    else:
        lambda_grid_list = _as_list(lambda_grid, name="lambda_grid")

    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer.")

    placebo_sets = _generate_placebo_sets(
        N,
        cv_sampling_method=cv_sampling_method,
        n_treated_units=n_treated_units,
        n_trials=n_trials,
        K=K,
        random_state=random_seed,
    )

    scores: List[float] = []
    for lamb in lambda_grid_list:
        if lamb < 0:
            raise ValueError("Lambda values must be nonnegative.")

        if lambda_cv == "unit":
            lambda_unit, lambda_time, lambda_nn = float(lamb), float(fixed_lambdas[0]), float(fixed_lambdas[1])
        elif lambda_cv == "time":
            lambda_unit, lambda_time, lambda_nn = float(fixed_lambdas[0]), float(lamb), float(fixed_lambdas[1])
        else:
            lambda_unit, lambda_time, lambda_nn = float(fixed_lambdas[0]), float(fixed_lambdas[1]), float(lamb)

        score = _placebo_rmse_for_lambdas(
            Y=Y,
            placebo_sets=placebo_sets,
            treated_periods=treated_periods,
            lambda_unit=lambda_unit,
            lambda_time=lambda_time,
            lambda_nn=lambda_nn,
            n_jobs=n_jobs,
            prefer=prefer,
            solver=solver,
            verbose=verbose,
        )

        if score is None:
            raise RuntimeError(
                "All placebo trials failed or returned non-finite ATEs for "
                f"lambda={lamb} (lambda_cv='{lambda_cv}'). Consider changing solver/settings."
            )

        scores.append(score)

    best_idx = int(np.argmin(scores))
    return float(lambda_grid_list[best_idx])


def TROP_cv_cycle(
    Y_control: ArrayLike,
    treated_periods: int,
    unit_grid: Sequence[float],
    time_grid: Sequence[float],
    nn_grid: Sequence[float],
    lambdas_init: Optional[Tuple[float, float, float]] = None,
    *,
    max_iter: int = 50,
    cv_sampling_method: str = "resample",
    n_trials: Optional[int] = 200,
    n_treated_units: Optional[int] = 1,
    K: Optional[int] = None,
    n_jobs: int = -1,
    prefer: str = "threads",
    random_seed: int = 0,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """
    Tune (lambda_unit, lambda_time, lambda_nn) by coordinate-descent placebo cross-validation.

    Iteratively updates one tuning parameter at a time using ``TROP_cv_single`` while
    holding the other two fixed, until the selected triplet stops changing or
    ``max_iter`` is reached. Each coordinate update selects the lambda minimizing the
    RMSE of placebo ATEs on a control-only panel (targeting zero placebo effects).

    Parameters
    ----------
    Y_control : array-like of shape (n_units, n_periods)
        Control-only outcome panel used for placebo cross-validation.
    treated_periods : int
        Number of placebo treated (post) periods, taken as the final columns.
        Must satisfy ``1 <= treated_periods <= n_periods - 1``.
    unit_grid : sequence of float
        Candidate values for ``lambda_unit`` (unit-distance decay).
    time_grid : sequence of float
        Candidate values for ``lambda_time`` (time-distance decay).
    nn_grid : sequence of float
        Candidate values for ``lambda_nn`` (nuclear-norm penalty).
    lambdas_init : tuple of float or None, default=None
        Initial values ``(lambda_unit, lambda_time, lambda_nn)``. If None, each
        parameter is initialized to the mean of its grid.

    max_iter : int, default=50
        Maximum number of coordinate-descent iterations.

    cv_sampling_method : {"resample", "kfold"}, default="resample"
        Method used to construct placebo treated sets for each coordinate update.

        - ``"resample"``: draw ``n_trials`` placebo sets, each containing
          ``n_treated_units`` units sampled without replacement.
        - ``"kfold"``: split units into ``K`` folds (after shuffling); treat each
          fold once.

        Placebo sets are generated using ``random_seed`` for each call to
        ``TROP_cv_single`` (so each coordinate uses the same placebo assignments
        across its grid).
    n_trials : int, default=200
        Number of placebo trials when ``cv_sampling_method="resample"``.
        Ignored when ``cv_sampling_method="kfold"``.
    n_treated_units : int, default=1
        Number of placebo treated units per trial when ``cv_sampling_method="resample"``.
        Ignored when ``cv_sampling_method="kfold"``.
    K : int, default=None
        Number of folds when ``cv_sampling_method="kfold"``. Required for ``"kfold"``.
        Ignored when ``cv_sampling_method="resample"``.

    n_jobs : int, default=-1
        Number of parallel jobs. ``-1`` uses all available cores.
    prefer : {"threads", "processes"}, default="threads"
        Joblib backend preference.
    random_seed : int, default=0
        Random seed used for placebo-set generation (and fold shuffling for kfold).
    solver : str or None, default=None
        CVXPY solver passed to ``TROP_TWFE_average``.
    verbose : bool, default=False
        Verbosity flag passed to ``TROP_TWFE_average``.

    Returns
    -------
    lambda_unit : float
        Selected value for ``lambda_unit``.
    lambda_time : float
        Selected value for ``lambda_time``.
    lambda_nn : float
        Selected value for ``lambda_nn``.

    Raises
    ------
    RuntimeError
        If the procedure does not converge (no fixed point) within ``max_iter``.
    """
    Y = np.asarray(Y_control, dtype=float)
    _validate_panel(Y, treated_periods)

    unit_grid_list = _as_list(unit_grid, name="unit_grid")
    time_grid_list = _as_list(time_grid, name="time_grid")
    nn_grid_list = _as_list(nn_grid, name="nn_grid")

    if lambdas_init is None:
        lambda_unit = float(np.mean(unit_grid_list))
        lambda_time = float(np.mean(time_grid_list))
        lambda_nn = float(np.mean(nn_grid_list))
    else:
        lambda_unit, lambda_time, lambda_nn = map(float, lambdas_init)

    for _ in range(max_iter):
        old = (lambda_unit, lambda_time, lambda_nn)

        lambda_unit = TROP_cv_single(
            Y,
            treated_periods,
            fixed_lambdas=(lambda_time, lambda_nn),
            lambda_grid=unit_grid_list,
            lambda_cv="unit",
            cv_sampling_method=cv_sampling_method,
            n_trials=n_trials,
            n_treated_units=n_treated_units,
            K=K,
            n_jobs=n_jobs,
            prefer=prefer,
            random_seed=random_seed,
            solver=solver,
            verbose=verbose,
        )

        lambda_time = TROP_cv_single(
            Y,
            treated_periods,
            fixed_lambdas=(lambda_unit, lambda_nn),
            lambda_grid=time_grid_list,
            lambda_cv="time",
            cv_sampling_method=cv_sampling_method,
            n_trials=n_trials,
            n_treated_units=n_treated_units,
            K=K,
            n_jobs=n_jobs,
            prefer=prefer,
            random_seed=random_seed,
            solver=solver,
            verbose=verbose,
        )

        lambda_nn = TROP_cv_single(
            Y,
            treated_periods,
            fixed_lambdas=(lambda_unit, lambda_time),
            lambda_grid=nn_grid_list,
            lambda_cv="nn",
            cv_sampling_method=cv_sampling_method,
            n_trials=n_trials,
            n_treated_units=n_treated_units,
            K=K,
            n_jobs=n_jobs,
            prefer=prefer,
            random_seed=random_seed,
            solver=solver,
            verbose=verbose,
        )

        new = (lambda_unit, lambda_time, lambda_nn)
        if new == old:
            return new

    raise RuntimeError("TROP_cv_cycle did not converge (no fixed point) within max_iter.")


def TROP_cv_joint(
    Y_control: ArrayLike,
    treated_periods: int,
    unit_grid: Sequence[float],
    time_grid: Sequence[float],
    nn_grid: Sequence[float],
    *,
    cv_sampling_method: str = "resample",
    n_trials: Optional[int] = 200,
    n_treated_units: Optional[int] = 1,
    K: Optional[int] = None,
    n_jobs: int = -1,
    prefer: str = "threads",
    random_seed: int = 0,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """
    Select (lambda_unit, lambda_time, lambda_nn) by joint placebo cross-validation.

    Performs a full grid search over ``unit_grid`` × ``time_grid`` × ``nn_grid``.
    For each candidate triple, repeatedly assigns placebo treatment to a subset of
    units and computes the corresponding TROP estimate. The selected triple is the
    one minimizing the root-mean-square error (RMSE) of placebo ATEs (targeting zero).

    Parameters
    ----------
    Y_control : array-like of shape (n_units, n_periods)
        Control-only outcome panel used for placebo cross-validation.
    treated_periods : int
        Number of placebo treated (post) periods, taken as the final columns.
        Must satisfy ``1 <= treated_periods <= n_periods - 1``.
    unit_grid : sequence of float
        Candidate values for ``lambda_unit`` (unit-distance decay).
    time_grid : sequence of float
        Candidate values for ``lambda_time`` (time-distance decay).
    nn_grid : sequence of float
        Candidate values for ``lambda_nn`` (nuclear-norm penalty).

    cv_sampling_method : {"resample", "kfold"}, default="resample"
        Method used to construct placebo treated sets.

        - ``"resample"``: draw ``n_trials`` placebo sets, each containing
          ``n_treated_units`` units sampled without replacement.
        - ``"kfold"``: split units into ``K`` folds (after shuffling); treat each
          fold once.

        Placebo sets are generated once using ``random_seed`` and reused across all
        candidate triples.
    n_trials : int, default=200
        Number of placebo trials when ``cv_sampling_method="resample"``.
        Ignored when ``cv_sampling_method="kfold"``.
    n_treated_units : int, default=1
        Number of placebo treated units per trial when ``cv_sampling_method="resample"``.
        Ignored when ``cv_sampling_method="kfold"``.
    K : int, default=None
        Number of folds when ``cv_sampling_method="kfold"``. Required for ``"kfold"``.
        Ignored when ``cv_sampling_method="resample"``.

    n_jobs : int, default=-1
        Number of parallel jobs. ``-1`` uses all available cores.
    prefer : {"threads", "processes"}, default="threads"
        Joblib backend preference.
    random_seed : int, default=0
        Random seed used to generate placebo sets (and fold shuffling for kfold).
    solver : str or None, default=None
        CVXPY solver passed to ``TROP_TWFE_average``.
    verbose : bool, default=False
        Verbosity flag passed to ``TROP_TWFE_average``.

    Returns
    -------
    lambda_unit : float
        Selected value for ``lambda_unit``.
    lambda_time : float
        Selected value for ``lambda_time``.
    lambda_nn : float
        Selected value for ``lambda_nn``.

    Raises
    ------
    ValueError
        If inputs are invalid (panel shape, treated_periods bounds, empty grids, or
        inconsistent CV sampling arguments).
    RuntimeError
        If all parameter combinations fail (e.g., all placebo evaluations return
        non-finite ATEs due to solver failures).
    """
    Y = np.asarray(Y_control, dtype=float)
    N, _ = _validate_panel(Y, treated_periods)

    unit_grid_list = _as_list(unit_grid, name="unit_grid")
    time_grid_list = _as_list(time_grid, name="time_grid")
    nn_grid_list = _as_list(nn_grid, name="nn_grid")

    placebo_sets = _generate_placebo_sets(
        N,
        cv_sampling_method=cv_sampling_method,
        n_treated_units=n_treated_units,
        n_trials=n_trials,
        K=K,
        random_state=random_seed,
    )

    best_params: Optional[Tuple[float, float, float]] = None
    best_score: float = float("inf")

    for lambda_unit in unit_grid_list:
        for lambda_time in time_grid_list:
            for lambda_nn in nn_grid_list:
                score = _placebo_rmse_for_lambdas(
                    Y=Y,
                    placebo_sets=placebo_sets,
                    treated_periods=treated_periods,
                    lambda_unit=float(lambda_unit),
                    lambda_time=float(lambda_time),
                    lambda_nn=float(lambda_nn),
                    n_jobs=n_jobs,
                    prefer=prefer,
                    solver=solver,
                    verbose=verbose,
                )
                if score is None:
                    continue

                if score < best_score:
                    best_score = score
                    best_params = (float(lambda_unit), float(lambda_time), float(lambda_nn))

    if best_params is None:
        raise RuntimeError("All parameter combinations failed during joint CV. Check solver/settings.")
    return best_params