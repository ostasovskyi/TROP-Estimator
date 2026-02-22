from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union, List

import numpy as np
from joblib import Parallel, delayed

from .estimator import TROP_TWFE_average

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def generate_placebo_sets(
    N,
    cv_sampling_method="resample",
    n_treated_units=None,
    n_trials=None,
    K=None,
    random_state=None,
) -> List[np.ndarray]:
    """
    Generate placebo treatment sets for cross-validation.

    Parameters
    ----------
    N : int
        Total number of units.
    cv_sampling_method : {"resample", "kfold"}
        - "resample": random placebo assignment repeated for a number of trials
        - "kfold": K-fold cross-validation (entire fold treated)
    n_treated_units : int, optional
        Required for cv_sampling_method="resample".
    n_trials : int, optional
        Required for cv_sampling_method="resample".
    K : int, optional
        Required for cv_sampling_method="kfold".
    random_state : int or None
        Random seed.

    Returns
    -------
    placebo_sets : list of np.ndarray
        List of arrays of placebo-treated unit indices.
    """

    rng = np.random.default_rng(random_state)
    units = np.arange(N)

    if cv_sampling_method == "resample":
        if n_treated_units is None or n_trials is None:
            raise ValueError(
                "resample requires n_treated_units and n_trials"
            )

        placebo_sets = [
            rng.choice(units, size=n_treated_units, replace=False)
            for _ in range(n_trials)
        ]

    elif cv_sampling_method == "kfold":
        if K is None:
            raise ValueError("kfold requires K")

        shuffled = rng.permutation(units)
        folds = np.array_split(shuffled, K)

        placebo_sets = [fold for fold in folds]

    else:
        raise ValueError("method must be 'resample' or 'kfold'")

    return placebo_sets
    
def _validate_panel(Y: np.ndarray, treated_periods: int, n_treated_units: int) -> None:
    """Validate panel dimensions and placebo CV inputs."""
    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array of shape (N, T).")
    N, T = Y.shape
    if treated_periods <= 0 or treated_periods >= T:
        raise ValueError(f"treated_periods must be in [1, T-1]. Got treated_periods={treated_periods}, T={T}.")
    if n_treated_units <= 0 or n_treated_units >= N:
        raise ValueError(f"n_treated_units must be in [1, N-1]. Got n_treated_units={n_treated_units}, N={N}.")


def _as_list(grid: Iterable[float]) -> List[float]:
    """Convert an iterable of grid values to a non-empty list of floats."""
    grid_list = list(grid)
    if len(grid_list) == 0:
        raise ValueError("lambda_grid must be non-empty.")
    grid_list = [float(x) for x in grid_list]
    return grid_list



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
    """
    Simulate a single placebo ATE given indices of placebo treated units.
    """
   
    N, _ = Y.shape

    #treated_units = rng.choice(N, size=n_treated_units, replace=False)
            
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


def TROP_cv_single(
    Y_control: ArrayLike,
    treated_periods: int,
    fixed_lambdas: Tuple[float, float] = (0.0, 0.0),
    lambda_grid: Optional[Iterable[float]] = None,
    lambda_cv: str = "unit",
    *,
    cv_sampling_method="resample",
    n_trials: Optional[int] = 200,
    n_treated_units:  Optional[int] = 1,
    K: Optional[int] = None,
    n_jobs: int = -1,
    prefer: str = "threads",
    random_seed: int = 0,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> float:
    """
    Tune one TROP tuning parameter via placebo cross-validation on a control-only panel.

    For each candidate value in `lambda_grid`, this routine repeatedly assigns a placebo
    treatment to random units in the last `treated_periods` columns, computes the
    corresponding TROP estimate, and selects the lambda that minimizes the RMSE of
    placebo effects.

    Parameters
    ----------
    Y_control : array_like of shape (N, T)
        Control-only outcome panel used for placebo cross-validation.
    treated_periods : int
        Number of placebo treated (post) periods, taken as the final columns.
    fixed_lambdas : tuple of float, default=(0.0, 0.0)
        Values held fixed for the two lambdas not being tuned. Interpretation depends on
        `lambda_cv`:
        - 'unit': (lambda_time, lambda_nn)
        - 'time': (lambda_unit, lambda_nn)
        - 'nn'  : (lambda_unit, lambda_time)
    lambda_grid : iterable of float or None, default=None
        Candidate values for the lambda being tuned. If None, uses ``np.arange(0, 2, 0.2)``.
    lambda_cv : {'unit', 'time', 'nn'}, default='unit'
        Which lambda to tune.
        
    cv_sampling_method : {"resample", "kfold"}
        - "resample": random placebo assignment repeated for a number of trials
        - "kfold": K-fold cross-validation (entire fold treated)
    n_trials : int, optional, default=200
        If using random sampling, number of trials.
        Required for cv_sampling_method="resample".
    n_treated_units : int, optional, default=1
        If using random sampling, number of placebo treated units sampled (without replacement) per trial.
        Required for cv_sampling_method="resample".
    K : int, optional
        If using K-fold splitting (splitting full data), number of folds
        Required for cv_sampling_method="kfold".
        
        
    n_jobs : int, default=-1
        Number of parallel jobs for placebo trials. ``-1`` uses all available cores.
    prefer : {'threads', 'processes'}, default='threads'
        joblib backend preference.
    random_seed : int, default=0
        Seed for generating trial seeds (deterministic tuning).
    solver : str or None, default=None
        CVXPY solver passed to ``TROP_TWFE_average``.
    verbose : bool, default=False
        Verbosity flag passed to ``TROP_TWFE_average``.

    Returns
    -------
    float
        Selected lambda value minimizing the RMSE of placebo estimates.
    """
    Y = np.asarray(Y_control, dtype=float)
    _validate_panel(Y, treated_periods, n_treated_units)

    if lambda_cv not in {"unit", "time", "nn"}:
        raise ValueError("lambda_cv must be one of {'unit','time','nn'}.")

    if lambda_grid is None:
        lambda_grid_list = _as_list(np.arange(0.0, 2.0, 0.2))
    else:
        lambda_grid_list = _as_list(lambda_grid)

    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    
   # generate the placebo sets, this is the indices of placebo treated samples for each trial/fold depending on the sampling method 
    placebo_sets = generate_placebo_sets(
        Y.shape[0],
        cv_sampling_method=cv_sampling_method,
        n_treated_units=n_treated_units,
        n_trials=n_trials,
        K=K,
        random_state=random_seed # note that we use the base random seed because we want to have the same splitting for each lambda
    )
    if cv_sampling_method == "resample":
        n_trial_or_fold = n_trials
    elif cv_sampling_method == "kfold":
        n_trial_or_fold = K

    scores: List[float] = []

    for lamb in lambda_grid_list:
        if lamb < 0:
            raise ValueError("Lambda values must be nonnegative.")

        if lambda_cv == "unit":
            lambda_unit, lambda_time, lambda_nn = lamb, float(fixed_lambdas[0]), float(fixed_lambdas[1])
        elif lambda_cv == "time":
            lambda_unit, lambda_time, lambda_nn = float(fixed_lambdas[0]), lamb, float(fixed_lambdas[1])
        else:  # 'nn'
            lambda_unit, lambda_time, lambda_nn = float(fixed_lambdas[0]), float(fixed_lambdas[1]), lamb

        ates = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(_simulate_ate)(
                Y,
                placebo_sets[trial_or_fold],
                treated_periods,
                lambda_unit,
                lambda_time,
                lambda_nn,
                solver,
                verbose,
            )
            for trial_or_fold in range(n_trial_or_fold)
        )

        ates_arr = np.asarray(ates, dtype=float)
        ates_arr = ates_arr[np.isfinite(ates_arr)]

        if ates_arr.size == 0:
            raise RuntimeError(
                f"All placebo trials failed or returned non-finite ATEs for lambda={lamb} "
                f"(lambda_cv='{lambda_cv}'). Consider changing solver/settings."
            )

        scores.append(float(np.sqrt(np.mean(ates_arr**2))))
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
    cv_sampling_method="resample",
    n_trials: Optional[int] = 200,
    n_treated_units:  Optional[int] = 1, 
    K: Optional[int] = None,
    n_jobs: int = -1,
    prefer: str = "threads",
    random_seed: int = 0,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """
    Tune (lambda_unit, lambda_time, lambda_nn) by coordinate-descent placebo cross-validation.

    Iteratively updates one tuning parameter at a time using `TROP_cv_single` (holding the
    other two fixed) until the selected triplet stops changing or `max_iter` is reached.
    Each update minimizes the RMSE of placebo effects on a control-only panel.

    Parameters
    ----------
    Y_control : array_like of shape (N, T)
        Control-only outcome panel used for placebo cross-validation.
    n_treated_units : int
        Number of placebo treated units sampled (without replacement) per trial.
    treated_periods : int
        Number of placebo treated (post) periods, taken as the final columns.
    unit_grid : sequence of float
        Candidate values for `lambda_unit` (unit-distance decay).
    time_grid : sequence of float
        Candidate values for `lambda_time` (time-distance decay).
    nn_grid : sequence of float
        Candidate values for `lambda_nn` (nuclear-norm penalty).
    lambdas_init : tuple of float or None, default=None
        Initial values (lambda_unit, lambda_time, lambda_nn). If None, initializes each
        parameter to the mean of its grid.
    max_iter : int, default=50
        Maximum number of coordinate-descent iterations.
    
    cv_sampling_method : {"resample", "kfold"}
        - "resample": random placebo assignment repeated for a number of trials
        - "kfold": K-fold cross-validation (entire fold treated)
    n_trials : int, optional, default=200
        If using random sampling, number of trials.
        Required for cv_sampling_method="resample".
    n_treated_units : int, optional, default=1
        If using random sampling, number of placebo treated units sampled (without replacement) per trial.
        Required for cv_sampling_method="resample".
    K : int, optional
        If using K-fold splitting (splitting full data), number of folds
        Required for cv_sampling_method="kfold".
        
    n_jobs : int, default=-1
        Number of parallel jobs for placebo trials. ``-1`` uses all available cores.
    prefer : {'threads', 'processes'}, default='threads'
        joblib backend preference.
    random_seed : int, default=0
        Seed for generating trial seeds (deterministic tuning).
    solver : str or None, default=None
        CVXPY solver passed to ``TROP_TWFE_average``.
    verbose : bool, default=False
        Verbosity flag passed to ``TROP_TWFE_average``.

    Returns
    -------
    tuple of float
        (lambda_unit, lambda_time, lambda_nn) at the fixed point of the coordinate updates.

    Raises
    ------
    RuntimeError
        If the procedure does not converge within `max_iter`.
    """
    Y = np.asarray(Y_control, dtype=float)
    _validate_panel(Y, treated_periods, n_treated_units)

    unit_grid_list = _as_list(unit_grid)
    time_grid_list = _as_list(time_grid)
    nn_grid_list = _as_list(nn_grid)

    if lambdas_init is None:
        lambda_unit = float(np.mean(unit_grid_list))
        lambda_time = float(np.mean(time_grid_list))
        lambda_nn = float(np.mean(nn_grid_list))
    else:
        lambda_unit, lambda_time, lambda_nn = map(float, lambdas_init)

    for _ in range(max_iter):
        old = (lambda_unit, lambda_time, lambda_nn)

        lambda_unit = TROP_cv_single(
            Y, treated_periods,
            fixed_lambdas=(lambda_time, lambda_nn),
            lambda_grid=unit_grid_list,
            lambda_cv="unit", 
            cv_sampling_method=cv_sampling_method, n_treated_units=n_treated_units, n_trials=n_trials, K=K,
            n_jobs=n_jobs, prefer=prefer,
            random_seed=random_seed, solver=solver, verbose=verbose
        )

        lambda_time = TROP_cv_single(
            Y, treated_periods,
            fixed_lambdas=(lambda_unit, lambda_nn),
            lambda_grid=time_grid_list,
            lambda_cv="time",
            cv_sampling_method=cv_sampling_method, n_treated_units=n_treated_units, n_trials=n_trials, K=K, 
            n_jobs=n_jobs, prefer=prefer,
            random_seed=random_seed, solver=solver, verbose=verbose
        )

        lambda_nn = TROP_cv_single(
            Y, treated_periods,
            fixed_lambdas=(lambda_unit, lambda_time),
            lambda_grid=nn_grid_list,
            lambda_cv="nn",
            cv_sampling_method=cv_sampling_method, n_treated_units=n_treated_units, n_trials=n_trials, K=K,
            n_jobs=n_jobs, prefer=prefer,
            random_seed=random_seed, solver=solver, verbose=verbose
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
    cv_sampling_method="resample",
    n_trials: Optional[int] = 200,
    n_treated_units:  Optional[int] = 1, 
    K: Optional[int] = None,
    n_jobs: int = -1,
    prefer: str = "threads",
    random_seed: int = 0,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """
    Select (lambda_unit, lambda_time, lambda_nn) by joint placebo cross-validation.

    Performs a full grid search over `unit_grid` × `time_grid` × `nn_grid`. For each
    candidate triple, repeatedly assigns a placebo treatment to random units in the
    last `treated_periods` columns and selects the triple that minimizes the RMSE of
    placebo effects on the control-only panel.

    Parameters
    ----------
    Y_control : array_like of shape (N, T)
        Control-only outcome panel used for placebo cross-validation.
    n_treated_units : int
        
    treated_periods : int
        Number of placebo treated (post) periods, taken as the final columns.
    unit_grid : sequence of float
        Candidate values for `lambda_unit` (unit-distance decay).
    time_grid : sequence of float
        Candidate values for `lambda_time` (time-distance decay).
    nn_grid : sequence of float
        Candidate values for `lambda_nn` (nuclear-norm penalty).
          
    cv_sampling_method : {"resample", "kfold"}
        - "resample": random placebo assignment repeated for a number of trials
        - "kfold": K-fold cross-validation (entire fold treated)
    n_trials : int, optional, default=200
        If using random sampling, number of trials.
        Required for cv_sampling_method="resample".
    n_treated_units : int, optional, default=1
        If using random sampling, number of placebo treated units sampled (without replacement) per trial.
        Required for cv_sampling_method="resample".
    K : int, optional
        If using K-fold splitting (splitting full data), number of folds
        Required for cv_sampling_method="kfold".

    n_jobs : int, default=-1
        Number of parallel jobs for placebo trials. ``-1`` uses all available cores.
    prefer : {'threads', 'processes'}, default='threads'
        joblib backend preference.
    random_seed : int, default=0
        Seed for generating trial seeds (deterministic tuning).
    solver : str or None, default=None
        CVXPY solver passed to ``TROP_TWFE_average``.
    verbose : bool, default=False
        Verbosity flag passed to ``TROP_TWFE_average``.

    Returns
    -------
    tuple of float
        (lambda_unit, lambda_time, lambda_nn) minimizing the RMSE of placebo estimates.

    Raises
    ------
    RuntimeError
        If all parameter combinations fail (e.g., solver failures for every triple).
    """
    Y = np.asarray(Y_control, dtype=float)
    _validate_panel(Y, treated_periods, n_treated_units)

    unit_grid_list = _as_list(unit_grid)
    time_grid_list = _as_list(time_grid)
    nn_grid_list = _as_list(nn_grid)

    best_params: Optional[Tuple[float, float, float]] = None
    best_score: float = float("inf")

    # generate the placebo sets, this is the indices of placebo treated samples for each trial/fold depending on the sampling method 
    placebo_sets = generate_placebo_sets(
        Y.shape[0],
        cv_sampling_method=cv_sampling_method,
        n_treated_units=n_treated_units,
        n_trials=n_trials,
        K=K,
        random_state=random_seed # note that we use the base random seed because we want to have the same splitting for each lambda
    )
    if cv_sampling_method == "resample":
        n_trial_or_fold = n_trials
    elif cv_sampling_method == "kfold":
        n_trial_or_fold = K

    for lambda_unit in unit_grid_list:
        for lambda_time in time_grid_list:
            for lambda_nn in nn_grid_list:
                ates = Parallel(n_jobs=n_jobs, prefer=prefer)(
                    delayed(_simulate_ate)(
                        Y,
                        placebo_sets[trial_or_fold], # use the current placebo trial/fold
                        treated_periods,
                        float(lambda_unit),
                        float(lambda_time),
                        float(lambda_nn),
                        solver,
                        verbose,
                    )
                    for trial_or_fold in range(n_trial_or_fold)
                )
    
                ates_arr = np.asarray(ates, dtype=float)
                ates_arr = ates_arr[np.isfinite(ates_arr)]
                if ates_arr.size == 0:
                    continue  # skip invalid setting

                score = float(np.sqrt(np.mean(ates_arr**2)))
                if score < best_score:
                    best_score = score
                    best_params = (float(lambda_unit), float(lambda_time), float(lambda_nn))

    if best_params is None:
        raise RuntimeError("All parameter combinations failed during joint CV. Check solver/settings.")
    return best_params