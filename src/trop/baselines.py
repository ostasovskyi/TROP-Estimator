from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import cvxpy as cp


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _check_panel(Y: np.ndarray, W: np.ndarray) -> Tuple[int, int]:
    if Y.ndim != 2 or W.ndim != 2:
        raise ValueError(f"Y and W must be 2D arrays. Got Y.ndim={Y.ndim}, W.ndim={W.ndim}.")
    if Y.shape != W.shape:
        raise ValueError(f"Y and W must have the same shape. Got Y={Y.shape}, W={W.shape}.")
    return Y.shape


def _check_treated_periods(treated_periods: int, T: int) -> None:
    if not isinstance(treated_periods, int) or treated_periods <= 0:
        raise ValueError("treated_periods must be a positive integer.")
    if treated_periods >= T:
        raise ValueError(f"treated_periods must be < T. Got treated_periods={treated_periods}, T={T}.")


def _check_treated_units(treated_units: Sequence[int], N: int) -> np.ndarray:
    arr = np.asarray(treated_units, dtype=int)
    if arr.size == 0:
        raise ValueError("treated_units must contain at least one unit index.")
    if np.any(arr < 0) or np.any(arr >= N):
        raise ValueError(f"treated_units contains out-of-range indices for N={N}: {arr}")
    return arr


def _solve(prob: cp.Problem, solver: Optional[str], default_solver: str, verbose: bool) -> None:
    chosen = solver or default_solver
    prob.solve(solver=chosen, verbose=verbose)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimization did not converge. Solver={chosen}, status={prob.status}.")


def DID_TWFE(
    Y: ArrayLike,
    W: ArrayLike,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> float:
    """
    Classic two-way fixed-effects difference-in-differences estimator.

    Fits unit and time fixed effects plus a single scalar coefficient ``tau``
    on the treatment indicator ``W``.

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Outcome matrix.
    W : array_like of shape (N, T)
        Treatment indicator matrix.
    solver : str or None, default=None
        CVXPY solver name. Defaults to "OSQP".
    verbose : bool, default=False
        Passed to ``cvxpy.Problem.solve``.

    Returns
    -------
    float
        Estimated treatment-effect coefficient ``tau``.
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    N, T = _check_panel(Y, W)

    unit_effects = cp.Variable((1, N))
    time_effects = cp.Variable((1, T))
    unit_factor = cp.kron(np.ones((T, 1)), unit_effects).T
    time_factor = cp.kron(np.ones((N, 1)), time_effects)
    mu = cp.Variable()
    tau = cp.Variable()

    residual = Y - unit_factor - time_factor - mu - W * tau
    prob = cp.Problem(cp.Minimize(cp.sum_squares(residual)))
    _solve(prob, solver, "OSQP", verbose)

    tau_hat = tau.value
    if tau_hat is None or not np.isfinite(tau_hat):
        raise RuntimeError("Optimization did not return a valid tau.")
    return float(tau_hat)


def SC_TWFE(
    Y: ArrayLike,
    W: ArrayLike,
    treated_units: Sequence[int],
    treated_periods: int = 10,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> float:
    """
    Synthetic Control estimator with convex donor-unit weighting.

    Fits nonnegative, sum-to-one weights over control units that best match the
    average treated pre-period trajectory, then compares the treated post-period
    average to the weighted-donor post-period average.

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Outcome matrix.
    W : array_like of shape (N, T)
        Treatment indicator matrix (used only to validate the panel shape).
    treated_units : sequence of int
        Row indices of treated units.
    treated_periods : int, default=10
        Number of final columns treated as the post-treatment block.
    solver : str or None, default=None
        CVXPY solver name. Defaults to "OSQP".
    verbose : bool, default=False
        Passed to ``cvxpy.Problem.solve``.

    Returns
    -------
    float
        Estimated treatment effect: treated post-period average minus the
        weighted-donor post-period average.
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    N, T = _check_panel(Y, W)
    _check_treated_periods(treated_periods, T)
    treated_units_arr = _check_treated_units(treated_units, N)

    X = np.delete(Y, treated_units_arr, axis=0)[:, :-treated_periods].T
    y = np.mean(Y[treated_units_arr, :-treated_periods], axis=0).T
    _, N_control = X.shape
    if N_control == 0:
        raise ValueError("No control units remain after removing treated_units.")

    unit_weights = cp.Variable((N_control,), nonneg=True)
    constraints = [cp.sum(unit_weights) == 1]
    objective = cp.sum_squares(y - X @ unit_weights)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    _solve(prob, solver, "OSQP", verbose)

    X_predict = np.delete(Y, treated_units_arr, axis=0)[:, -treated_periods:].T
    y_predict = X_predict.dot(unit_weights.value)

    tau_hat = float(np.mean(Y[treated_units_arr, -treated_periods:]) - np.mean(y_predict))
    if not np.isfinite(tau_hat):
        raise RuntimeError("Optimization did not return a valid tau.")
    return tau_hat


def DIFP_TWFE(
    Y: ArrayLike,
    W: ArrayLike,
    treated_units: Sequence[int],
    treated_periods: int = 10,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> float:
    """
    Difference-in-fitted-parts estimator: Synthetic Control with an intercept.

    Same as ``SC_TWFE`` but allows a free intercept alongside the convex donor
    weights when fitting the pre-period treated trajectory.

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Outcome matrix.
    W : array_like of shape (N, T)
        Treatment indicator matrix (used only to validate the panel shape).
    treated_units : sequence of int
        Row indices of treated units.
    treated_periods : int, default=10
        Number of final columns treated as the post-treatment block.
    solver : str or None, default=None
        CVXPY solver name. Defaults to "OSQP".
    verbose : bool, default=False
        Passed to ``cvxpy.Problem.solve``.

    Returns
    -------
    float
        Estimated treatment effect: treated post-period average minus the
        weighted-donor (plus intercept) post-period average.
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    N, T = _check_panel(Y, W)
    _check_treated_periods(treated_periods, T)
    treated_units_arr = _check_treated_units(treated_units, N)

    X = np.delete(Y, treated_units_arr, axis=0)[:, :-treated_periods].T
    y = np.mean(Y[treated_units_arr, :-treated_periods], axis=0).T
    _, N_control = X.shape
    if N_control == 0:
        raise ValueError("No control units remain after removing treated_units.")

    unit_weights = cp.Variable((N_control,), nonneg=True)
    intercept = cp.Variable()
    constraints = [cp.sum(unit_weights) == 1]
    objective = cp.sum_squares(y - X @ unit_weights - intercept)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    _solve(prob, solver, "OSQP", verbose)

    X_predict = np.delete(Y, treated_units_arr, axis=0)[:, -treated_periods:].T
    y_predict = X_predict.dot(unit_weights.value) + intercept.value

    tau_hat = float(np.mean(Y[treated_units_arr, -treated_periods:]) - np.mean(y_predict))
    if not np.isfinite(tau_hat):
        raise RuntimeError("Optimization did not return a valid tau.")
    return tau_hat


def SDID_weights(
    Y: ArrayLike,
    treated_units: Sequence[int],
    treated_periods: int,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic Difference-in-Differences unit and time weights.

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Outcome matrix.
    treated_units : sequence of int
        Row indices of treated units.
    treated_periods : int
        Number of final columns treated as the post-treatment block.
    solver : str or None, default=None
        CVXPY solver name. Defaults to "OSQP".
    verbose : bool, default=False
        Passed to ``cvxpy.Problem.solve``.

    Returns
    -------
    unit_weights : ndarray of shape (N,)
        Nonnegative unit weights (treated units weighted uniformly, control
        units fit by regularized convex regression), summing to 1 over
        treated and, separately, over control units.
    time_weights : ndarray of shape (T,)
        Nonnegative pre-period weights (post-period weighted uniformly),
        summing to 1 over pre-periods.
    """
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    if Y.ndim != 2:
        raise ValueError(f"Y must be a 2D array of shape (N, T). Got Y.ndim={Y.ndim}.")
    _check_treated_periods(treated_periods, T)
    treated_units_arr = _check_treated_units(treated_units, N)
    if T - treated_periods < 2:
        raise ValueError(
            "SDID_weights requires at least 2 pre-treatment periods. "
            f"Got T={T}, treated_periods={treated_periods}."
        )

    unit_weights_full = np.zeros((N,))
    time_weights_full = np.zeros((T,))

    control_units = ~np.isin(np.arange(N), treated_units_arr)
    if not np.any(control_units):
        raise ValueError("No control units remain after removing treated_units.")

    # unit weights
    X = Y[control_units, :-treated_periods].T
    y = np.mean(Y[treated_units_arr, :-treated_periods].T, axis=1)
    unit_weights = cp.Variable((np.sum(control_units),), nonneg=True)
    constraints = [cp.sum(unit_weights) == 1]

    # regularization (zeta^2)
    Delta = Y[control_units, :-treated_periods][:, 1:] - Y[control_units, :-treated_periods][:, :-1]
    var = np.var(Delta)
    reg = np.sqrt(treated_units_arr.shape[0] * treated_periods) * var

    mu = cp.Variable()
    objective = cp.sum_squares(y - X @ unit_weights - mu) + reg * (T - treated_periods) * cp.sum_squares(unit_weights)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    _solve(prob, solver, "OSQP", verbose)
    unit_weights_full[control_units] = unit_weights.value
    unit_weights_full[treated_units_arr] = 1.0 / treated_units_arr.shape[0]

    # time weights
    X = Y[control_units, :-treated_periods]
    y = np.mean(Y[control_units, -treated_periods:], axis=1)
    time_weights = cp.Variable((T - treated_periods,), nonneg=True)
    constraints = [cp.sum(time_weights) == 1]

    mu = cp.Variable()
    objective = cp.sum_squares(y - X @ time_weights - mu)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    _solve(prob, solver, "OSQP", verbose)
    time_weights_full[:-treated_periods] = time_weights.value
    time_weights_full[-treated_periods:] = 1.0 / treated_periods

    return unit_weights_full, time_weights_full


def SDID_TWFE(
    Y: ArrayLike,
    W: ArrayLike,
    treated_units: Sequence[int],
    treated_periods: int = 10,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> float:
    """
    Synthetic Difference-in-Differences (SDID) estimator.

    Weighted two-way fixed-effects regression using ``SDID_weights`` as the
    unit/time weighting scheme.

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Outcome matrix.
    W : array_like of shape (N, T)
        Treatment indicator matrix.
    treated_units : sequence of int
        Row indices of treated units.
    treated_periods : int, default=10
        Number of final columns treated as the post-treatment block.
    solver : str or None, default=None
        CVXPY solver name. Defaults to "OSQP".
    verbose : bool, default=False
        Passed to ``cvxpy.Problem.solve``.

    Returns
    -------
    float
        Estimated treatment-effect coefficient ``tau``.
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    N, T = _check_panel(Y, W)

    delta_unit, delta_time = SDID_weights(Y, treated_units, treated_periods, solver=solver, verbose=verbose)
    delta = np.outer(delta_unit, delta_time)

    unit_effects = cp.Variable((1, N))
    time_effects = cp.Variable((1, T))
    unit_factor = cp.kron(np.ones((T, 1)), unit_effects).T
    time_factor = cp.kron(np.ones((N, 1)), time_effects)
    mu = cp.Variable()
    tau = cp.Variable()

    objective = cp.sum_squares(cp.multiply(Y - mu - unit_factor - time_factor - W * tau, delta))
    prob = cp.Problem(cp.Minimize(objective))
    _solve(prob, solver, "OSQP", verbose)

    tau_hat = tau.value
    if tau_hat is None or not np.isfinite(tau_hat):
        raise RuntimeError("Optimization did not return a valid tau.")
    return float(tau_hat)
