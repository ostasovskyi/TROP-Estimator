from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import cvxpy as cp


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def TROP_TWFE_average(
    Y: ArrayLike,
    W: ArrayLike,
    treated_units: Sequence[int],
    lambda_unit: float,
    lambda_time: float,
    lambda_nn: float,
    treated_periods: int = 10,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> float:
    """
    Compute the TROP treatment effect with unit/time weighting and optional low-rank
    outcome model.

    Parameters
    ----------
    Y : array_like of shape (N, T)
        Outcome matrix.
    W : array_like of shape (N, T)
        Treatment indicator (often binary). The estimator uses ``W`` as provided;
        ``treated_periods`` is used only to construct weights/masks, not to infer
        treatment timing.
    treated_units : sequence of int
        Row indices of treated units used to form the reference (average) treated
        trajectory for unit-distance weighting.
    lambda_unit : float
        Nonnegative decay parameter for unit weights: ``exp(-lambda_unit * dist_unit)``.
    lambda_time : float
        Nonnegative decay parameter for time weights: ``exp(-lambda_time * dist_time)``.
    lambda_nn : float
        Nuclear-norm penalty weight for the low-rank component ``L``. Use
        ``np.inf`` to disable the low-rank adjustment (i.e., omit ``L``).
    treated_periods : int, default=10
        Number of final columns treated as the "post/tail block" for constructing
        (a) the pre-period mask (all but last ``treated_periods`` columns) used in
        unit distances, and (b) the time-distance center.
    solver : str or None, default=None
        CVXPY solver name. If None, uses "SCS" when ``lambda_nn`` is finite and
        "OSQP" when ``lambda_nn`` is infinite.
    verbose : bool, default=False
        Passed to ``cvxpy.Problem.solve``.

    Returns
    -------
    float
        Estimated treatment-effect parameter ``tau`` from the weighted TWFE objective.

    Raises
    ------
    ValueError
        If input shapes are inconsistent or tuning parameters are invalid.
    RuntimeError
        If the optimization fails to produce a finite ``tau``.
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)

    if Y.ndim != 2 or W.ndim != 2:
        raise ValueError(f"Y and W must be 2D arrays. Got Y.ndim={Y.ndim}, W.ndim={W.ndim}.")
    if Y.shape != W.shape:
        raise ValueError(f"Y and W must have the same shape. Got Y={Y.shape}, W={W.shape}.")

    N, T = Y.shape

    if not isinstance(treated_periods, int) or treated_periods <= 0:
        raise ValueError("treated_periods must be a positive integer.")
    if treated_periods >= T:
        raise ValueError(f"treated_periods must be < T. Got treated_periods={treated_periods}, T={T}.")

    treated_units_arr = np.asarray(treated_units, dtype=int)
    if treated_units_arr.size == 0:
        raise ValueError("treated_units must contain at least one unit index.")
    if np.any(treated_units_arr < 0) or np.any(treated_units_arr >= N):
        raise ValueError(f"treated_units contains out-of-range indices for N={N}: {treated_units_arr}")

    if lambda_unit < 0 or lambda_time < 0:
        raise ValueError("lambda_unit and lambda_time should be nonnegative.")

    # ---------------------------------------------------------------------
    # Distance-based time weights
    # ---------------------------------------------------------------------
    # Distance to the center of the treated block near the end of the panel.
    # dist_time = abs(arange(T) - (T - treated_periods/2))
    center = T - treated_periods / 2.0
    dist_time = np.abs(np.arange(T, dtype=float) - center)

    # ---------------------------------------------------------------------
    # Distance-based unit weights
    # ---------------------------------------------------------------------
    average_treated = np.mean(Y[treated_units_arr, :], axis=0)

    # Pre-period mask: 1 in pre, 0 in treated/post
    mask = np.ones((N, T), dtype=float)
    mask[:, -treated_periods:] = 0.0

    # RMS distance to average treated trajectory over pre-periods
    # dist_unit[i] = sqrt( sum_pre (avg_tr - Y_i)^2 / (#pre) )
    A = np.sum(((average_treated - Y) ** 2) * mask, axis=1)
    B = np.sum(mask, axis=1)

    if np.any(B == 0):
        raise ValueError(
            "Pre-period mask has zero pre-periods for at least one unit."
        )

    dist_unit = np.sqrt(A / B)

    # Convert distances to weights
    delta_unit = np.exp(-lambda_unit * dist_unit)          # shape (N,)
    delta_time = np.exp(-lambda_time * dist_time)          # shape (T,)
    delta = np.outer(delta_unit, delta_time)               # shape (N, T)

    # ---------------------------------------------------------------------
    # CVXPY problem: weighted TWFE
    # ---------------------------------------------------------------------
    unit_effects = cp.Variable((1, N))
    time_effects = cp.Variable((1, T))
    mu = cp.Variable()     # intercept
    tau = cp.Variable()    # treatment effect

    # Broadcast TWFE components to N x T
    unit_factor = cp.kron(np.ones((T, 1)), unit_effects).T
    time_factor = cp.kron(np.ones((N, 1)), time_effects)

    is_low_rank = not math.isinf(float(lambda_nn))

    if is_low_rank:
        L = cp.Variable((N, T))
        residual = Y - mu - unit_factor - time_factor - L - W * tau
        weights = np.multiply(1.0 - W, delta)
        loss = cp.sum(cp.multiply(weights, cp.square(residual))) + float(lambda_nn) * cp.norm(L, "nuc")
        default_solver = "SCS"  # robust choice for nuclear norm problems
    else:
        residual = Y - mu - unit_factor - time_factor - W * tau
        weights = np.multiply(1.0 - W, delta)
        loss = cp.sum(cp.multiply(weights, cp.square(residual)))
        default_solver = "OSQP"  # fast for pure quadratic objective

    prob = cp.Problem(cp.Minimize(loss))

    chosen_solver = solver or default_solver
    prob.solve(solver=chosen_solver, verbose=verbose)

    if tau.value is None or not np.isfinite(tau.value):
        raise RuntimeError(
            "Optimization did not return a valid tau. "
            f"Solver={chosen_solver}, status={prob.status}."
        )

    return float(tau.value)