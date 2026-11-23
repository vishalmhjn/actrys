"""Bayesian optimization helper (wraps ``bayes_opt``) with persistent JSON logs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs


def bayes_maximize(
    f: Callable[..., float],
    pbounds: dict[str, tuple[float, float]],
    n_init: int,
    n_iter: int,
    random_state: int,
    log_path: Path,
) -> dict:
    """Maximize ``f`` over ``pbounds``; returns ``{"target": ..., "params": {...}}``.

    Points from an existing ``log_path`` (previous runs) are registered first,
    and all new probes are logged there.
    """
    optimizer = BayesianOptimization(
        f=f, pbounds=pbounds, random_state=random_state, allow_duplicate_points=True
    )
    log_path = Path(log_path)
    if log_path.exists():
        load_logs(optimizer, logs=[str(log_path)])
    optimizer.subscribe(Events.OPTIMIZATION_STEP, JSONLogger(path=str(log_path)))
    optimizer.maximize(init_points=n_init, n_iter=n_iter)
    return optimizer.max
