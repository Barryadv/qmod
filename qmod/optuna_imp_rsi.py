from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import json
import optuna

# 👇 Use package-relative imports so this works inside qmod/
try:
    from . import config as cfg_mod          # optional / best-effort
except Exception:  # pragma: no cover
    cfg_mod = None  # type: ignore

try:
    from .paths import artifacts_dir         # stable output dir
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Cannot import artifacts_dir from qmod.paths: {e!r}") from e


def _get_optuna_settings() -> dict[str, Any]:
    """
    Best-effort extraction of optuna_settings from cfg_mod.CONFIG,
    but safe even if CONFIG / init_CONFIG don't exist.
    """
    if cfg_mod is None:
        return {}

    cfg_now: dict[str, Any] = {}

    # Old-style CONFIG global
    if hasattr(cfg_mod, "CONFIG"):
        config_obj = getattr(cfg_mod, "CONFIG")
        # init_CONFIG() if present and CONFIG is None
        if config_obj is None and hasattr(cfg_mod, "init_CONFIG"):
            try:
                cfg_mod.init_CONFIG()  # type: ignore[attr-defined]
                config_obj = getattr(cfg_mod, "CONFIG")
            except Exception:
                config_obj = None

        if isinstance(config_obj, dict):
            cfg_now = dict(config_obj)

    # Extract optuna_settings if present
    return dict(cfg_now.get("optuna_settings", {}))


def optimize_with_optuna(
    objective_func: Callable[[optuna.trial.Trial], float],
    *,
    n_trials: Optional[int] = None,
    vis_dir: str | Path | None = None,
    study_name: Optional[str] = None,
    direction: str = "maximize",
    sampler: optuna.samplers.BaseSampler | None = None,
) -> dict[str, Any]:
    """
    Run an Optuna study using the provided objective function (RSI).
    Returns: best_params (dict)

    Designed to be called from run_full_composite, e.g.:

        _optuna_imp_rsi.optimize_with_optuna(objective_rsi,
                                             n_trials=n_trials,
                                             vis_dir=...)

    - Will read `optuna_settings` from cfg_mod.CONFIG if available.
    - Default output dir: artifacts_dir("vis_rsi", "optuna")
    - Single-process (n_jobs=1) for API / CLI safety.
    """
    optuna_settings = _get_optuna_settings()

    # Trials: explicit > YAML > default 50
    if n_trials is None:
        n_trials = int(optuna_settings.get("n_trials", 50))

    # Output folder
    out_dir = Path(vis_dir) if vis_dir else artifacts_dir("vis_rsi", "optuna")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sampler & study
    sampler = sampler or optuna.samplers.TPESampler(
        seed=optuna_settings.get("seed")
    )
    study = optuna.create_study(
        study_name=study_name,
        direction=direction or optuna_settings.get("direction", "maximize"),
        sampler=sampler,
    )

    # Keep API path stable
    study.optimize(
        objective_func,
        n_trials=int(n_trials),
        n_jobs=1,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_value = study.best_value

    # Persist artifacts with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"{timestamp}_best_params.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    (out_dir / f"{timestamp}_best_trial_repr.txt").write_text(
        repr(study.best_trial), encoding="utf-8"
    )
    (out_dir / f"{timestamp}_study_summary.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "direction": study.direction.name,
                "best_value": best_value,
                "best_params": best_params,
                "n_trials": int(n_trials),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n✅ Best Score: {best_value:.6f}")
    print("✅ Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params
