"""Stage 7.5 runtime model loader for BioSynth-EDU.

This module connects the Stage 7.4 ``model_selection.json`` artifact with the
working inference core. It prefers selected v2 models when a selection artifact
is available and falls back to legacy ``models/rf_*.joblib`` paths otherwise.

Design goals
------------
* Do not require the main Streamlit app to know where each model lives.
* Do not load models disabled by Stage 7.4 policy, for example weak Clint v2.
* Preserve backward compatibility when ``model_selection.json`` does not exist.
* Expose thresholds, roles and QA/selection metadata to ``bbb_calculation.py``.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import joblib

logger = logging.getLogger(__name__)

ENV_MODEL_SELECTION_PATH = "BIOSYNTH_MODEL_SELECTION_PATH"

DEFAULT_SELECTION_CANDIDATES = (
    "models/v2/model_selection.json",
    "models/v2_experiment/model_selection.json",
)

LEGACY_MODEL_PATHS = {
    "rf_pka_model": "models/rf_pka_model.joblib",
    "rf_pgp_model": "models/rf_pgp_model.joblib",
    "rf_clint_model": "models/rf_clint_model.joblib",
    "rf_catmos_model": "models/rf_catmos_model.joblib",
    "rf_caco2_model": "models/rf_caco2_model.joblib",
    "rf_bbb_model": "models/rf_bbb_model.joblib",
}

SHORT_TO_LEGACY = {
    "pka": "rf_pka_model",
    "pgp": "rf_pgp_model",
    "clint": "rf_clint_model",
    "catmos": "rf_catmos_model",
    "caco2": "rf_caco2_model",
    "bbb_rf": "rf_bbb_model",
}

LEGACY_TO_SHORT = {value: key for key, value in SHORT_TO_LEGACY.items()}


@dataclass(frozen=True)
class RuntimeModelPlan:
    legacy_model_name: str
    short_label: str
    model_path: Optional[str]
    resolved_model_path: Optional[str]
    source: str  # selection_v2 | legacy_fallback | disabled_by_selection | missing_selection_entry
    load_for_runtime: bool
    selection_status: str
    runtime_role: str
    threshold: Optional[float]
    feature_kind: Optional[str]
    task_type: Optional[str]
    unit_status: Optional[str] = None
    reason: Optional[Mapping[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_model_selection_path(explicit_path: str | Path | None = None) -> Optional[Path]:
    """Return the first available model selection artifact, if any."""
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None

    env_path = os.environ.get(ENV_MODEL_SELECTION_PATH)
    if env_path:
        path = Path(env_path)
        return path if path.exists() else None

    for candidate in DEFAULT_SELECTION_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    return None


def resolve_runtime_path(raw_path: Any, *, selection_path: Path | None = None) -> Optional[Path]:
    """Resolve model paths stored in model_selection/model_registry files.

    Stage 7.3 artifacts usually store paths such as
    ``models/v2_experiment/rf_pgp_model_v2.joblib``. This is correct when the
    process runs from the project root. For tests or copied artifacts, the same
    file may live next to ``model_selection.json``; this function supports both.
    """
    if not raw_path:
        return None

    path = Path(str(raw_path))
    if path.is_absolute():
        return path

    # Normal project-root case.
    if path.exists():
        return path

    if selection_path is not None:
        parent = selection_path.parent
        candidates = [
            parent / path.name,
            parent / path,
            parent.parent / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    return path


def _threshold_from_entry(entry: Mapping[str, Any]) -> Optional[float]:
    threshold = entry.get("threshold")
    if isinstance(threshold, Mapping):
        return _safe_float(threshold.get("threshold"))
    return _safe_float(threshold)


def _unit_status_from_entry(legacy_name: str, entry: Mapping[str, Any]) -> Optional[str]:
    if legacy_name == "rf_catmos_model" and entry.get("runtime_role") == "score_only_units_unverified":
        return "requires_validation"
    return None


def _selection_model_entry(selection: Mapping[str, Any], legacy_name: str) -> Optional[Dict[str, Any]]:
    item = selection.get("models", {}).get(legacy_name)
    return dict(item) if isinstance(item, Mapping) else None


def build_runtime_plan(selection_path: str | Path | None = None) -> Dict[str, RuntimeModelPlan]:
    """Build a runtime loading plan from Stage 7.4 selection or legacy fallback."""
    path = find_model_selection_path(selection_path)
    selection: Dict[str, Any] = {}
    if path is not None:
        try:
            selection = _read_json(path)
        except Exception as exc:
            logger.warning("Could not read model selection %s: %s", path, exc)
            selection = {}
            path = None

    plan: Dict[str, RuntimeModelPlan] = {}
    for legacy_name, legacy_path in LEGACY_MODEL_PATHS.items():
        short = LEGACY_TO_SHORT[legacy_name]
        entry = _selection_model_entry(selection, legacy_name) if selection else None

        if entry is None:
            resolved = resolve_runtime_path(legacy_path, selection_path=path)
            plan[legacy_name] = RuntimeModelPlan(
                legacy_model_name=legacy_name,
                short_label=short,
                model_path=legacy_path,
                resolved_model_path=str(resolved) if resolved is not None else None,
                source="legacy_fallback" if path is None else "missing_selection_entry",
                load_for_runtime=True,
                selection_status="legacy_fallback" if path is None else "missing_selection_entry",
                runtime_role="legacy_runtime",
                threshold=None,
                feature_kind=None,
                task_type=None,
            )
            continue

        load_for_runtime = bool(entry.get("load_for_runtime"))
        raw_model_path = entry.get("model_path")
        resolved = resolve_runtime_path(raw_model_path, selection_path=path)
        source = "selection_v2" if load_for_runtime else "disabled_by_selection"
        plan[legacy_name] = RuntimeModelPlan(
            legacy_model_name=legacy_name,
            short_label=short,
            model_path=str(raw_model_path) if raw_model_path else None,
            resolved_model_path=str(resolved) if resolved is not None else None,
            source=source,
            load_for_runtime=load_for_runtime,
            selection_status=str(entry.get("selection_status", "unknown")),
            runtime_role=str(entry.get("runtime_role", "unknown")),
            threshold=_threshold_from_entry(entry),
            feature_kind=entry.get("feature_kind"),
            task_type=entry.get("task_type"),
            unit_status=_unit_status_from_entry(legacy_name, entry),
            reason=entry.get("reason") if isinstance(entry.get("reason"), Mapping) else None,
        )

    return plan


def _load_one_model(plan: RuntimeModelPlan) -> tuple[Any, Dict[str, Any]]:
    if not plan.load_for_runtime:
        return None, {
            "status": "disabled_by_selection",
            "path": plan.resolved_model_path,
            "source": plan.source,
            "selection_status": plan.selection_status,
            "runtime_role": plan.runtime_role,
            "error": None,
        }

    if not plan.resolved_model_path:
        return None, {
            "status": "missing_path",
            "path": None,
            "source": plan.source,
            "selection_status": plan.selection_status,
            "runtime_role": plan.runtime_role,
            "error": "Model path is missing",
        }

    path = Path(plan.resolved_model_path)
    if not path.exists():
        return None, {
            "status": "missing_or_failed",
            "path": str(path),
            "source": plan.source,
            "selection_status": plan.selection_status,
            "runtime_role": plan.runtime_role,
            "error": f"Model file not found: {path}",
        }

    try:
        model = joblib.load(path)
        return model, {
            "status": "loaded",
            "path": str(path),
            "source": plan.source,
            "selection_status": plan.selection_status,
            "runtime_role": plan.runtime_role,
            "error": None,
        }
    except Exception as exc:
        return None, {
            "status": "missing_or_failed",
            "path": str(path),
            "source": plan.source,
            "selection_status": plan.selection_status,
            "runtime_role": plan.runtime_role,
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_runtime_context(selection_path: str | Path | None = None) -> Dict[str, Any]:
    """Load selected runtime models and return a serializable context."""
    selected_path = find_model_selection_path(selection_path)
    selection: Dict[str, Any] = {}
    if selected_path is not None:
        try:
            selection = _read_json(selected_path)
        except Exception as exc:
            logger.warning("Failed to read model selection %s: %s", selected_path, exc)
            selected_path = None
            selection = {}

    plan = build_runtime_plan(selected_path)
    models: Dict[str, Any] = {}
    load_status: Dict[str, Dict[str, Any]] = {}

    for legacy_name, item in plan.items():
        model, status = _load_one_model(item)
        models[legacy_name] = model
        load_status[legacy_name] = status

    load_status_by_short = {
        LEGACY_TO_SHORT[legacy_name]: status for legacy_name, status in load_status.items()
    }

    return {
        "schema_version": "runtime_models_v1.0",
        "selection_path": str(selected_path) if selected_path is not None else None,
        "selection_available": selected_path is not None and bool(selection),
        "selection": selection,
        "plan": {name: item.to_dict() for name, item in plan.items()},
        "models": models,
        "load_status": load_status,
        "load_status_by_short": load_status_by_short,
        "primary_runtime": selection.get("primary_runtime", {}) if selection else {},
    }


@lru_cache(maxsize=4)
def _cached_runtime_context(selection_path_string: str | None = None) -> Dict[str, Any]:
    return load_runtime_context(selection_path_string)


def get_runtime_context(selection_path: str | Path | None = None, *, refresh: bool = False) -> Dict[str, Any]:
    if refresh:
        _cached_runtime_context.cache_clear()
    return _cached_runtime_context(str(selection_path) if selection_path is not None else None)


def refresh_runtime_context() -> None:
    _cached_runtime_context.cache_clear()


def get_runtime_model(legacy_name: str, context: Optional[Mapping[str, Any]] = None) -> Any:
    ctx = context or get_runtime_context()
    return ctx.get("models", {}).get(legacy_name)


def get_runtime_entry(legacy_name: str, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    ctx = context or get_runtime_context()
    entry = ctx.get("plan", {}).get(legacy_name)
    return deepcopy(entry) if isinstance(entry, Mapping) else {}


def get_runtime_threshold(legacy_name: str, context: Optional[Mapping[str, Any]] = None) -> Optional[float]:
    entry = get_runtime_entry(legacy_name, context=context)
    return _safe_float(entry.get("threshold"))


def is_runtime_disabled(legacy_name: str, context: Optional[Mapping[str, Any]] = None) -> bool:
    entry = get_runtime_entry(legacy_name, context=context)
    return entry.get("source") == "disabled_by_selection" or entry.get("load_for_runtime") is False


def get_runtime_load_status_by_short_label(context: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    ctx = context or get_runtime_context()
    return deepcopy(ctx.get("load_status_by_short", {}))


def build_runtime_status_summary(context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    ctx = context or get_runtime_context()
    plan = ctx.get("plan", {})
    return {
        "schema_version": "runtime_status_summary_v1.0",
        "selection_path": ctx.get("selection_path"),
        "selection_available": ctx.get("selection_available"),
        "primary_runtime": deepcopy(ctx.get("primary_runtime", {})),
        "models": {
            legacy_name: {
                "short_label": item.get("short_label"),
                "source": item.get("source"),
                "selection_status": item.get("selection_status"),
                "runtime_role": item.get("runtime_role"),
                "load_for_runtime": item.get("load_for_runtime"),
                "threshold": item.get("threshold"),
                "feature_kind": item.get("feature_kind"),
                "task_type": item.get("task_type"),
                "unit_status": item.get("unit_status"),
                "load_status": ctx.get("load_status", {}).get(legacy_name, {}),
            }
            for legacy_name, item in plan.items()
        },
    }
