"""Stage 7.4 model selection layer for BioSynth-EDU.

This module consumes the Stage 7.3 training registry/model cards and produces a
runtime-oriented model selection contract. It does not train models and does not
modify joblib files. Its job is to decide how each v2 artifact should be used:
primary, supplementary, helper, score-only, or disabled by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

SCHEMA_VERSION = "model_selection_v1.0"
DEFAULT_REGISTRY_PATH = "models/v2_experiment/model_registry.json"


@dataclass(frozen=True)
class MetricThresholds:
    """Simple policy thresholds used by the default selection rules."""

    pgp_min_roc_auc: float = 0.85
    pgp_min_mcc: float = 0.50
    pgp_min_balanced_accuracy: float = 0.75

    bbb_rf_min_roc_auc: float = 0.80
    bbb_rf_min_balanced_accuracy: float = 0.70
    bbb_rf_max_false_positive_rate_for_primary: float = 0.25

    pka_min_r2: float = 0.50
    pka_max_rmse: float = 2.20

    caco2_min_r2: float = 0.45
    caco2_max_rmse: float = 0.75

    catmos_min_r2_for_score_only: float = 0.40
    catmos_max_rmse_for_score_only: float = 0.50

    clint_min_roc_auc: float = 0.70
    clint_min_pr_auc: float = 0.45
    clint_min_mcc: float = 0.30


DEFAULT_POLICY = MetricThresholds()


MODEL_LABELS = {
    "rf_pgp_model": {
        "ru": "P-gp RF v2",
        "kk": "P-gp RF v2",
        "en": "P-gp RF v2",
    },
    "rf_bbb_model": {
        "ru": "BBB RF v2",
        "kk": "BBB RF v2",
        "en": "BBB RF v2",
    },
    "rf_pka_model": {
        "ru": "pKa RF v2",
        "kk": "pKa RF v2",
        "en": "pKa RF v2",
    },
    "rf_caco2_model": {
        "ru": "Caco-2 RF v2",
        "kk": "Caco-2 RF v2",
        "en": "Caco-2 RF v2",
    },
    "rf_catmos_model": {
        "ru": "CATMoS RF v2",
        "kk": "CATMoS RF v2",
        "en": "CATMoS RF v2",
    },
    "rf_clint_model": {
        "ru": "Clint RF v2",
        "kk": "Clint RF v2",
        "en": "Clint RF v2",
    },
    "gupta_bbb_corrected": {
        "ru": "Исправленная формула Gupta BBB",
        "kk": "Түзетілген Gupta BBB формуласы",
        "en": "Corrected Gupta BBB formula",
    },
}


ROLE_LABELS = {
    "primary_formula": {
        "ru": "Основной deterministic BBB-блок",
        "kk": "Негізгі deterministic BBB блогы",
        "en": "Primary deterministic BBB block",
    },
    "primary_efflux": {
        "ru": "Основная P-gp ML-модель",
        "kk": "Негізгі P-gp ML моделі",
        "en": "Primary P-gp ML model",
    },
    "supplementary_bbb_ml": {
        "ru": "Дополнительный BBB ML-сигнал",
        "kk": "Қосымша BBB ML сигналы",
        "en": "Supplementary BBB ML signal",
    },
    "helper_pka_experimental": {
        "ru": "Вспомогательный pKa-сигнал",
        "kk": "Көмекші pKa сигналы",
        "en": "Auxiliary pKa signal",
    },
    "auxiliary_admet_experimental": {
        "ru": "Вспомогательная ADMET-модель",
        "kk": "Көмекші ADMET моделі",
        "en": "Auxiliary ADMET model",
    },
    "score_only_units_unverified": {
        "ru": "Score-only: единицы не подтверждены",
        "kk": "Score-only: бірліктер расталмаған",
        "en": "Score-only: units not verified",
    },
    "disabled_by_default": {
        "ru": "Отключена по умолчанию",
        "kk": "Әдепкіде өшірілген",
        "en": "Disabled by default",
    },
}


STATUS_LABELS = {
    "selected_primary": {
        "ru": "Выбрана как основная",
        "kk": "Негізгі ретінде таңдалды",
        "en": "Selected as primary",
    },
    "selected_validated": {
        "ru": "Выбрана: хорошая валидация",
        "kk": "Таңдалды: валидациясы жақсы",
        "en": "Selected: good validation",
    },
    "selected_supplementary_caution": {
        "ru": "Выбрана как дополнительная, осторожная интерпретация",
        "kk": "Қосымша ретінде таңдалды, сақтықпен түсіндіру керек",
        "en": "Selected as supplementary, interpret with caution",
    },
    "selected_experimental": {
        "ru": "Выбрана как экспериментальная вспомогательная модель",
        "kk": "Эксперименттік көмекші модель ретінде таңдалды",
        "en": "Selected as experimental auxiliary model",
    },
    "selected_score_only": {
        "ru": "Выбрана только как score; физические единицы не подтверждены",
        "kk": "Тек score ретінде таңдалды; физикалық бірліктер расталмаған",
        "en": "Selected as score only; physical units are not verified",
    },
    "disabled_weak_validation": {
        "ru": "Отключена по умолчанию из-за слабой валидации",
        "kk": "Әлсіз валидацияға байланысты әдепкіде өшірілген",
        "en": "Disabled by default because validation is weak",
    },
    "missing_or_failed": {
        "ru": "Не выбрана: нет артефакта или ошибка",
        "kk": "Таңдалмады: артефакт жоқ немесе қате бар",
        "en": "Not selected: missing artifact or error",
    },
}


SEVERITY_BY_SELECTION_STATUS = {
    "selected_primary": "ok",
    "selected_validated": "ok",
    "selected_supplementary_caution": "warning",
    "selected_experimental": "warning",
    "selected_score_only": "warning",
    "disabled_weak_validation": "danger",
    "missing_or_failed": "danger",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        x = float(value)
    except Exception:
        return default
    return x


def _metrics(entry: Mapping[str, Any], card: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if card and isinstance(card.get("metrics"), Mapping):
        return dict(card.get("metrics", {}))
    if isinstance(entry.get("metrics"), Mapping):
        return dict(entry.get("metrics", {}))
    return {}


def _card_path_from_entry(entry: Mapping[str, Any], registry_path: Path) -> Optional[Path]:
    raw = entry.get("model_card_path")
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path
    # Prefer path relative to current working directory; fallback relative to registry parent.
    if path.exists():
        return path
    candidate = registry_path.parent / path.name
    if candidate.exists():
        return candidate
    return path


def _load_card(entry: Mapping[str, Any], registry_path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    path = _card_path_from_entry(entry, registry_path)
    if path is None:
        return None, "model_card_path is missing"
    try:
        if not path.exists():
            return None, f"model card not found: {path}"
        return load_json(path), None
    except Exception as exc:
        return None, f"could not read model card {path}: {type(exc).__name__}: {exc}"


def _false_positive_rate(metrics: Mapping[str, Any]) -> Optional[float]:
    fp = _safe_float(metrics.get("fp"))
    tn = _safe_float(metrics.get("tn"))
    if fp is None or tn is None or fp + tn <= 0:
        return None
    return fp / (fp + tn)


def _default_reason_ru(legacy_name: str, metrics: Mapping[str, Any]) -> str:
    if legacy_name == "rf_pgp_model":
        return "P-gp v2 показывает устойчивые classification-метрики и используется как основной блок эффлюкса."
    if legacy_name == "rf_bbb_model":
        return "BBB RF v2 имеет сильный ROC-AUC, но используется как дополнительный сигнал рядом с corrected Gupta, а не как основной вердикт."
    if legacy_name == "rf_pka_model":
        return "pKa v2 полезна как приближённый basic pKa, но RMSE остаётся около 1.9 pKa units."
    if legacy_name == "rf_caco2_model":
        return "Caco-2 v2 имеет умеренное качество и используется как вспомогательный ADMET-показатель."
    if legacy_name == "rf_catmos_model":
        return "CATMoS v2 можно использовать только как score, потому что шкала consensus_LD50 выглядит log/transformed и единицы требуют подтверждения."
    if legacy_name == "rf_clint_model":
        return "Clint v2 показывает слабые classification-метрики и отключается по умолчанию."
    return "Runtime decision was assigned by Stage 7.4 model selection policy."


def _default_reason_en(legacy_name: str, metrics: Mapping[str, Any]) -> str:
    if legacy_name == "rf_pgp_model":
        return "P-gp v2 shows stable classification metrics and is used as the primary efflux model."
    if legacy_name == "rf_bbb_model":
        return "BBB RF v2 has strong ROC-AUC, but is used as a supplementary signal next to corrected Gupta, not as the primary verdict."
    if legacy_name == "rf_pka_model":
        return "pKa v2 is useful as an approximate basic pKa helper, but RMSE remains around 1.9 pKa units."
    if legacy_name == "rf_caco2_model":
        return "Caco-2 v2 has moderate quality and is used as an auxiliary ADMET endpoint."
    if legacy_name == "rf_catmos_model":
        return "CATMoS v2 is score-only because consensus_LD50 looks log/transformed and units require confirmation."
    if legacy_name == "rf_clint_model":
        return "Clint v2 shows weak classification metrics and is disabled by default."
    return "Runtime decision was assigned by the Stage 7.4 model selection policy."


def _default_reason_kk(legacy_name: str, metrics: Mapping[str, Any]) -> str:
    if legacy_name == "rf_pgp_model":
        return "P-gp v2 classification метрикалары тұрақты және негізгі эффлюкс моделі ретінде қолданылады."
    if legacy_name == "rf_bbb_model":
        return "BBB RF v2 ROC-AUC бойынша жақсы, бірақ негізгі шешім емес, corrected Gupta жанындағы қосымша сигнал ретінде қолданылады."
    if legacy_name == "rf_pka_model":
        return "pKa v2 жуық basic pKa көмекші сигналы ретінде пайдалы, бірақ RMSE шамамен 1.9 pKa бірлігі."
    if legacy_name == "rf_caco2_model":
        return "Caco-2 v2 сапасы орташа және көмекші ADMET көрсеткіші ретінде қолданылады."
    if legacy_name == "rf_catmos_model":
        return "CATMoS v2 тек score ретінде қолданылады, себебі consensus_LD50 шкаласы log/transformed болып көрінеді және бірліктерді растау қажет."
    if legacy_name == "rf_clint_model":
        return "Clint v2 classification метрикалары әлсіз, сондықтан әдепкіде өшіріледі."
    return "Runtime шешімі Stage 7.4 model selection policy арқылы берілді."


def corrected_gupta_entry() -> Dict[str, Any]:
    return {
        "model_name": "gupta_bbb_corrected",
        "legacy_model_name": "gupta_bbb_corrected",
        "runtime_role": "primary_formula",
        "selection_status": "selected_primary",
        "severity": "ok",
        "load_for_runtime": False,
        "use_in_final_cns_decision": True,
        "show_in_ui": True,
        "model_path": None,
        "feature_kind": "gupta_descriptors",
        "task_type": "formula",
        "threshold": 3.0,
        "metrics": {},
        "labels": MODEL_LABELS["gupta_bbb_corrected"],
        "role_labels": ROLE_LABELS["primary_formula"],
        "status_labels": STATUS_LABELS["selected_primary"],
        "reason": {
            "ru": "Исправленная формула Gupta использует p_mwhbn вместо сырого MWHBN и остаётся основным deterministic BBB score.",
            "kk": "Түзетілген Gupta формуласы raw MWHBN орнына p_mwhbn қолданады және негізгі deterministic BBB score болып қалады.",
            "en": "The corrected Gupta formula uses p_mwhbn instead of raw MWHBN and remains the primary deterministic BBB score.",
        },
        "warnings": [],
        "recommendations": [],
    }


def select_one_model(
    legacy_name: str,
    entry: Mapping[str, Any],
    card: Optional[Mapping[str, Any]] = None,
    card_error: Optional[str] = None,
    policy: MetricThresholds = DEFAULT_POLICY,
) -> Dict[str, Any]:
    """Apply Stage 7.4 selection rules to one registry entry."""

    metrics = _metrics(entry, card)
    qa_status = str(entry.get("qa_status") or (card or {}).get("qa_status") or "unknown")
    task_type = str(entry.get("task_type") or (card or {}).get("task_type") or "unknown")
    feature_kind = str(entry.get("feature_kind") or (card or {}).get("feature_kind") or "unknown")
    model_name = str(entry.get("model_name") or (card or {}).get("model_name") or legacy_name)

    threshold = metrics.get("threshold")
    if card and isinstance(card.get("threshold"), Mapping):
        threshold = card["threshold"].get("threshold", threshold)

    warnings: List[str] = []
    recommendations: List[str] = []
    if card_error:
        warnings.append(card_error)
    if card:
        warnings.extend(str(x) for x in card.get("issues", []) if x)
        recommendations.extend(str(x) for x in card.get("recommendations", []) if x)

    model_path = entry.get("model_path") or (card or {}).get("model_path")

    runtime_role = "auxiliary_admet_experimental"
    selection_status = "selected_experimental"
    load_for_runtime = True
    use_in_final_cns_decision = False
    show_in_ui = True

    if not model_path:
        runtime_role = "disabled_by_default"
        selection_status = "missing_or_failed"
        load_for_runtime = False
        show_in_ui = False
        warnings.append("model_path is missing")

    elif legacy_name == "rf_pgp_model":
        roc_auc = _safe_float(metrics.get("roc_auc"), 0.0) or 0.0
        mcc = _safe_float(metrics.get("mcc"), 0.0) or 0.0
        bal = _safe_float(metrics.get("balanced_accuracy"), 0.0) or 0.0
        runtime_role = "primary_efflux"
        use_in_final_cns_decision = True
        if roc_auc >= policy.pgp_min_roc_auc and mcc >= policy.pgp_min_mcc and bal >= policy.pgp_min_balanced_accuracy:
            selection_status = "selected_validated"
        else:
            selection_status = "selected_supplementary_caution"
            warnings.append("P-gp metrics are below the default validated threshold.")

    elif legacy_name == "rf_bbb_model":
        runtime_role = "supplementary_bbb_ml"
        use_in_final_cns_decision = False
        roc_auc = _safe_float(metrics.get("roc_auc"), 0.0) or 0.0
        bal = _safe_float(metrics.get("balanced_accuracy"), 0.0) or 0.0
        fpr = _false_positive_rate(metrics)
        if roc_auc >= policy.bbb_rf_min_roc_auc and bal >= policy.bbb_rf_min_balanced_accuracy:
            selection_status = "selected_supplementary_caution"
            if fpr is not None and fpr <= policy.bbb_rf_max_false_positive_rate_for_primary:
                recommendations.append("False-positive rate improved after threshold tuning; keep as supplementary signal.")
        else:
            selection_status = "selected_experimental"
            warnings.append("BBB RF metrics are not strong enough for supplementary use.")

    elif legacy_name == "rf_pka_model":
        runtime_role = "helper_pka_experimental"
        use_in_final_cns_decision = True
        r2 = _safe_float(metrics.get("r2"), 0.0) or 0.0
        rmse = _safe_float(metrics.get("rmse"), 999.0) or 999.0
        if r2 >= policy.pka_min_r2 and rmse <= policy.pka_max_rmse:
            selection_status = "selected_experimental"
            recommendations.append("Use as approximate predicted basic pKa, not as exact acid/base-aware pKa.")
        else:
            selection_status = "disabled_weak_validation"
            load_for_runtime = False
            use_in_final_cns_decision = False
            warnings.append("pKa metrics are too weak for runtime use.")

    elif legacy_name == "rf_caco2_model":
        runtime_role = "auxiliary_admet_experimental"
        r2 = _safe_float(metrics.get("r2"), 0.0) or 0.0
        rmse = _safe_float(metrics.get("rmse"), 999.0) or 999.0
        if r2 >= policy.caco2_min_r2 and rmse <= policy.caco2_max_rmse:
            selection_status = "selected_experimental"
        else:
            selection_status = "disabled_weak_validation"
            load_for_runtime = False
            warnings.append("Caco-2 metrics are below auxiliary-use threshold.")

    elif legacy_name == "rf_catmos_model":
        runtime_role = "score_only_units_unverified"
        use_in_final_cns_decision = False
        r2 = _safe_float(metrics.get("r2"), 0.0) or 0.0
        rmse = _safe_float(metrics.get("rmse"), 999.0) or 999.0
        if r2 >= policy.catmos_min_r2_for_score_only and rmse <= policy.catmos_max_rmse_for_score_only:
            selection_status = "selected_score_only"
            recommendations.append("Display as CATMoS score only; do not display as LD50 mg/kg until target scale is confirmed.")
        else:
            selection_status = "disabled_weak_validation"
            load_for_runtime = False
            warnings.append("CATMoS metrics are below score-only threshold.")

    elif legacy_name == "rf_clint_model":
        runtime_role = "disabled_by_default"
        use_in_final_cns_decision = False
        roc_auc = _safe_float(metrics.get("roc_auc"), 0.0) or 0.0
        pr_auc = _safe_float(metrics.get("pr_auc"), 0.0) or 0.0
        mcc = _safe_float(metrics.get("mcc"), 0.0) or 0.0
        if roc_auc >= policy.clint_min_roc_auc and pr_auc >= policy.clint_min_pr_auc and mcc >= policy.clint_min_mcc:
            selection_status = "selected_experimental"
            runtime_role = "auxiliary_admet_experimental"
            load_for_runtime = True
        else:
            selection_status = "disabled_weak_validation"
            load_for_runtime = False
            warnings.append("Clint v2 metrics are weak; disabled by default.")

    severity = SEVERITY_BY_SELECTION_STATUS.get(selection_status, "warning")

    return {
        "legacy_model_name": legacy_name,
        "model_name": model_name,
        "model_path": model_path,
        "model_card_path": entry.get("model_card_path"),
        "feature_schema_path": entry.get("feature_schema_path"),
        "validation_path": entry.get("validation_path"),
        "training_qa_status": qa_status,
        "selection_status": selection_status,
        "severity": severity,
        "runtime_role": runtime_role,
        "load_for_runtime": bool(load_for_runtime),
        "use_in_final_cns_decision": bool(use_in_final_cns_decision),
        "show_in_ui": bool(show_in_ui),
        "task_type": task_type,
        "feature_kind": feature_kind,
        "threshold": threshold,
        "metrics": metrics,
        "labels": MODEL_LABELS.get(legacy_name, {"ru": legacy_name, "kk": legacy_name, "en": legacy_name}),
        "role_labels": ROLE_LABELS.get(runtime_role, {"ru": runtime_role, "kk": runtime_role, "en": runtime_role}),
        "status_labels": STATUS_LABELS.get(selection_status, {"ru": selection_status, "kk": selection_status, "en": selection_status}),
        "reason": {
            "ru": _default_reason_ru(legacy_name, metrics),
            "kk": _default_reason_kk(legacy_name, metrics),
            "en": _default_reason_en(legacy_name, metrics),
        },
        "warnings": sorted(set(warnings)),
        "recommendations": sorted(set(recommendations)),
    }


def build_model_selection(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    policy: MetricThresholds = DEFAULT_POLICY,
    include_corrected_gupta: bool = True,
) -> Dict[str, Any]:
    registry_path = Path(registry_path)
    registry = load_json(registry_path)

    models = registry.get("models", {})
    selected: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    if include_corrected_gupta:
        selected["gupta_bbb_corrected"] = corrected_gupta_entry()

    for legacy_name, entry in models.items():
        card, card_error = _load_card(entry, registry_path)
        decision = select_one_model(
            legacy_name=legacy_name,
            entry=entry,
            card=card,
            card_error=card_error,
            policy=policy,
        )
        selected[legacy_name] = decision
        if card_error:
            errors[legacy_name] = card_error

    counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    for item in selected.values():
        counts[item["selection_status"]] = counts.get(item["selection_status"], 0) + 1
        role_counts[item["runtime_role"]] = role_counts.get(item["runtime_role"], 0) + 1

    primary_runtime = {
        "bbb_primary": "gupta_bbb_corrected",
        "pgp_primary": "rf_pgp_model" if selected.get("rf_pgp_model", {}).get("load_for_runtime") else None,
        "pka_helper": "rf_pka_model" if selected.get("rf_pka_model", {}).get("load_for_runtime") else None,
        "bbb_ml_supplementary": "rf_bbb_model" if selected.get("rf_bbb_model", {}).get("load_for_runtime") else None,
        "caco2_auxiliary": "rf_caco2_model" if selected.get("rf_caco2_model", {}).get("load_for_runtime") else None,
        "catmos_score_only": "rf_catmos_model" if selected.get("rf_catmos_model", {}).get("load_for_runtime") else None,
        "clint_auxiliary": "rf_clint_model" if selected.get("rf_clint_model", {}).get("load_for_runtime") else None,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "biosynth_stage": "7.4",
        "source_registry_path": str(registry_path),
        "source_registry_created_at": registry.get("created_at"),
        "policy": asdict(policy),
        "n_models_in_registry": len(models),
        "n_runtime_entries": len(selected),
        "selection_status_counts": counts,
        "runtime_role_counts": role_counts,
        "primary_runtime": primary_runtime,
        "models": selected,
        "errors": errors,
        "notes": [
            "Corrected Gupta remains the primary deterministic BBB score.",
            "P-gp v2 is selected as primary efflux model when validation metrics pass thresholds.",
            "BBB RF v2 is supplementary, not primary, because corrected Gupta is the main BBB block.",
            "CATMoS v2 is score-only until consensus_LD50 units/scale are confirmed.",
            "Clint v2 is disabled by default when validation metrics are weak.",
        ],
    }


def render_model_selection_markdown(selection: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# BioSynth-EDU Stage 7.4 Model Selection Report")
    lines.append("")
    lines.append(f"Schema version: `{selection.get('schema_version')}`")
    lines.append(f"Created at: `{selection.get('created_at')}`")
    lines.append(f"Source registry: `{selection.get('source_registry_path')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Registry models: **{selection.get('n_models_in_registry')}**")
    lines.append(f"Runtime entries: **{selection.get('n_runtime_entries')}**")
    lines.append(f"Selection status counts: `{selection.get('selection_status_counts')}`")
    lines.append(f"Runtime role counts: `{selection.get('runtime_role_counts')}`")
    lines.append("")
    lines.append("## Primary runtime map")
    lines.append("")
    for key, value in selection.get("primary_runtime", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Per-model selection")
    lines.append("")

    for legacy_name, item in selection.get("models", {}).items():
        metrics = item.get("metrics", {})
        lines.append(f"### {legacy_name}")
        lines.append("")
        lines.append(f"- Model name: `{item.get('model_name')}`")
        lines.append(f"- Runtime role: `{item.get('runtime_role')}`")
        lines.append(f"- Selection status: **{item.get('selection_status')}**")
        lines.append(f"- Severity: `{item.get('severity')}`")
        lines.append(f"- Load for runtime: `{item.get('load_for_runtime')}`")
        lines.append(f"- Use in final CNS decision: `{item.get('use_in_final_cns_decision')}`")
        lines.append(f"- Feature kind: `{item.get('feature_kind')}`")
        if item.get("threshold") is not None:
            lines.append(f"- Threshold: `{item.get('threshold')}`")
        lines.append(f"- Reason: {item.get('reason', {}).get('en')}")
        if metrics:
            lines.append("- Metrics:")
            for metric_key in ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "mcc", "r2", "mae", "rmse", "threshold"]:
                if metric_key in metrics:
                    lines.append(f"  - {metric_key}: `{metrics[metric_key]}`")
        if item.get("warnings"):
            lines.append("- Warnings:")
            for warning in item["warnings"]:
                lines.append(f"  - {warning}")
        if item.get("recommendations"):
            lines.append("- Recommendations:")
            for rec in item["recommendations"]:
                lines.append(f"  - {rec}")
        lines.append("")

    if selection.get("notes"):
        lines.append("## Notes")
        lines.append("")
        for note in selection.get("notes", []):
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def write_selection_artifacts(
    selection: Mapping[str, Any],
    output_dir: str | Path,
    json_name: str = "model_selection.json",
    markdown_name: str = "model_selection_report.md",
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / json_name
    md_path = output_dir / markdown_name
    write_json(selection, json_path)
    md_path.write_text(render_model_selection_markdown(selection), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def build_and_write_model_selection(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    output_dir: Optional[str | Path] = None,
    policy: MetricThresholds = DEFAULT_POLICY,
) -> Dict[str, Any]:
    selection = build_model_selection(registry_path=registry_path, policy=policy)
    if output_dir is not None:
        artifacts = write_selection_artifacts(selection, output_dir)
        selection = dict(selection)
        selection["saved_artifacts"] = artifacts
    return selection


def get_runtime_entry(selection: Mapping[str, Any], legacy_name: str) -> Optional[Dict[str, Any]]:
    item = selection.get("models", {}).get(legacy_name)
    return dict(item) if isinstance(item, Mapping) else None


def get_loadable_models(selection: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for legacy_name, item in selection.get("models", {}).items():
        if item.get("load_for_runtime") and item.get("model_path"):
            result[legacy_name] = dict(item)
    return result


def build_ui_status_rows(selection: Mapping[str, Any], lang: str = "ru") -> List[Dict[str, Any]]:
    """Return small localized rows suitable for a future Streamlit QA-status block."""
    if lang not in {"ru", "kk", "en"}:
        lang = "ru"
    rows: List[Dict[str, Any]] = []
    for legacy_name, item in selection.get("models", {}).items():
        metrics = item.get("metrics", {})
        metric_summary = ""
        if "roc_auc" in metrics:
            metric_summary = f"ROC-AUC={metrics.get('roc_auc'):.3f}, MCC={metrics.get('mcc', 0):.3f}"
        elif "r2" in metrics:
            metric_summary = f"R2={metrics.get('r2'):.3f}, RMSE={metrics.get('rmse', 0):.3f}"
        rows.append(
            {
                "legacy_model_name": legacy_name,
                "model": item.get("labels", {}).get(lang, legacy_name),
                "role": item.get("role_labels", {}).get(lang, item.get("runtime_role")),
                "status": item.get("status_labels", {}).get(lang, item.get("selection_status")),
                "severity": item.get("severity"),
                "load_for_runtime": item.get("load_for_runtime"),
                "use_in_final_cns_decision": item.get("use_in_final_cns_decision"),
                "metric_summary": metric_summary,
                "reason": item.get("reason", {}).get(lang, ""),
            }
        )
    return rows


def validate_selection(selection: Mapping[str, Any]) -> Dict[str, Any]:
    """Lightweight consistency checks for selection artifacts."""
    issues: List[str] = []
    models = selection.get("models", {})
    primary = selection.get("primary_runtime", {})

    if "gupta_bbb_corrected" not in models:
        issues.append("Corrected Gupta entry is missing.")
    if primary.get("bbb_primary") != "gupta_bbb_corrected":
        issues.append("bbb_primary must point to gupta_bbb_corrected.")

    pgp = models.get("rf_pgp_model")
    if not pgp or not pgp.get("load_for_runtime"):
        issues.append("P-gp model is not loadable; final CNS matrix will lose efflux block.")

    catmos = models.get("rf_catmos_model")
    if catmos and catmos.get("selection_status") == "selected_score_only" and catmos.get("use_in_final_cns_decision"):
        issues.append("CATMoS score-only model must not be used in final CNS decision.")

    clint = models.get("rf_clint_model")
    if clint and clint.get("selection_status") == "disabled_weak_validation" and clint.get("load_for_runtime"):
        issues.append("Weak Clint model should not be loadable by default.")

    return {
        "ok": not issues,
        "issues": issues,
    }
