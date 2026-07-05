"""Rule-based educational explainability backend for BioSynth-EDU.

This module does not run RDKit or ML models.  It takes the ordinary output of
``process_smiles_pipeline`` and transforms it into a stable ``explanation_dict``
that Streamlit can render later.

The functions are intentionally defensive: they accept several common key names
(``MolWt``/``MW``, ``bbb_classifier_probability``/``BBB_probability``, etc.) so
that the module can be connected to an existing project without a large pipeline
refactor.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any, Mapping

from .teaching_templates import (
    BASE_DESCRIPTOR_EXPLANATIONS_RU,
    BBB_HIGH_THRESHOLD,
    BBB_LOW_THRESHOLD,
    BORDERLINE_DISTANCE,
    DEFAULT_APPLICABILITY_MESSAGE_RU,
    DESCRIPTOR_ALIASES,
    DESCRIPTOR_META,
    DESCRIPTOR_ORDER,
    EFFECT_LABELS_RU,
    FINAL_DECISION_TEXTS,
    IN_SILICO_DISCLAIMER_RU,
    PGP_HIGH_THRESHOLD,
    PGP_LOW_THRESHOLD,
    WHAT_IF_DISCLAIMER_RU,
    ZONE_COMMENTS_RU,
    ZONE_LABELS_RU,
)
from core.matrix_text import matrix_cells as teaching_matrix_cells, matrix_intro as teaching_matrix_intro

from core.i18n import (
    descriptor_base_text,
    descriptor_display_name,
    descriptor_short_label,
    descriptor_threshold_note,
    descriptor_zone_comment,
    effect_label as localized_effect_label,
    final_decision_text,
    localize_warning,
    normalize_language,
    t,
    zone_label as localized_zone_label,
)

DescriptorDict = dict[str, Any]
ExplanationDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "nan", "null", "n/a", "na"}:
        return True
    return False


def _safe_float(value: Any) -> float | None:
    """Convert common scalar values to float; return None if impossible."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", ".")
        # Remove a trailing percent sign and convert 74% to 0.74 only for values
        # that clearly look like probabilities.
        if cleaned.endswith("%"):
            try:
                return float(cleaned[:-1]) / 100.0
            except ValueError:
                return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "ok", "valid"}:
            return True
        if lowered in {"false", "0", "no", "n", "invalid", "error"}:
            return False
    return bool(value)


def _normalize_key_string(name: str) -> tuple[str, str, str]:
    raw = str(name).strip().lower()
    underscored = re.sub(r"[^a-zа-я0-9]+", "_", raw).strip("_")
    compact = re.sub(r"[^a-zа-я0-9]+", "", raw)
    return raw, underscored, compact


def canonical_descriptor_name(name: str) -> str:
    """Return the stable BioSynth-EDU descriptor name for a pipeline key."""
    if name in DESCRIPTOR_META:
        return name

    raw, underscored, compact = _normalize_key_string(name)

    for candidate in (raw, underscored, compact):
        if candidate in DESCRIPTOR_ALIASES:
            return DESCRIPTOR_ALIASES[candidate]

    # Preserve unknown keys in a stable, display-safe way rather than dropping
    # them.  This makes the contract forward-compatible with new ADMET models.
    return str(name)


def _descriptor_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    name, _ = item
    try:
        return DESCRIPTOR_ORDER.index(name), name
    except ValueError:
        return len(DESCRIPTOR_ORDER), name


def _get_from_many(mapping: Mapping[str, Any], candidate_keys: list[str]) -> Any:
    if not isinstance(mapping, Mapping):
        return None

    direct = {str(k): v for k, v in mapping.items()}
    for key in candidate_keys:
        if key in direct and not _is_missing(direct[key]):
            return direct[key]

    canonical_candidates = {canonical_descriptor_name(k): k for k in direct.keys()}
    for key in candidate_keys:
        canonical = canonical_descriptor_name(key)
        original_key = canonical_candidates.get(canonical)
        if original_key is not None and not _is_missing(direct[original_key]):
            return direct[original_key]

    lower_candidates = {str(k).strip().lower(): k for k in direct.keys()}
    for key in candidate_keys:
        lowered = key.strip().lower()
        original_key = lower_candidates.get(lowered)
        if original_key is not None and not _is_missing(direct[original_key]):
            return direct[original_key]

    return None


def _format_value(value: Any, unit: str = "") -> str:
    numeric = _safe_float(value)
    if numeric is not None:
        if abs(numeric) >= 100:
            text = f"{numeric:.1f}"
        elif abs(numeric) >= 10:
            text = f"{numeric:.2f}"
        else:
            text = f"{numeric:.3f}".rstrip("0").rstrip(".")
    else:
        text = str(value)

    return f"{text} {unit}".strip()


def _status_from_zone(zone: str) -> str:
    if zone == "green":
        return "ok"
    if zone == "yellow":
        return "warning"
    if zone == "red":
        return "warning"
    return "info"


def _warning_to_dict(warning: Any) -> dict[str, Any]:
    if isinstance(warning, Mapping):
        return {
            "code": str(warning.get("code", "warning")),
            "severity": str(warning.get("severity", "warning")),
            "message": str(warning.get("message", warning.get("text", warning))),
        }
    return {"code": "warning", "severity": "warning", "message": str(warning)}


def _normalize_warnings(*warning_sources: Any) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for source in warning_sources:
        if _is_missing(source):
            continue
        if isinstance(source, str):
            warnings.append(_warning_to_dict(source))
        elif isinstance(source, Mapping):
            warnings.append(_warning_to_dict(source))
        elif isinstance(source, (list, tuple, set)):
            for item in source:
                warnings.append(_warning_to_dict(item))

    # Deduplicate by message while preserving order.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in warnings:
        key = item.get("message", "")
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


# ---------------------------------------------------------------------------
# Descriptor classification and explanation
# ---------------------------------------------------------------------------


def classify_descriptor_zone(name: str, value: Any, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Classify a descriptor into green/yellow/red/gray educational zones.

    Parameters
    ----------
    name:
        Descriptor name. Common aliases are accepted.
    value:
        Descriptor value.
    context:
        Optional pipeline or descriptor context. For pKa, this may contain
        ``pKa_type``/``acid_base_type`` with values such as ``"acid"`` or
        ``"base"``.

    Returns
    -------
    dict
        ``{"zone": ..., "effect": ..., "reason": ..., ...}``.
    """
    canonical = canonical_descriptor_name(name)
    numeric = _safe_float(value)

    if _is_missing(value):
        return {
            "name": canonical,
            "zone": "gray",
            "zone_label": ZONE_LABELS_RU["gray"],
            "effect": "unknown",
            "effect_label": EFFECT_LABELS_RU["unknown"],
            "reason": "Значение недоступно.",
        }

    if canonical == "MW":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif 150 <= numeric <= 450:
            zone, effect = "green", "supports_bbb"
        elif numeric < 150 or 450 < numeric <= 500:
            zone, effect = "yellow", "borderline"
        else:
            zone, effect = "red", "opposes_bbb"

    elif canonical == "LogP":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif 1.5 <= numeric <= 3.5:
            zone, effect = "green", "supports_bbb"
        elif 0.5 <= numeric < 1.5 or 3.5 < numeric <= 5.0:
            zone, effect = "yellow", "borderline"
        else:
            zone, effect = "red", "opposes_bbb"

    elif canonical == "TPSA":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif numeric <= 70:
            zone, effect = "green", "supports_bbb"
        elif numeric <= 90:
            zone, effect = "yellow", "borderline"
        else:
            zone, effect = "red", "opposes_bbb"

    elif canonical == "HBD":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif numeric <= 1:
            zone, effect = "green", "supports_bbb"
        elif numeric == 2:
            zone, effect = "yellow", "borderline"
        else:
            zone, effect = "red", "opposes_bbb"

    elif canonical == "HBA":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif numeric <= 5:
            zone, effect = "green", "supports_bbb"
        elif numeric <= 7:
            zone, effect = "yellow", "borderline"
        else:
            zone, effect = "red", "opposes_bbb"

    elif canonical == "RotatableBonds":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif numeric <= 5:
            zone, effect = "green", "supports_bbb"
        elif numeric <= 8:
            zone, effect = "yellow", "borderline"
        else:
            zone, effect = "red", "opposes_bbb"

    elif canonical == "AromaticRings":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif 1 <= numeric <= 3:
            zone, effect = "green", "supports_bbb"
        elif numeric == 0 or numeric == 4:
            zone, effect = "yellow", "borderline"
        else:
            zone, effect = "red", "opposes_bbb"

    elif canonical == "pKa_pred":
        zone, effect, extra = _classify_pka(value=numeric, context=context)
        return {
            "name": canonical,
            "zone": zone,
            "zone_label": ZONE_LABELS_RU[zone],
            "effect": effect,
            "effect_label": EFFECT_LABELS_RU[effect],
            "reason": ZONE_COMMENTS_RU.get(canonical, {}).get(zone, "pKa требует контекстной интерпретации."),
            **extra,
        }

    elif canonical == "FormalCharge":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif numeric == 0:
            zone, effect = "green", "supports_bbb"
        elif abs(numeric) == 1:
            zone, effect = "yellow", "borderline"
        else:
            zone, effect = "red", "opposes_bbb"

    elif canonical in {"GasteigerMin", "GasteigerMax", "GasteigerAbsMax"}:
        if numeric is None:
            zone, effect = "gray", "unknown"
        else:
            abs_value = abs(numeric)
            if abs_value < 0.45:
                zone, effect = "green", "supports_bbb"
            elif abs_value <= 0.75:
                zone, effect = "yellow", "borderline"
            else:
                zone, effect = "red", "opposes_bbb"

    elif canonical == "BBB_probability":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif numeric >= BBB_HIGH_THRESHOLD:
            zone, effect = "green", "supports_bbb"
        elif numeric >= BBB_LOW_THRESHOLD:
            zone, effect = "yellow", "borderline"
        else:
            zone, effect = "red", "opposes_bbb"

    elif canonical == "Pgp_probability":
        if numeric is None:
            zone, effect = "gray", "unknown"
        elif numeric < PGP_LOW_THRESHOLD:
            zone, effect = "green", "supports_cns_exposure"
        elif numeric < PGP_HIGH_THRESHOLD:
            zone, effect = "yellow", "uncertain"
        else:
            zone, effect = "red", "opposes_cns_exposure"

    elif canonical == "Clint_risk":
        zone, effect = _classify_ordinal_risk(value, low_green=True)

    elif canonical == "CATMoS_LD50":
        # LD50 units may differ between models. Do not overclaim here.
        zone, effect = "gray", "context_dependent"

    else:
        zone, effect = "gray", "context_dependent"

    return {
        "name": canonical,
        "zone": zone,
        "zone_label": ZONE_LABELS_RU.get(zone, zone),
        "effect": effect,
        "effect_label": EFFECT_LABELS_RU.get(effect, effect),
        "reason": ZONE_COMMENTS_RU.get(canonical, {}).get(zone, "Влияние зависит от контекста."),
    }


def _classify_pka(value: float | None, context: Mapping[str, Any] | None = None) -> tuple[str, str, dict[str, Any]]:
    if value is None:
        return "gray", "unknown", {}

    context = context or {}
    raw_type = _get_from_many(context, ["pKa_type", "pka_type", "acid_base_type", "ionization_type"])
    pka_type = str(raw_type).strip().lower() if raw_type is not None else ""

    if pka_type in {"base", "basic", "amine", "basic_center"}:
        neutral_fraction = 1.0 / (1.0 + 10.0 ** (value - 7.4))
        return _classify_neutral_fraction(neutral_fraction)

    if pka_type in {"acid", "acidic", "acidic_center"}:
        neutral_fraction = 1.0 / (1.0 + 10.0 ** (7.4 - value))
        return _classify_neutral_fraction(neutral_fraction)

    # Without acid/base type, pKa around physiological pH is educationally
    # ambiguous. Avoid pretending that a single pKa cut-off is universal.
    if 6.5 <= value <= 8.5:
        return "yellow", "context_dependent", {"pka_type": None, "neutral_fraction": None}
    if value > 8.5:
        return "yellow", "context_dependent", {"pka_type": None, "neutral_fraction": None}
    return "gray", "context_dependent", {"pka_type": None, "neutral_fraction": None}


def _classify_neutral_fraction(neutral_fraction: float) -> tuple[str, str, dict[str, Any]]:
    neutral_fraction = _clamp(neutral_fraction)
    if neutral_fraction >= 0.5:
        zone, effect = "green", "supports_bbb"
    elif neutral_fraction >= 0.1:
        zone, effect = "yellow", "borderline"
    else:
        zone, effect = "red", "opposes_bbb"
    return zone, effect, {"neutral_fraction_at_pH_7_4": neutral_fraction}


def _classify_ordinal_risk(value: Any, low_green: bool = True) -> tuple[str, str]:
    text = str(value).strip().lower()
    numeric = _safe_float(value)

    if numeric is not None:
        # Assume probability-like risk if numeric is 0..1.
        if 0 <= numeric <= 1:
            if numeric < 0.35:
                return ("green", "supports_cns_exposure") if low_green else ("red", "opposes_cns_exposure")
            if numeric < 0.65:
                return "yellow", "uncertain"
            return ("red", "opposes_cns_exposure") if low_green else ("green", "supports_cns_exposure")
        return "gray", "context_dependent"

    if text in {"low", "низкий", "низкая", "minor"}:
        return ("green", "supports_cns_exposure") if low_green else ("red", "opposes_cns_exposure")
    if text in {"medium", "moderate", "средний", "средняя", "умеренный", "умеренная"}:
        return "yellow", "uncertain"
    if text in {"high", "высокий", "высокая", "major"}:
        return ("red", "opposes_cns_exposure") if low_green else ("green", "supports_cns_exposure")
    return "gray", "context_dependent"


def _explain_descriptor_ru(
    name: str,
    value: Any,
    zone: str | None = None,
    effect: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Generate a short Russian educational explanation for one descriptor."""
    lang = "ru"
    canonical = canonical_descriptor_name(name)
    meta = DESCRIPTOR_META.get(canonical, {})
    unit = meta.get("unit", "")

    if zone is None or effect is None:
        zone_info = classify_descriptor_zone(canonical, value, context=context)
        zone = zone_info["zone"]
        effect = zone_info["effect"]
    else:
        zone_info = classify_descriptor_zone(canonical, value, context=context)

    label = descriptor_short_label(canonical, meta.get("short_label", canonical), lang)
    base = descriptor_base_text(canonical, lang)
    if base.startswith("descriptor."):
        base = BASE_DESCRIPTOR_EXPLANATIONS_RU.get(
            canonical,
            "Этот показатель используется как дополнительный признак в образовательной интерпретации ADMET-профиля.",
        )
    zone_comment = descriptor_zone_comment(canonical, str(zone), lang)
    if zone_comment.startswith("descriptor."):
        zone_comment = ZONE_COMMENTS_RU.get(canonical, {}).get(zone, zone_info.get("reason", "Влияние зависит от контекста."))
    threshold_note = descriptor_threshold_note(canonical, lang)
    if threshold_note.startswith("descriptor."):
        threshold_note = meta.get("threshold_note")

    if _is_missing(value):
        prefix = f"{label}: {t('effect.unknown', lang)}."
    else:
        prefix = f"{label} = {_format_value(value, unit)}."

    parts = [prefix, base, zone_comment]
    if threshold_note:
        parts.append(threshold_note)

    return " ".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# Model-output interpretation
# ---------------------------------------------------------------------------


def classify_bbb_score(bbb_score: Any) -> str:
    numeric = _safe_float(bbb_score)
    if numeric is None:
        return "bbb_unknown"
    if numeric >= BBB_HIGH_THRESHOLD:
        return "bbb_high"
    if numeric < BBB_LOW_THRESHOLD:
        return "bbb_low"
    return "bbb_borderline"


def classify_pgp_score(pgp_score: Any) -> str:
    numeric = _safe_float(pgp_score)
    if numeric is None:
        return "pgp_unknown"
    if numeric >= PGP_HIGH_THRESHOLD:
        return "pgp_likely_substrate"
    if numeric < PGP_LOW_THRESHOLD:
        return "pgp_not_substrate_like"
    return "pgp_borderline"


def classify_final_cns_profile(bbb_score: Any, pgp_score: Any) -> str:
    bbb = _safe_float(bbb_score)
    pgp = _safe_float(pgp_score)

    if bbb is None or pgp is None:
        return "insufficient_data"

    bbb_high = bbb >= BBB_HIGH_THRESHOLD
    bbb_low = bbb < BBB_LOW_THRESHOLD
    pgp_high = pgp >= PGP_HIGH_THRESHOLD
    pgp_low = pgp < PGP_LOW_THRESHOLD

    if bbb_high and pgp_low:
        return "likely_cns_active"
    if bbb_high and pgp_high:
        return "peripheral_action_risk"
    if bbb_low and pgp_low:
        return "likely_not_bbb_penetrant"
    if bbb_low and pgp_high:
        return "full_barrier"
    return "uncertain_or_borderline"


def _generate_cns_decision_explanation_ru(
    bbb_score: Any,
    pgp_score: Any,
    uncertainty: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the final Russian BBB x P-gp interpretation."""
    lang = "ru"
    final_class = classify_final_cns_profile(bbb_score, pgp_score)
    template = final_decision_text(final_class, lang)
    bbb_numeric = _safe_float(bbb_score)
    pgp_numeric = _safe_float(pgp_score)

    notes: list[str] = []
    if bbb_numeric is not None and abs(bbb_numeric - BBB_HIGH_THRESHOLD) <= BORDERLINE_DISTANCE:
        notes.append("Оценка прохождения через ГЭБ близка к верхнему порогу; интерпретация чувствительна к выбранной границе.")
    if bbb_numeric is not None and abs(bbb_numeric - BBB_LOW_THRESHOLD) <= BORDERLINE_DISTANCE:
        notes.append("Оценка прохождения через ГЭБ близка к нижнему порогу; интерпретация чувствительна к выбранной границе.")
    if pgp_numeric is not None and abs(pgp_numeric - PGP_HIGH_THRESHOLD) <= BORDERLINE_DISTANCE:
        notes.append("Оценка P-gp близка к порогу активного выведения; вывод о P-gp следует считать осторожным.")
    if pgp_numeric is not None and abs(pgp_numeric - PGP_LOW_THRESHOLD) <= BORDERLINE_DISTANCE:
        notes.append("P-gp score близок к порогу non-substrate; вывод о P-gp следует считать осторожным.")

    if uncertainty and uncertainty.get("reasons"):
        notes.extend(str(item) for item in uncertainty["reasons"])

    label = template.get("final_label") or template.get("final_label_ru") or template.get("title") or final_class
    return {
        "final_class": final_class,
        "title": template.get("title", label),
        "final_label_ru": label,
        "final_label": label,
        "summary": template.get("summary", ""),
        "student_interpretation": template.get("student_interpretation") or template.get("student", ""),
        "bbb_score": bbb_numeric,
        "pgp_score": pgp_numeric,
        "bbb_class": classify_bbb_score(bbb_score),
        "pgp_class": classify_pgp_score(pgp_score),
        "caution_notes": notes,
    }


def _generate_bbb_pgp_matrix_ru(bbb_score: Any, pgp_score: Any) -> dict[str, Any]:
    """Return the BBB x P-gp educational matrix and current cell."""
    bbb = _safe_float(bbb_score)
    pgp = _safe_float(pgp_score)

    if bbb is None or pgp is None:
        current_cell = "insufficient_data"
    elif bbb >= BBB_HIGH_THRESHOLD and pgp < PGP_LOW_THRESHOLD:
        current_cell = "bbb_high_pgp_low"
    elif bbb >= BBB_HIGH_THRESHOLD and pgp >= PGP_HIGH_THRESHOLD:
        current_cell = "bbb_high_pgp_high"
    elif bbb < BBB_LOW_THRESHOLD and pgp < PGP_LOW_THRESHOLD:
        current_cell = "bbb_low_pgp_low"
    elif bbb < BBB_LOW_THRESHOLD and pgp >= PGP_HIGH_THRESHOLD:
        current_cell = "bbb_low_pgp_high"
    else:
        current_cell = "borderline"

    lang = "ru"
    cells = teaching_matrix_cells(lang)
    return {
        "intro_text": teaching_matrix_intro(lang),
        "current_cell": current_cell,
        "current_interpretation": cells[current_cell]["interpretation"],
        "bbb_score": bbb,
        "pgp_score": pgp,
        "bbb_class": classify_bbb_score(bbb),
        "pgp_class": classify_pgp_score(pgp),
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Factor summary
# ---------------------------------------------------------------------------


def generate_factor_table(
    descriptors: Mapping[str, Any],
    model_outputs: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Group descriptor explanations into positive, negative and borderline factors.

    ``descriptors`` may be either raw values or already enriched descriptor
    entries from ``build_explanation_dict``.
    """
    model_outputs = model_outputs or {}
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    borderline: list[dict[str, Any]] = []
    neutral: list[dict[str, Any]] = []

    for raw_name, raw_item in sorted(descriptors.items(), key=_descriptor_sort_key):
        canonical = canonical_descriptor_name(raw_name)

        # BBB probability is an output rather than a causal factor; keep it in
        # the descriptor table but exclude it from factors za/protiv.
        if canonical in {"BBB_probability", "CATMoS_LD50"}:
            continue

        if isinstance(raw_item, Mapping) and "value" in raw_item:
            value = raw_item.get("value")
            zone = str(raw_item.get("zone", "gray"))
            effect = str(raw_item.get("effect", "context_dependent"))
            explanation = str(raw_item.get("explanation", ""))
            display_name = str(raw_item.get("short_label", raw_item.get("display_name", canonical)))
            importance = str(raw_item.get("importance", DESCRIPTOR_META.get(canonical, {}).get("importance", "low")))
        else:
            value = raw_item
            zone_info = classify_descriptor_zone(canonical, value, context=descriptors)
            zone = zone_info["zone"]
            effect = zone_info["effect"]
            explanation = explain_descriptor(canonical, value, zone=zone, effect=effect, context=descriptors)
            display_name = DESCRIPTOR_META.get(canonical, {}).get("short_label", canonical)
            importance = DESCRIPTOR_META.get(canonical, {}).get("importance", "low")

        factor = {
            "name": canonical,
            "display_name": display_name,
            "value": value,
            "zone": zone,
            "effect": effect,
            "importance": importance,
            "reason": _short_reason_from_explanation(explanation),
        }

        if zone == "green":
            positive.append(factor)
        elif zone == "red":
            negative.append(factor)
        elif zone == "yellow":
            borderline.append(factor)
        else:
            neutral.append(factor)

    return {
        "positive": _sort_factors(positive),
        "negative": _sort_factors(negative),
        "borderline": _sort_factors(borderline),
        "neutral": _sort_factors(neutral),
    }


def _short_reason_from_explanation(explanation: str) -> str:
    # Use the first two sentences to avoid flooding the factor cards.
    sentences = re.split(r"(?<=[.!?])\s+", explanation.strip())
    return " ".join(sentences[:2]).strip()


def _sort_factors(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    importance_rank = {"high": 0, "medium": 1, "low": 2}
    order_rank = {name: idx for idx, name in enumerate(DESCRIPTOR_ORDER)}
    return sorted(
        factors,
        key=lambda item: (
            importance_rank.get(str(item.get("importance", "low")), 3),
            order_rank.get(str(item.get("name")), 999),
        ),
    )


# ---------------------------------------------------------------------------
# Contract builder
# ---------------------------------------------------------------------------


def _build_explanation_dict_ru(pipeline_result: Mapping[str, Any]) -> ExplanationDict:
    """Transform ordinary ``process_smiles_pipeline`` output into explanation_dict.

    This is the primary public entry point for explanation generation.
    """
    if not isinstance(pipeline_result, Mapping):
        raise TypeError("pipeline_result must be a mapping/dict")
    lang = "ru"

    molecule = _build_molecule_info(pipeline_result, lang=lang)
    raw_model_outputs = _extract_model_outputs(pipeline_result)
    raw_descriptors = _extract_descriptors(pipeline_result, raw_model_outputs)
    applicability = _assess_applicability_domain(pipeline_result, molecule, raw_descriptors, lang=lang)

    enriched_descriptors: dict[str, dict[str, Any]] = {}
    for name, value in sorted(raw_descriptors.items(), key=_descriptor_sort_key):
        canonical = canonical_descriptor_name(name)
        meta = DESCRIPTOR_META.get(canonical, {})
        zone_info = classify_descriptor_zone(canonical, value, context={**raw_descriptors, **raw_model_outputs})
        enriched_descriptors[canonical] = {
            "value": value,
            "unit": meta.get("unit", ""),
            "zone": zone_info["zone"],
            "zone_label": localized_zone_label(zone_info["zone"], lang),
            "effect": zone_info["effect"],
            "effect_label": localized_effect_label(zone_info["effect"], lang),
            "importance": meta.get("importance", "low"),
            "display_name": descriptor_display_name(canonical, meta.get("display_name", canonical), lang),
            "short_label": descriptor_short_label(canonical, meta.get("short_label", canonical), lang),
            "threshold_note": descriptor_threshold_note(canonical, lang) if not descriptor_threshold_note(canonical, lang).startswith("descriptor.") else meta.get("threshold_note", ""),
            "explanation": explain_descriptor(canonical, value, zone=zone_info["zone"], effect=zone_info["effect"], context={**raw_descriptors, **raw_model_outputs}, lang=lang),
        }
        # Preserve useful calculated side information such as neutral fraction.
        for extra_key in ("neutral_fraction_at_pH_7_4", "pka_type"):
            if extra_key in zone_info:
                enriched_descriptors[canonical][extra_key] = zone_info[extra_key]

    model_outputs = _normalize_model_outputs(raw_model_outputs, enriched_descriptors)
    factor_summary = generate_factor_table(enriched_descriptors, model_outputs)
    uncertainty = _estimate_uncertainty(molecule, applicability, model_outputs, factor_summary, lang=lang)
    decision_explanation = generate_cns_decision_explanation(
        bbb_score=model_outputs.get("bbb_classifier_probability"),
        pgp_score=model_outputs.get("pgp_probability"),
        uncertainty=uncertainty,
        lang=lang,
    )
    decision_explanation["main_positive_factors"] = [item["name"] for item in factor_summary["positive"][:4]]
    decision_explanation["main_negative_factors"] = [item["name"] for item in factor_summary["negative"][:4]]

    return {
        "schema_version": "explainability_v1.0",
        "mode": "single_molecule",
        "language": lang,
        "molecule": molecule,
        "applicability_domain": applicability,
        "descriptors": enriched_descriptors,
        "model_outputs": model_outputs,
        "model_statuses": pipeline_result.get("model_statuses", {}) if isinstance(pipeline_result.get("model_statuses"), Mapping) else raw_model_outputs.get("model_statuses", {}),
        "model_errors": pipeline_result.get("model_errors", {}) if isinstance(pipeline_result.get("model_errors"), Mapping) else raw_model_outputs.get("model_errors", {}),
        "qa_warnings": pipeline_result.get("qa_warnings", []) if isinstance(pipeline_result.get("qa_warnings"), list) else raw_model_outputs.get("qa_warnings", []),
        "factor_summary": factor_summary,
        "bbb_pgp_matrix": generate_bbb_pgp_matrix(
            model_outputs.get("bbb_classifier_probability"),
            model_outputs.get("pgp_probability"),
            lang=lang,
        ),
        "stepwise_trace": _generate_stepwise_model_trace(molecule, enriched_descriptors, model_outputs, decision_explanation, lang=lang),
        "decision_explanation": decision_explanation,
        "uncertainty": uncertainty,
        "disclaimers": {
            "in_silico": t("msg.in_silico", lang),
            "what_if": t("msg.what_if_disclaimer", lang),
        },
        "what_if_base": _build_what_if_base_info(enriched_descriptors, model_outputs, lang=lang),
    }


def _build_molecule_info(pipeline_result: Mapping[str, Any], lang: str = "ru") -> dict[str, Any]:
    molecule_source = pipeline_result.get("molecule", {})
    if not isinstance(molecule_source, Mapping):
        molecule_source = {}

    input_smiles = _get_from_many(
        {**pipeline_result, **molecule_source},
        ["input_smiles", "smiles", "SMILES", "original_smiles", "query_smiles"],
    )
    canonical_smiles = _get_from_many(
        {**pipeline_result, **molecule_source},
        ["canonical_smiles", "canonicalSMILES", "canonical", "CanonSmiles"],
    )

    valid_value = _get_from_many(
        {**pipeline_result, **molecule_source},
        ["valid", "is_valid", "rdkit_parse_ok", "parse_ok"],
    )
    if valid_value is None:
        # Default to True only if there is no obvious error marker. This keeps
        # the module usable with old pipelines that did not expose a valid flag.
        valid = not bool(pipeline_result.get("error"))
    else:
        valid = bool(_safe_bool(valid_value))

    warnings = _normalize_warnings(
        pipeline_result.get("warnings"),
        molecule_source.get("warnings"),
        pipeline_result.get("applicability_warnings"),
    )

    if not valid and not any(item.get("code") == "invalid_smiles" for item in warnings):
        warnings.insert(
            0,
            {
                "code": "invalid_smiles",
                "severity": "error",
                "message": "SMILES не удалось корректно распознать или обработать.",
            },
        )

    warnings = [localize_warning(item, lang=lang) for item in warnings]

    return {
        "input_smiles": input_smiles,
        "canonical_smiles": canonical_smiles,
        "valid": valid,
        "rdkit_parse_ok": valid,
        "structure_image_available": bool(valid and canonical_smiles),
        "warnings": warnings,
    }


def _extract_descriptors(pipeline_result: Mapping[str, Any], model_outputs: Mapping[str, Any]) -> DescriptorDict:
    raw: dict[str, Any] = {}

    descriptor_source = pipeline_result.get("descriptors", {})
    if isinstance(descriptor_source, Mapping):
        raw.update(descriptor_source)

    # Some pipelines keep descriptor-like values at top level. Add only known
    # aliases to avoid polluting the descriptor table with arbitrary metadata.
    for key, value in pipeline_result.items():
        canonical = canonical_descriptor_name(str(key))
        if canonical in DESCRIPTOR_META:
            raw.setdefault(canonical, value)

    # Include selected model outputs in the educational descriptor table if they
    # are not already present.
    for key in ["pKa_pred", "BBB_probability", "Pgp_probability", "Clint_risk", "CATMoS_LD50"]:
        canonical = canonical_descriptor_name(key)
        if canonical not in raw:
            model_value = _get_from_many(model_outputs, [key, canonical])
            if not _is_missing(model_value):
                raw[canonical] = model_value

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = canonical_descriptor_name(str(key))
        if canonical in DESCRIPTOR_META:
            normalized[canonical] = value

    return dict(sorted(normalized.items(), key=_descriptor_sort_key))


def _extract_model_outputs(pipeline_result: Mapping[str, Any]) -> dict[str, Any]:
    model_source = pipeline_result.get("model_outputs", {})
    if not isinstance(model_source, Mapping):
        model_source = {}

    merged: dict[str, Any] = {}
    for source in (pipeline_result, model_source):
        if not isinstance(source, Mapping):
            continue
        merged.update(source)

    descriptors = pipeline_result.get("descriptors", {})
    if isinstance(descriptors, Mapping):
        # Keep descriptors lower priority than explicit model_outputs.
        descriptor_copy = dict(descriptors)
        descriptor_copy.update(merged)
        merged = descriptor_copy

    return {
        "bbb_v1_score": _get_from_many(
            merged,
            ["bbb_v1_score", "gupta_v1_score", "bbb_gupta_v1", "GuptaV1", "gupta_score"],
        ),
        "bbb_v2_score": _get_from_many(
            merged,
            ["bbb_v2_score", "gupta_v2_score", "hybrid_v2_score", "bbb_hybrid_v2", "HybridV2"],
        ),
        "bbb_classifier_probability": _get_from_many(
            merged,
            ["bbb_classifier_probability", "bbb_probability", "BBB_probability", "bbb_prob", "BBB prob", "BBB score"],
        ),
        "pgp_probability": _get_from_many(
            merged,
            ["pgp_probability", "Pgp_probability", "p_gp_probability", "pgp_prob", "pgp_score", "P-gp probability"],
        ),
        "pka_pred": _get_from_many(
            merged,
            ["pka_pred", "pKa_pred", "predicted_pka", "pka", "pKa"],
        ),
        "clint_risk": _get_from_many(
            merged,
            ["clint_risk", "Clint_risk", "clint", "intrinsic_clearance_risk"],
        ),
        "catmos_ld50": _get_from_many(
            merged,
            ["catmos_ld50", "CATMoS_LD50", "ld50", "LD50", "catmos"],
        ),
    }


def _normalize_model_outputs(
    raw_model_outputs: Mapping[str, Any],
    enriched_descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    bbb_prob = _safe_float(raw_model_outputs.get("bbb_classifier_probability"))
    pgp_prob = _safe_float(raw_model_outputs.get("pgp_probability"))
    bbb_v1 = _safe_float(raw_model_outputs.get("bbb_v1_score"))
    bbb_v2 = _safe_float(raw_model_outputs.get("bbb_v2_score"))
    pka_pred = _safe_float(raw_model_outputs.get("pka_pred"))

    # Fallback: if pKa was only present in descriptors, expose it in model_outputs
    # too because pKa_pred is model-generated in the current architecture.
    if pka_pred is None and "pKa_pred" in enriched_descriptors:
        pka_pred = _safe_float(enriched_descriptors["pKa_pred"].get("value"))

    final_cns_class = classify_final_cns_profile(bbb_prob, pgp_prob)

    return {
        "bbb_v1_score": bbb_v1,
        "bbb_v2_score": bbb_v2,
        "bbb_classifier_probability": bbb_prob,
        "bbb_class": classify_bbb_score(bbb_prob),
        "pgp_probability": pgp_prob,
        "pgp_class": classify_pgp_score(pgp_prob),
        "pka_pred": pka_pred,
        "clint_risk": raw_model_outputs.get("clint_risk"),
        "catmos_ld50": raw_model_outputs.get("catmos_ld50"),
        "final_cns_class": final_cns_class,
        "gupta_components": raw_model_outputs.get("gupta_components"),
        "gupta_legacy_components": raw_model_outputs.get("gupta_legacy_components"),
        "gupta_formula_version": raw_model_outputs.get("gupta_formula_version"),
        "bbb_rf_probability": raw_model_outputs.get("bbb_rf_probability"),
        "bbb_rf_class": raw_model_outputs.get("bbb_rf_class"),
        "model_statuses": raw_model_outputs.get("model_statuses", {}),
        "model_errors": raw_model_outputs.get("model_errors", {}),
        "qa_warnings": raw_model_outputs.get("qa_warnings", []),
    }


def _assess_applicability_domain(
    pipeline_result: Mapping[str, Any],
    molecule: Mapping[str, Any],
    descriptors: Mapping[str, Any],
    lang: str = "ru",
) -> dict[str, Any]:
    explicit = pipeline_result.get("applicability_domain")
    if isinstance(explicit, Mapping):
        level = explicit.get("level", "unknown")
        raw_warnings = explicit.get("warnings", []) or []
        warnings = [localize_warning(item, lang=lang) for item in raw_warnings]
        reasons = [item.get("message", "") for item in warnings] or (list(explicit.get("reasons", [])) if isinstance(explicit.get("reasons", []), list) else [])
        if level == "outside":
            message = t("applicability.outside", lang)
        elif level == "caution":
            message = t("applicability.caution", lang)
        elif level == "inside":
            message = t("applicability.inside", lang)
        else:
            message = explicit.get("student_message") or explicit.get("message") or DEFAULT_APPLICABILITY_MESSAGE_RU
        return {"level": level, "reasons": reasons, "warnings": warnings, "flags": list(explicit.get("flags", [])) if isinstance(explicit.get("flags", []), list) else [], "student_message": message}

    reasons: list[str] = []
    level = "inside"

    if not molecule.get("valid", True):
        return {
            "level": "outside",
            "reasons": ["invalid_smiles"],
            "student_message": t("applicability.invalid", lang),
        }

    for warning in molecule.get("warnings", []):
        severity = str(warning.get("severity", "warning")).lower()
        message = str(warning.get("message", ""))
        reasons.append(message)
        if severity == "error":
            level = "outside"
        elif level != "outside":
            level = "caution"

    mw = _safe_float(descriptors.get("MW"))
    tpsa = _safe_float(descriptors.get("TPSA"))
    hbd = _safe_float(descriptors.get("HBD"))
    hba = _safe_float(descriptors.get("HBA"))
    charge = _safe_float(descriptors.get("FormalCharge"))

    if mw is not None and mw > 700:
        level = "outside"
        reasons.append("Очень большая молекулярная масса: молекула может быть плохо сопоставима с примерами, на которых модель обычно надёжна.")
    elif mw is not None and mw > 500 and level == "inside":
        level = "caution"
        reasons.append("Молекулярная масса выше типичного drug-like/CNS диапазона.")

    if tpsa is not None and tpsa > 140:
        if level != "outside":
            level = "caution"
        reasons.append("Очень высокая TPSA: возможны гликозиды, полифенолы или сильно полярные соединения.")

    if hbd is not None and hbd >= 6:
        if level != "outside":
            level = "caution"
        reasons.append("Высокое число доноров водородных связей: профиль нетипичен для пассивного прохождения через ГЭБ.")

    if hba is not None and hba >= 12:
        if level != "outside":
            level = "caution"
        reasons.append("Высокое число акцепторов водородных связей: профиль нетипичен для пассивного прохождения через ГЭБ.")

    if charge is not None and abs(charge) >= 2:
        if level != "outside":
            level = "caution"
        reasons.append("Выраженный формальный заряд: интерпретация пассивного прохождения через ГЭБ может быть менее надёжной.")

    if not reasons:
        message = DEFAULT_APPLICABILITY_MESSAGE_RU
    elif level == "outside":
        message = "Молекула заметно отличается от типичных примеров для этой модели; результат следует трактовать очень осторожно."
    else:
        message = "Есть предупреждения о надёжности прогноза для этой молекулы; прогноз полезен как учебная гипотеза, но не как твёрдый вывод."

    # Deduplicate reasons.
    unique_reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    return {"level": level, "reasons": unique_reasons, "student_message": message}


def _estimate_uncertainty(
    molecule: Mapping[str, Any],
    applicability: Mapping[str, Any],
    model_outputs: Mapping[str, Any],
    factor_summary: Mapping[str, list[dict[str, Any]]],
    lang: str = "ru",
) -> dict[str, Any]:
    reasons: list[str] = []
    level = "low"

    if not molecule.get("valid", True):
        return {
            "level": "high",
            "reasons": ["invalid_smiles"],
            "student_message": t("uncertainty.high", lang),
        }

    app_level = applicability.get("level")
    if app_level == "outside":
        level = "high"
        reasons.append("Молекула вне базового домена применимости.")
    elif app_level == "caution":
        level = "medium"
        reasons.append("Есть предупреждения о надёжности прогноза для этой молекулы.")

    bbb = _safe_float(model_outputs.get("bbb_classifier_probability"))
    pgp = _safe_float(model_outputs.get("pgp_probability"))
    if bbb is None or pgp is None:
        level = "medium" if level == "low" else level
        reasons.append("Недоступна оценка прохождения через ГЭБ или оценка P-gp.")
    else:
        if BBB_LOW_THRESHOLD <= bbb < BBB_HIGH_THRESHOLD:
            level = "medium" if level == "low" else level
            reasons.append("Оценка прохождения через ГЭБ находится в пограничной зоне.")
        if PGP_LOW_THRESHOLD <= pgp < PGP_HIGH_THRESHOLD:
            level = "medium" if level == "low" else level
            reasons.append("Оценка P-gp находится в зоне неопределённости.")
        if bbb >= BBB_HIGH_THRESHOLD and pgp >= PGP_HIGH_THRESHOLD:
            level = "medium" if level == "low" else level
            reasons.append("BBB and P-gp signals conflict.")

    if factor_summary.get("negative") and factor_summary.get("positive"):
        if len(factor_summary["negative"]) >= 2 and level == "low":
            level = "medium"
            reasons.append("Есть одновременно положительные и отрицательные факторы.")

    if not reasons:
        message = t("uncertainty.low", lang)
    elif level == "high":
        message = t("uncertainty.high", lang)
    else:
        message = t("uncertainty.medium", lang)

    return {"level": level, "reasons": list(dict.fromkeys(reasons)), "student_message": message}


def _generate_stepwise_model_trace(
    molecule: Mapping[str, Any],
    descriptors: Mapping[str, Mapping[str, Any]],
    model_outputs: Mapping[str, Any],
    decision_explanation: Mapping[str, Any],
    lang: str = "ru",
) -> list[dict[str, Any]]:
    valid = bool(molecule.get("valid", False))
    warnings = molecule.get("warnings", [])

    if valid and not warnings:
        step1_status = "ok"
        step1_message = {"ru": "SMILES успешно распознан; явных предупреждений на этапе проверки нет.", "kk": "SMILES сәтті танылды; тексеру кезеңінде айқын ескертулер жоқ.", "en": "SMILES was parsed successfully; no obvious validation warnings."}.get(lang, "SMILES успешно распознан; явных предупреждений на этапе проверки нет.")
    elif valid:
        step1_status = "warning"
        prefix = {"ru": "SMILES распознан, но есть предупреждения: ", "kk": "SMILES танылды, бірақ ескертулер бар: ", "en": "SMILES was parsed, but there are warnings: "}.get(lang, "SMILES распознан, но есть предупреждения: ")
        step1_message = prefix + "; ".join(str(w.get("message", w)) for w in warnings)
    else:
        step1_status = "error"
        step1_message = {"ru": "SMILES не удалось корректно обработать. Проверьте запись молекулы.", "kk": "SMILES дұрыс өңделмеді. Молекула жазбасын тексеріңіз.", "en": "SMILES could not be processed correctly. Check the molecule string."}.get(lang, "SMILES не удалось корректно обработать. Проверьте запись молекулы.")

    descriptor_names = [name for name in DESCRIPTOR_ORDER if name in descriptors]
    if descriptor_names:
        step2_message = {"ru": "Рассчитаны или получены признаки: ", "kk": "Есептелген немесе алынған белгілер: ", "en": "Calculated or retrieved features: "}.get(lang, "Рассчитаны или получены признаки: ") + ", ".join(descriptor_names) + "."
    else:
        step2_message = t("msg.descriptors_unavailable", lang)

    bbb_prob = model_outputs.get("bbb_classifier_probability")
    if bbb_prob is None:
        step3_status = "warning"
        step3_message = {"ru": "Оценка прохождения через ГЭБ недоступна; пассивное прохождение можно оценить только по дескрипторам.", "kk": "Қан-ми тосқауылынан өту бағасы қолжетімсіз; пассивті өтуді тек дескрипторлар бойынша бағалауға болады.", "en": "BBB score is unavailable; passive BBB permeability can only be assessed from descriptors."}.get(lang, "Оценка прохождения через ГЭБ недоступна; пассивное прохождение можно оценить только по дескрипторам.")
    else:
        bbb_zone = classify_descriptor_zone("BBB_probability", bbb_prob)["zone"]
        step3_status = _status_from_zone(bbb_zone)
        step3_message = {"ru": f"Оценка прохождения через гематоэнцефалический барьер (ГЭБ, BBB): {_format_value(bbb_prob)}.", "kk": f"Қан-ми тосқауылынан (BBB) өту бағасы: {_format_value(bbb_prob)}.", "en": f"Blood-brain barrier (BBB) passage estimate: {_format_value(bbb_prob)}."}.get(lang, f"Оценка прохождения через гематоэнцефалический барьер (ГЭБ, BBB): {_format_value(bbb_prob)}.")

    pgp_prob = model_outputs.get("pgp_probability")
    if pgp_prob is None:
        step4_status = "warning"
        step4_message = {"ru": "Оценка P-gp недоступна; риск активного выведения не оценён.", "kk": "P-gp бағасы қолжетімсіз; белсенді шығарылу қаупі бағаланбады.", "en": "P-gp score is unavailable; active efflux risk was not evaluated."}.get(lang, "Оценка P-gp недоступна; риск активного выведения не оценён.")
    else:
        pgp_zone = classify_descriptor_zone("Pgp_probability", pgp_prob)["zone"]
        step4_status = _status_from_zone(pgp_zone)
        step4_message = {"ru": f"Оценка риска активного выведения через P-gp: {_format_value(pgp_prob)}.", "kk": f"P-gp арқылы белсенді шығарылу қаупінің бағасы: {_format_value(pgp_prob)}.", "en": f"P-gp active-efflux risk estimate: {_format_value(pgp_prob)}."}.get(lang, f"Оценка риска активного выведения через P-gp: {_format_value(pgp_prob)}.")

    final_class = decision_explanation.get("final_class", "uncertain_or_borderline")
    step5_status = "ok" if final_class == "likely_cns_active" else "warning"
    if final_class in {"full_barrier", "insufficient_data"}:
        step5_status = "error" if final_class == "insufficient_data" else "warning"

    return [
        {
            "step": 1,
            "title": {"ru": "Проверка SMILES", "kk": "SMILES тексеру", "en": "SMILES validation"}.get(lang, "Проверка SMILES"),
            "status": step1_status,
            "message": step1_message,
        },
        {
            "step": 2,
            "title": {"ru": "Расчёт дескрипторов", "kk": "Дескрипторларды есептеу", "en": "Descriptor calculation"}.get(lang, "Расчёт дескрипторов"),
            "status": "ok" if descriptor_names else "warning",
            "message": step2_message,
        },
        {
            "step": 3,
            "title": {"ru": "Оценка пассивного прохождения через ГЭБ", "kk": "Қан-ми тосқауылынан пассивті өтуді бағалау", "en": "Passive BBB permeability assessment"}.get(lang, "Оценка пассивного прохождения через ГЭБ"),
            "status": step3_status,
            "message": step3_message,
        },
        {
            "step": 4,
            "title": {"ru": "Оценка P-gp", "kk": "P-gp бағалау", "en": "P-gp assessment"}.get(lang, "Оценка P-gp"),
            "status": step4_status,
            "message": step4_message,
        },
        {
            "step": 5,
            "title": {"ru": "Финальная учебная интерпретация", "kk": "Қорытынды оқу түсіндірмесі", "en": "Final educational interpretation"}.get(lang, "Финальная учебная интерпретация"),
            "status": step5_status,
            "message": str(decision_explanation.get("summary", {"ru": "Итоговая интерпретация недоступна.", "kk": "Қорытынды түсіндірме қолжетімсіз.", "en": "Final interpretation is unavailable."}.get(lang, "Итоговая интерпретация недоступна."))),
        },
    ]


def _build_what_if_base_descriptors(enriched_descriptors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keys = ["MW", "LogP", "TPSA", "HBD", "HBA", "pKa_pred", "Pgp_probability"]
    return {key: enriched_descriptors[key]["value"] for key in keys if key in enriched_descriptors}


def _build_what_if_base_info(
    enriched_descriptors: Mapping[str, Mapping[str, Any]],
    model_outputs: Mapping[str, Any],
    lang: str = "ru",
) -> dict[str, Any]:
    base_descriptors = _build_what_if_base_descriptors(enriched_descriptors)
    if "Pgp_probability" not in base_descriptors and model_outputs.get("pgp_probability") is not None:
        base_descriptors["Pgp_probability"] = model_outputs.get("pgp_probability")

    info: dict[str, Any] = {
        "available": True,
        "mode": "educational_heuristic_v1",
        "disclaimer": t("msg.what_if_disclaimer", lang),
        "base_descriptors": base_descriptors,
    }

    try:
        from core.what_if import calculate_simplified_bbb_score, calculate_simplified_cns_score

        passive_score = calculate_simplified_bbb_score(base_descriptors)
        cns_score = calculate_simplified_cns_score(
            passive_score,
            base_descriptors.get("Pgp_probability", model_outputs.get("pgp_probability")),
        )
        info["base_educational_scores"] = {
            "passive_bbb_score": passive_score,
            "cns_score": cns_score,
        }
    except Exception as exc:  # pragma: no cover - defensive fallback for partial installs
        info["base_educational_scores"] = {}
        info["score_warning"] = f"What-if score не рассчитан: {exc}"

    return info


# ---------------------------------------------------------------------------
# Multilingual explanation helpers
# ---------------------------------------------------------------------------


def explain_descriptor(
    name: str,
    value: Any,
    zone: str | None = None,
    effect: str | None = None,
    context: Mapping[str, Any] | None = None,
    lang: str = "ru",
) -> str:
    """Generate an educational descriptor explanation in ru/kk/en."""
    from core.i18n import normalize_language, descriptor_explanation

    lang = normalize_language(lang)
    if lang == "ru":
        return _explain_descriptor_ru(name, value, zone=zone, effect=effect, context=context)
    canonical = canonical_descriptor_name(name)
    meta = DESCRIPTOR_META.get(canonical, {})
    if zone is None or effect is None:
        zone_info = classify_descriptor_zone(canonical, value, context=context)
        zone = str(zone_info.get("zone", "gray"))
    unit = str(meta.get("unit", ""))
    return descriptor_explanation(canonical, _format_value(value, unit), str(zone), lang)


def generate_cns_decision_explanation(
    bbb_score: Any,
    pgp_score: Any,
    uncertainty: Mapping[str, Any] | None = None,
    lang: str = "ru",
) -> dict[str, Any]:
    """Return the final BBB x P-gp interpretation in ru/kk/en."""
    from core.i18n import normalize_language, final_decision_text

    lang = normalize_language(lang)
    result = _generate_cns_decision_explanation_ru(bbb_score, pgp_score, uncertainty=uncertainty)
    if lang != "ru":
        localized = final_decision_text(str(result.get("final_class", "uncertain_or_borderline")), lang)
        result.update(localized)
        result["final_label"] = localized.get("final_label_ru") or localized.get("title")
    return result


def generate_bbb_pgp_matrix(bbb_score: Any, pgp_score: Any, lang: str = "ru") -> dict[str, Any]:
    """Return the BBB x P-gp educational matrix in ru/kk/en."""
    from core.i18n import normalize_language
    from core.matrix_text import matrix_cells, matrix_intro

    lang = normalize_language(lang)
    result = _generate_bbb_pgp_matrix_ru(bbb_score, pgp_score)
    cells = matrix_cells(lang)
    current = str(result.get("current_cell", "insufficient_data"))
    result["intro_text"] = matrix_intro(lang)
    result["cells"] = cells
    result["current_interpretation"] = cells.get(current, cells.get("insufficient_data", {})).get("interpretation", "")
    return result


def build_explanation_dict(pipeline_result: Mapping[str, Any], lang: str = "ru") -> ExplanationDict:
    """Transform pipeline output into a localized explanation_dict."""
    from core.i18n import normalize_language, localize_explanation_dict

    lang = normalize_language(lang)
    data = _build_explanation_dict_ru(pipeline_result)
    # If the adapter already supplied a richer applicability-domain object, keep it.
    explicit_app = pipeline_result.get("applicability_domain") if isinstance(pipeline_result, Mapping) else None
    if isinstance(explicit_app, Mapping):
        data["applicability_domain"] = deepcopy(dict(explicit_app))
    return localize_explanation_dict(data, lang)
