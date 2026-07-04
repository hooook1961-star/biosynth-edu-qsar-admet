"""Batch explainability helpers for BioSynth-EDU mass screening.

Stage 5 goal
------------
Mass screening needs compact, spreadsheet-friendly explanations. This module
turns current flat ADMET rows into ``xai_*`` columns, summarizes a whole batch,
and prepares multi-sheet Excel exports.

The implementation reuses the accepted Stage 1-4 pipeline:

``current app row -> adapter -> explanation_dict -> compact xai row``

No ML model is trained here, no SMILES is modified, and no result is presented
as experimental validation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from rdkit import Chem

from core.explainability import build_explanation_dict
from core.explainability_adapter import build_pipeline_result_from_current_app
from core.i18n import normalize_language, batch_priority_label, batch_summary_text, warning_message
from core.matrix_text import matrix_current_label


BATCH_XAI_SCHEMA_VERSION = "batch_xai_v1.0"
BATCH_EXPLAINABILITY_VERSION = "batch_explainability_v1.0"

XAI_PRIORITY_ORDER: dict[str, int] = {
    "CNS candidate": 0,
    "CNS candidate / caution": 1,
    "Efflux risk": 2,
    "Review": 3,
    "Low passive BBB": 4,
    "Full barrier": 5,
    "Outside domain": 6,
    "Invalid/error": 7,
}

XAI_EXPORT_COLUMNS: tuple[str, ...] = (
    "xai_schema_version",
    "xai_valid_smiles",
    "xai_canonical_smiles",
    "xai_final_class",
    "xai_final_label",
    "xai_short_summary",
    "xai_bbb_normalized_score",
    "xai_bbb_gupta_score",
    "xai_bbb_class",
    "xai_pgp_probability",
    "xai_pgp_class",
    "xai_pka_pred",
    "xai_bbb_pgp_scenario",
    "xai_matrix_interpretation",
    "xai_positive_factors",
    "xai_negative_factors",
    "xai_borderline_factors",
    "xai_applicability_level",
    "xai_uncertainty_level",
    "xai_warnings",
    "xai_review_reasons",
    "xai_batch_priority",
)

FINAL_LABEL_FALLBACK_RU: dict[str, str] = {
    "likely_cns_active": "Вероятно ЦНС-активный профиль",
    "peripheral_action_risk": "Хорошая оценка ГЭБ, но есть риск активного выведения через P-gp",
    "likely_not_bbb_penetrant": "Вероятно не проходит BBB",
    "full_barrier": "Полный барьер BBB + P-gp",
    "uncertain_or_borderline": "Неопределённо / погранично",
    "invalid_or_error": "Некорректный SMILES / ошибка расчёта",
}


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_explanation_from_current_batch_row(
    row: Mapping[str, Any],
    *,
    smiles_key: str = "SMILES",
    lang: str = "ru",
) -> dict[str, Any]:
    """Build a full ``explanation_dict`` from one current-app batch row."""
    lang = normalize_language(lang)
    source = dict(row or {})
    smi = str(source.get(smiles_key) or source.get("SMILES") or source.get("smiles") or "").strip()
    mol = Chem.MolFromSmiles(smi) if smi else None

    if mol is None:
        results = {**source, "error": source.get("error") or warning_message("invalid_smiles", lang)}
    else:
        results = source

    pipeline_result = build_pipeline_result_from_current_app(
        input_smiles=smi,
        mol=mol,
        descriptors=source,
        results=results,
        lang=lang,
    )
    return build_explanation_dict(pipeline_result, lang=lang)


def build_batch_explanation_row(
    input_smiles: str,
    *,
    descriptors: Mapping[str, Any] | None = None,
    results: Mapping[str, Any] | None = None,
    mol: Chem.Mol | None = None,
    include_long_text: bool = False,
    lang: str = "ru",
) -> dict[str, Any]:
    """Build compact ``xai_*`` columns for one molecule from current app data."""
    lang = normalize_language(lang)
    descriptors = descriptors or {}
    results = results or {}
    clean_smiles = str(input_smiles or "").strip()

    if mol is None and clean_smiles:
        mol = Chem.MolFromSmiles(clean_smiles)

    if mol is None or results.get("error"):
        return _build_invalid_xai_row(clean_smiles, results=results, include_long_text=include_long_text, lang=lang)

    pipeline_result = build_pipeline_result_from_current_app(
        input_smiles=clean_smiles,
        mol=mol,
        descriptors=descriptors,
        results=results,
        lang=lang,
    )
    explanation_dict = build_explanation_dict(pipeline_result, lang=lang)
    return build_batch_explanation_row_from_explanation(
        explanation_dict,
        include_long_text=include_long_text,
        lang=lang,
    )


def build_batch_explanation_row_from_explanation(
    explanation_dict: Mapping[str, Any],
    *,
    include_long_text: bool = False,
    lang: str | None = None,
) -> dict[str, Any]:
    """Build compact ``xai_*`` columns from an existing ``explanation_dict``."""
    lang = normalize_language(lang or explanation_dict.get("language"))
    molecule = _as_mapping(explanation_dict.get("molecule"))
    model_outputs = _as_mapping(explanation_dict.get("model_outputs"))
    decision = _as_mapping(explanation_dict.get("decision_explanation"))
    matrix = _as_mapping(explanation_dict.get("bbb_pgp_matrix"))
    applicability = _as_mapping(explanation_dict.get("applicability_domain"))
    uncertainty = _as_mapping(explanation_dict.get("uncertainty"))

    if not bool(molecule.get("valid")):
        return _build_invalid_xai_row(
            str(molecule.get("input_smiles") or ""),
            results={},
            include_long_text=include_long_text,
            lang=lang,
        )

    final_class = str(
        decision.get("final_class")
        or model_outputs.get("final_cns_class")
        or "uncertain_or_borderline"
    )
    final_label = str(
        decision.get("final_label_ru")
        or decision.get("title")
        or FINAL_LABEL_FALLBACK_RU.get(final_class, final_class)
    )

    positive = _factor_names(explanation_dict, "positive")
    negative = _factor_names(explanation_dict, "negative")
    borderline = _factor_names(explanation_dict, "borderline")
    warnings = _join_warnings(molecule.get("warnings"))

    row: dict[str, Any] = {
        "xai_schema_version": BATCH_XAI_SCHEMA_VERSION,
        "xai_valid_smiles": True,
        "xai_canonical_smiles": molecule.get("canonical_smiles"),
        "xai_final_class": final_class,
        "xai_final_label": final_label,
        "xai_short_summary": _build_short_summary(final_label, matrix, negative, borderline),
        "xai_bbb_normalized_score": _round_or_none(model_outputs.get("bbb_classifier_probability")),
        "xai_bbb_gupta_score": _round_or_none(model_outputs.get("bbb_v2_score")),
        "xai_bbb_class": model_outputs.get("bbb_class"),
        "xai_pgp_probability": _round_or_none(model_outputs.get("pgp_probability")),
        "xai_pgp_class": model_outputs.get("pgp_class"),
        "xai_pka_pred": _round_or_none(model_outputs.get("pka_pred")),
        "xai_bbb_pgp_scenario": matrix_current_label(str(matrix.get("current_cell") or "insufficient_data"), lang),
        "xai_matrix_interpretation": matrix.get("current_interpretation"),
        "xai_positive_factors": _join_names(positive),
        "xai_negative_factors": _join_names(negative),
        "xai_borderline_factors": _join_names(borderline),
        "xai_applicability_level": applicability.get("level", "unknown"),
        "xai_uncertainty_level": uncertainty.get("level", "unknown"),
        "xai_warnings": warnings,
        "xai_review_reasons": _build_review_reasons(explanation_dict, final_class, warnings),
        "xai_batch_priority": _classify_xai_priority(explanation_dict, final_class),
    }
    row["xai_batch_priority_label"] = batch_priority_label(str(row["xai_batch_priority"]), lang)

    if include_long_text:
        row.update(
            {
                "xai_decision_summary_long": decision.get("summary", ""),
                "xai_student_interpretation_long": decision.get("student_interpretation", ""),
                "xai_applicability_message_long": applicability.get("student_message", ""),
                "xai_uncertainty_message_long": uncertainty.get("student_message", ""),
            }
        )

    return row


def build_batch_explainability_row(
    explanation_dict: Mapping[str, Any],
    *,
    row_index: int | None = None,
    include_long_text: bool = False,
    lang: str | None = None,
) -> dict[str, Any]:
    """Build newer ``explain_*`` columns from an ``explanation_dict``.

    The app example uses the compact ``xai_*`` columns for Excel.  This function
    is provided for code that prefers the more explicit ``explain_*`` naming.
    """
    selected_lang = normalize_language(lang or explanation_dict.get("language"))
    xai = build_batch_explanation_row_from_explanation(explanation_dict, include_long_text=include_long_text, lang=selected_lang)
    row = {
        "explain_schema_version": BATCH_EXPLAINABILITY_VERSION,
        "explain_valid": xai.get("xai_valid_smiles"),
        "explain_canonical_smiles": xai.get("xai_canonical_smiles"),
        "explain_final_class": xai.get("xai_final_class"),
        "explain_final_label_ru": xai.get("xai_final_label"),
        "explain_short_summary_ru": xai.get("xai_short_summary"),
        "explain_bbb_score_normalized": xai.get("xai_bbb_normalized_score"),
        "explain_bbb_class": xai.get("xai_bbb_class"),
        "explain_pgp_probability": xai.get("xai_pgp_probability"),
        "explain_pgp_class": xai.get("xai_pgp_class"),
        "explain_pka_pred": xai.get("xai_pka_pred"),
        "explain_gupta_score": xai.get("xai_bbb_gupta_score"),
        "explain_matrix_cell": xai.get("xai_bbb_pgp_scenario"),
        "explain_matrix_interpretation": xai.get("xai_matrix_interpretation"),
        "explain_positive_factors": xai.get("xai_positive_factors"),
        "explain_negative_factors": xai.get("xai_negative_factors"),
        "explain_borderline_factors": xai.get("xai_borderline_factors"),
        "explain_priority_flag": _priority_flag_from_xai_label(str(xai.get("xai_batch_priority") or "")),
        "explain_priority_label_ru": str(xai.get("xai_batch_priority") or ""),
        "explain_uncertainty_level": xai.get("xai_uncertainty_level"),
        "explain_applicability_level": xai.get("xai_applicability_level"),
        "explain_warnings": xai.get("xai_warnings"),
        "explain_teacher_note_ru": _teacher_note_from_xai_priority(str(xai.get("xai_batch_priority") or "")),
    }
    if row_index is not None:
        row["explain_row_index"] = int(row_index)
    return row


def build_batch_export_dataframe(
    rows: Sequence[Mapping[str, Any]],
    *,
    smiles_key: str = "SMILES",
    include_long_text: bool = False,
    lang: str = "ru",
) -> pd.DataFrame:
    """Append compact ``xai_*`` columns to current-app batch rows."""
    lang = normalize_language(lang)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        base = dict(row)
        smi = str(base.get(smiles_key) or base.get("SMILES") or base.get("smiles") or "").strip()
        mol = Chem.MolFromSmiles(smi) if smi else None
        xai = build_batch_explanation_row(
            smi,
            descriptors=base,
            results=base,
            mol=mol,
            include_long_text=include_long_text,
            lang=lang,
        )
        output_rows.append({**base, **xai})
    return pd.DataFrame(output_rows)


# ---------------------------------------------------------------------------
# Batch summaries and Excel sheets
# ---------------------------------------------------------------------------


def _summarize_batch_explanations_ru(batch_df: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build summary statistics for a batch table with ``xai_*`` columns."""
    df = _ensure_dataframe(batch_df)
    n_total = int(len(df))
    if n_total == 0:
        return _empty_summary()

    n_valid = int(df.get("xai_valid_smiles", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    n_invalid = n_total - n_valid

    final_class_counts = _value_counts(df, "xai_final_class")
    scenario_counts = _value_counts(df, "xai_bbb_pgp_scenario")
    priority_counts = _value_counts(df, "xai_batch_priority")
    uncertainty_counts = _value_counts(df, "xai_uncertainty_level")
    applicability_counts = _value_counts(df, "xai_applicability_level")

    top_negative_factors = _top_semicolon_values(df, "xai_negative_factors", item_key="factor")
    top_warnings = _top_semicolon_values(df, "xai_warnings", item_key="warning")

    summary_text = _build_summary_text(
        n_total=n_total,
        n_valid=n_valid,
        n_invalid=n_invalid,
        final_class_counts=final_class_counts,
    )

    return {
        "schema_version": BATCH_XAI_SCHEMA_VERSION,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "total": n_total,
        "valid_count": n_valid,
        "invalid_count": n_invalid,
        "final_class_counts": final_class_counts,
        "scenario_counts": scenario_counts,
        "priority_counts": priority_counts,
        "uncertainty_counts": uncertainty_counts,
        "applicability_counts": applicability_counts,
        "top_negative_factors": top_negative_factors,
        "top_warnings": top_warnings,
        "summary_text": summary_text,
        "teaching_summary_ru": summary_text,
        "recommended_next_steps": _build_recommended_next_steps(final_class_counts, priority_counts, n_invalid),
    }


def build_batch_summary_dataframe(summary: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten summary statistics into a two-column DataFrame for Excel."""
    rows: list[dict[str, Any]] = []
    for key in ["n_total", "n_valid", "n_invalid", "summary_text"]:
        rows.append({"metric": key, "value": summary.get(key)})

    for group_key in ["final_class_counts", "scenario_counts", "priority_counts", "uncertainty_counts", "applicability_counts"]:
        for name, value in _as_mapping(summary.get(group_key)).items():
            rows.append({"metric": f"{group_key}.{name}", "value": value})

    return pd.DataFrame(rows, columns=["metric", "value"])


def build_frequency_dataframe(items: Sequence[Mapping[str, Any]] | None, item_key: str) -> pd.DataFrame:
    """Turn ``[{item_key: ..., count: ...}]`` into a stable DataFrame."""
    if not items:
        return pd.DataFrame(columns=[item_key, "count"])
    rows = []
    for item in items:
        mapping = _as_mapping(item)
        rows.append({item_key: mapping.get(item_key), "count": mapping.get("count", 0)})
    return pd.DataFrame(rows, columns=[item_key, "count"])


def build_batch_excel_sheets(batch_df: pd.DataFrame, summary: Mapping[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Prepare named DataFrames for multi-sheet Excel export."""
    summary = summary or summarize_batch_explanations(batch_df)
    return {
        "Full_ADMET_XAI": batch_df,
        "XAI_Summary": build_batch_summary_dataframe(summary),
        "Final_Class_Counts": _counter_to_dataframe(summary.get("final_class_counts", {}), "final_class"),
        "Priority_Counts": _counter_to_dataframe(summary.get("priority_counts", {}), "priority"),
        "Top_Negative_Factors": build_frequency_dataframe(summary.get("top_negative_factors", []), "factor"),
        "Top_Warnings": build_frequency_dataframe(summary.get("top_warnings", []), "warning"),
    }


def sort_batch_by_explainability_priority(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Sort batch rows so likely CNS candidates appear first."""
    df = batch_df.copy()
    if "xai_batch_priority" not in df.columns:
        return df
    df["_xai_priority_order"] = df["xai_batch_priority"].map(XAI_PRIORITY_ORDER).fillna(99).astype(int)
    if "xai_bbb_normalized_score" in df.columns:
        df = df.sort_values(["_xai_priority_order", "xai_bbb_normalized_score"], ascending=[True, False])
    else:
        df = df.sort_values(["_xai_priority_order"], ascending=[True])
    return df.drop(columns=["_xai_priority_order"])


# Compatibility alias is assigned after the Stage 6 localized summary override.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_invalid_xai_row(input_smiles: str, *, results: Mapping[str, Any], include_long_text: bool, lang: str = "ru") -> dict[str, Any]:
    lang = normalize_language(lang)
    error = str(results.get("error") or warning_message("invalid_smiles", lang))
    row: dict[str, Any] = {
        "xai_schema_version": BATCH_XAI_SCHEMA_VERSION,
        "xai_valid_smiles": False,
        "xai_canonical_smiles": None,
        "xai_final_class": "invalid_or_error",
        "xai_final_label": {"ru": FINAL_LABEL_FALLBACK_RU["invalid_or_error"], "kk": "Invalid SMILES / есептеу қатесі", "en": "Invalid SMILES / calculation error"}.get(lang, FINAL_LABEL_FALLBACK_RU["invalid_or_error"]),
        "xai_short_summary": {"ru": f"SMILES `{input_smiles}` не удалось корректно обработать: {error}.", "kk": f"SMILES `{input_smiles}` дұрыс өңделмеді: {error}.", "en": f"SMILES `{input_smiles}` could not be processed correctly: {error}."}[lang],
        "xai_bbb_normalized_score": None,
        "xai_bbb_gupta_score": None,
        "xai_bbb_class": None,
        "xai_pgp_probability": None,
        "xai_pgp_class": None,
        "xai_pka_pred": None,
        "xai_bbb_pgp_scenario": matrix_current_label("invalid_or_error", lang),
        "xai_matrix_interpretation": {"ru": "Матрица BBB × P-gp недоступна для некорректной структуры.", "kk": "Қате құрылым үшін BBB × P-gp матрицасы қолжетімсіз.", "en": "The BBB × P-gp matrix is unavailable for an invalid structure."}[lang],
        "xai_positive_factors": "",
        "xai_negative_factors": "",
        "xai_borderline_factors": "",
        "xai_applicability_level": "outside",
        "xai_uncertainty_level": "high",
        "xai_warnings": error,
        "xai_review_reasons": {"ru": "Сначала исправить SMILES; модельные выводы недоступны.", "kk": "Алдымен SMILES түзету керек; модель қорытындысы қолжетімсіз.", "en": "Fix the SMILES first; model conclusions are unavailable."}[lang],
        "xai_batch_priority": "Invalid/error",
        "xai_batch_priority_label": batch_priority_label("Invalid/error", lang),
    }
    if include_long_text:
        row.update(
            {
                "xai_decision_summary_long": row["xai_short_summary"],
                "xai_student_interpretation_long": "Невалидная структура не должна попадать в интерпретацию ADMET как реальный прогноз.",
                "xai_applicability_message_long": "Вне домена: структура не распознана.",
                "xai_uncertainty_message_long": "Неопределённость высокая, потому что молекулярный граф не построен.",
            }
        )
    return row


def _classify_xai_priority(explanation_dict: Mapping[str, Any], final_class: str) -> str:
    applicability = _as_mapping(explanation_dict.get("applicability_domain"))
    uncertainty = _as_mapping(explanation_dict.get("uncertainty"))
    warnings = _join_warnings(_as_mapping(explanation_dict.get("molecule")).get("warnings"))

    applicability_level = str(applicability.get("level") or "unknown")
    uncertainty_level = str(uncertainty.get("level") or "unknown")

    if applicability_level == "outside":
        return "Outside domain"
    if final_class == "likely_cns_active":
        if applicability_level in {"caution", "unknown"} or uncertainty_level in {"medium", "high"} or warnings:
            return "CNS candidate / caution"
        return "CNS candidate"
    if final_class == "peripheral_action_risk":
        return "Efflux risk"
    if final_class == "likely_not_bbb_penetrant":
        return "Low passive BBB"
    if final_class == "full_barrier":
        return "Full barrier"
    return "Review"


def _priority_flag_from_xai_label(label: str) -> str:
    mapping = {
        "CNS candidate": "cns_candidate",
        "CNS candidate / caution": "cns_candidate_with_caution",
        "Efflux risk": "pgp_efflux_risk",
        "Low passive BBB": "poor_passive_bbb",
        "Full barrier": "full_barrier",
        "Outside domain": "outside_domain",
        "Invalid/error": "invalid_smiles",
        "Review": "borderline_review",
    }
    return mapping.get(label, "unknown")


def _teacher_note_from_xai_priority(label: str) -> str:
    flag = _priority_flag_from_xai_label(label)
    notes = {
        "cns_candidate": "Хороший пример согласованного BBB/P-gp профиля для обсуждения CNS-доступности.",
        "cns_candidate_with_caution": "Кандидат интересен, но стоит разобрать предупреждения и пограничные признаки.",
        "pgp_efflux_risk": "Полезный учебный пример конфликта: пассивная проницаемость против активного эффлюкса.",
        "full_barrier": "Пример двойного ограничения: физико-химический BBB-барьер плюс P-gp.",
        "poor_passive_bbb": "Пример неблагоприятного физико-химического профиля для BBB.",
        "outside_domain": "Использовать для обсуждения домена применимости модели и ограничений in silico-прогноза.",
        "invalid_smiles": "Сначала исправить структуру SMILES; модельный вывод недоступен.",
    }
    return notes.get(flag, "Пограничный случай для ручного разбора.")


def _build_review_reasons(explanation_dict: Mapping[str, Any], final_class: str, warnings: str) -> str:
    reasons: list[str] = []
    if final_class == "peripheral_action_risk":
        reasons.append("конфликт ???????? ???/P-gp: пассивная проницаемость против активного эффлюкса")
    if final_class == "full_barrier":
        reasons.append("двойной барьер: плохая пассивная BBB-проницаемость и P-gp efflux")
    if final_class == "likely_not_bbb_penetrant":
        reasons.append("основное ограничение — физико-химический профиль пассивной BBB-проницаемости")
    if final_class == "uncertain_or_borderline":
        reasons.append("пограничные вероятности или противоречивые сигналы модели")

    applicability = _as_mapping(explanation_dict.get("applicability_domain"))
    if str(applicability.get("level") or "") in {"caution", "outside"}:
        reasons.append("есть предупреждение о домене применимости")
    if warnings:
        reasons.append("есть структурные/model warnings")

    negative = _factor_names(explanation_dict, "negative")
    borderline = _factor_names(explanation_dict, "borderline")
    if negative:
        reasons.append("негативные факторы: " + ", ".join(negative[:3]))
    elif borderline:
        reasons.append("пограничные факторы: " + ", ".join(borderline[:3]))

    return "; ".join(reasons)


def _build_short_summary(
    final_label: str,
    matrix: Mapping[str, Any],
    negative: Sequence[str],
    borderline: Sequence[str],
) -> str:
    parts = [final_label]
    interpretation = str(matrix.get("current_interpretation") or "").strip()
    if interpretation:
        parts.append(interpretation)
    if negative:
        parts.append("Мешают: " + ", ".join(negative[:3]))
    elif borderline:
        parts.append("Погранично: " + ", ".join(borderline[:3]))
    return ". ".join(part.rstrip(".") for part in parts if part) + "."


def _build_summary_text(
    *,
    n_total: int,
    n_valid: int,
    n_invalid: int,
    final_class_counts: Mapping[str, int],
) -> str:
    cns = int(final_class_counts.get("likely_cns_active", 0))
    efflux = int(final_class_counts.get("peripheral_action_risk", 0))
    low_bbb = int(final_class_counts.get("likely_not_bbb_penetrant", 0))
    full_barrier = int(final_class_counts.get("full_barrier", 0))
    review = int(final_class_counts.get("uncertain_or_borderline", 0))

    return (
        f"Обработано молекул: {n_total}; валидных: {n_valid}; ошибок/invalid: {n_invalid}. "
        f"CNS-кандидаты: {cns}; риск P-gp эффлюкса при хорошем BBB: {efflux}; "
        f"низкая пассивная BBB-проницаемость: {low_bbb}; полный барьер: {full_barrier}; "
        f"пограничных случаев: {review}."
    )


def _build_recommended_next_steps(
    final_class_counts: Mapping[str, int],
    priority_counts: Mapping[str, int],
    n_invalid: int,
) -> list[str]:
    steps: list[str] = []
    if priority_counts.get("CNS candidate") or priority_counts.get("CNS candidate / caution"):
        steps.append("Начать ручной разбор с CNS-кандидатов, затем проверить предупреждения о домене применимости.")
    if final_class_counts.get("peripheral_action_risk"):
        steps.append("Отдельно разобрать молекулы ???????? ???/P-gp как примеры конфликта пассивной проницаемости и эффлюкса.")
    if final_class_counts.get("uncertain_or_borderline"):
        steps.append("Для пограничных молекул посмотреть значения TPSA, LogP, pKa и P-gp probability около порогов.")
    if n_invalid:
        steps.append("Исправить invalid SMILES до повторного запуска batch-анализа.")
    if not steps:
        steps.append("Использовать таблицу xai_* как учебный чек-лист факторов BBB/P-gp для каждой молекулы.")
    return steps


def _empty_summary() -> dict[str, Any]:
    return {
        "schema_version": BATCH_XAI_SCHEMA_VERSION,
        "n_total": 0,
        "n_valid": 0,
        "n_invalid": 0,
        "total": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "final_class_counts": {},
        "scenario_counts": {},
        "priority_counts": {},
        "uncertainty_counts": {},
        "applicability_counts": {},
        "top_negative_factors": [],
        "top_warnings": [],
        "summary_text": "Batch-таблица пуста.",
        "teaching_summary_ru": "Batch-таблица пуста.",
        "recommended_next_steps": [],
    }


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    series = df[column].fillna("unknown").astype(str)
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).to_dict().items()}


def _top_semicolon_values(df: pd.DataFrame, column: str, *, item_key: str, limit: int = 10) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    counter: Counter[str] = Counter()
    for value in df[column].fillna("").astype(str):
        for item in _split_semicolon(value):
            counter[item] += 1
    return [{item_key: key, "count": int(value)} for key, value in counter.most_common(limit)]


def _counter_to_dataframe(counter: Mapping[str, Any], column_name: str) -> pd.DataFrame:
    rows = [{column_name: key, "count": value} for key, value in _as_mapping(counter).items()]
    if not rows:
        return pd.DataFrame(columns=[column_name, "count"])
    return pd.DataFrame(rows, columns=[column_name, "count"]).sort_values("count", ascending=False)


def _ensure_dataframe(value: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame(list(value or []))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _factor_names(explanation_dict: Mapping[str, Any], group: str) -> list[str]:
    factor_summary = _as_mapping(explanation_dict.get("factor_summary"))
    raw_items = factor_summary.get(group) or []
    names: list[str] = []
    for item in raw_items:
        if isinstance(item, Mapping):
            name = item.get("name") or item.get("display_name") or item.get("short_label")
        else:
            name = item
        if name:
            names.append(str(name))
    return names


def _join_names(names: Sequence[str]) -> str:
    return "; ".join(str(name) for name in names if str(name).strip())


def _join_warnings(warnings: Any) -> str:
    if not warnings:
        return ""
    if isinstance(warnings, str):
        return warnings
    if isinstance(warnings, Mapping):
        return str(warnings.get("message") or warnings)
    if isinstance(warnings, Iterable):
        messages: list[str] = []
        for item in warnings:
            if isinstance(item, Mapping):
                messages.append(str(item.get("message") or item.get("text") or item))
            else:
                messages.append(str(item))
        return "; ".join(message for message in messages if message.strip())
    return str(warnings)


def _split_semicolon(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def _round_or_none(value: Any, digits: int = 3) -> float | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", ".")
        if cleaned.lower() in {"", "none", "nan", "n/a", "na", "null", "ошибка"}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None

# Stage 5 small compatibility helpers.
def build_batch_explainability_rows(
    explanation_dicts: Iterable[Mapping[str, Any]],
    *,
    include_long_text: bool = False,
) -> list[dict[str, Any]]:
    return [
        build_batch_explainability_row(item, row_index=index, include_long_text=include_long_text)
        for index, item in enumerate(explanation_dicts)
    ]


def select_batch_display_columns(table: Any) -> list[str]:
    columns = list(getattr(table, "columns", []) or [])
    preferred = [
        "SMILES",
        "smiles",
        "explain_canonical_smiles",
        "explain_final_label_ru",
        "explain_short_summary_ru",
        "explain_bbb_score_normalized",
        "explain_gupta_score",
        "explain_pgp_probability",
        "explain_priority_label_ru",
        "explain_negative_factors",
        "explain_uncertainty_level",
        "explain_applicability_level",
        "explain_warnings",
    ]
    selected = [name for name in preferred if name in columns]
    selected.extend([name for name in columns if name.startswith("explain_") and name not in selected])
    return selected


def build_batch_teaching_summary(
    *,
    total: int,
    valid_count: int,
    invalid_count: int,
    final_class_counts: Mapping[str, int],
    priority_counts: Mapping[str, int] | None = None,
    pgp_high_count: int | None = None,
    warning_count: int | None = None,
) -> str:
    return _build_summary_text(
        n_total=total,
        n_valid=valid_count,
        n_invalid=invalid_count,
        final_class_counts=final_class_counts,
    )

# Override with pandas-safe implementation.
def select_batch_display_columns(table: Any) -> list[str]:
    columns = list(getattr(table, "columns", []))
    preferred = [
        "SMILES",
        "smiles",
        "xai_canonical_smiles",
        "xai_final_label",
        "xai_short_summary",
        "xai_bbb_normalized_score",
        "xai_bbb_gupta_score",
        "xai_pgp_probability",
        "xai_batch_priority",
        "xai_negative_factors",
        "xai_uncertainty_level",
        "xai_applicability_level",
        "xai_warnings",
        "explain_canonical_smiles",
        "explain_final_label_ru",
        "explain_short_summary_ru",
        "explain_bbb_score_normalized",
        "explain_pgp_probability",
        "explain_priority_label_ru",
    ]
    selected = [name for name in preferred if name in columns]
    selected.extend([name for name in columns if (name.startswith("xai_") or name.startswith("explain_")) and name not in selected])
    return selected

# Override with additional bbb_high_count / pgp_high_count aliases for Stage 5 UI and tests.
def _summarize_batch_explanations_stage5(batch_df: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    df = _ensure_dataframe(batch_df)
    n_total = int(len(df))
    if n_total == 0:
        summary = _empty_summary()
        summary["bbb_high_count"] = 0
        summary["pgp_high_count"] = 0
        return summary

    n_valid = int(df.get("xai_valid_smiles", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    n_invalid = n_total - n_valid
    final_class_counts = _value_counts(df, "xai_final_class")
    scenario_counts = _value_counts(df, "xai_bbb_pgp_scenario")
    priority_counts = _value_counts(df, "xai_batch_priority")
    uncertainty_counts = _value_counts(df, "xai_uncertainty_level")
    applicability_counts = _value_counts(df, "xai_applicability_level")
    top_negative_factors = _top_semicolon_values(df, "xai_negative_factors", item_key="factor")
    top_warnings = _top_semicolon_values(df, "xai_warnings", item_key="warning")
    bbb_high_count = int((pd.to_numeric(df.get("xai_bbb_normalized_score"), errors="coerce") >= 0.70).sum()) if "xai_bbb_normalized_score" in df else 0
    pgp_high_count = int((pd.to_numeric(df.get("xai_pgp_probability"), errors="coerce") >= 0.65).sum()) if "xai_pgp_probability" in df else 0

    summary_text = _build_summary_text(
        n_total=n_total,
        n_valid=n_valid,
        n_invalid=n_invalid,
        final_class_counts=final_class_counts,
    )
    return {
        "schema_version": BATCH_XAI_SCHEMA_VERSION,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "total": n_total,
        "valid_count": n_valid,
        "invalid_count": n_invalid,
        "bbb_high_count": bbb_high_count,
        "pgp_high_count": pgp_high_count,
        "final_class_counts": final_class_counts,
        "scenario_counts": scenario_counts,
        "bbb_pgp_scenario_counts": scenario_counts,
        "priority_counts": priority_counts,
        "uncertainty_counts": uncertainty_counts,
        "applicability_counts": applicability_counts,
        "top_negative_factors": top_negative_factors,
        "top_warnings": top_warnings,
        "summary_text": summary_text,
        "teaching_summary_ru": summary_text,
        "recommended_next_steps": _build_recommended_next_steps(final_class_counts, priority_counts, n_invalid),
    }


# Stage 6 localized summary override.
def summarize_batch_explanations(batch_df: pd.DataFrame | Sequence[Mapping[str, Any]], lang: str = "ru") -> dict[str, Any]:
    lang = normalize_language(lang)
    summary = _summarize_batch_explanations_stage5(batch_df)
    final_counts = summary.get("final_class_counts") or {}
    summary_text = batch_summary_text(
        total=int(summary.get("total", summary.get("n_total", 0)) or 0),
        valid=int(summary.get("valid_count", summary.get("n_valid", 0)) or 0),
        invalid=int(summary.get("invalid_count", summary.get("n_invalid", 0)) or 0),
        cns=int(final_counts.get("likely_cns_active", 0)),
        efflux=int(final_counts.get("peripheral_action_risk", 0)),
        borderline=int(final_counts.get("uncertain_or_borderline", 0)),
        lang=lang,
    )
    summary["language"] = lang
    summary["summary_text"] = summary_text
    summary["teaching_summary_ru"] = summary_text
    summary["teaching_summary"] = summary_text
    return summary

# Compatibility alias for code that uses the longer name.
summarize_batch_explainability = summarize_batch_explanations

# ---------------------------------------------------------------------------
# Stage 6 multilingual overrides
# ---------------------------------------------------------------------------

from core.i18n import normalize_language as _i18n_normalize_language, t as _i18n_t, batch_priority_label as _i18n_batch_priority_label, batch_summary_text as _i18n_batch_summary_text


def build_explanation_from_current_batch_row(
    row: Mapping[str, Any],
    *,
    smiles_key: str = "SMILES",
    lang: str = "ru",
) -> dict[str, Any]:
    """Build a localized ``explanation_dict`` from one current-app batch row."""
    lang = _i18n_normalize_language(lang)
    source = dict(row or {})
    smi = str(source.get(smiles_key) or source.get("SMILES") or source.get("smiles") or "").strip()
    mol = Chem.MolFromSmiles(smi) if smi else None
    results = {**source, "error": source.get("error") or "Некорректная структура SMILES"} if mol is None else source
    pipeline_result = build_pipeline_result_from_current_app(
        input_smiles=smi,
        mol=mol,
        descriptors=source,
        results=results,
        lang=lang,
    )
    return build_explanation_dict(pipeline_result, lang=lang)


def build_batch_explanation_row(
    input_smiles: str,
    *,
    descriptors: Mapping[str, Any] | None = None,
    results: Mapping[str, Any] | None = None,
    mol: Chem.Mol | None = None,
    include_long_text: bool = False,
    lang: str = "ru",
) -> dict[str, Any]:
    """Build localized compact ``xai_*`` columns for one molecule."""
    lang = _i18n_normalize_language(lang)
    descriptors = descriptors or {}
    results = results or {}
    clean_smiles = str(input_smiles or "").strip()
    if mol is None and clean_smiles:
        mol = Chem.MolFromSmiles(clean_smiles)
    pipeline_result = build_pipeline_result_from_current_app(
        input_smiles=clean_smiles,
        mol=mol,
        descriptors=descriptors,
        results=({**results, "error": results.get("error") or "invalid_smiles"} if mol is None else results),
        lang=lang,
    )
    explanation_dict = build_explanation_dict(pipeline_result, lang=lang)
    return build_batch_explanation_row_from_explanation(explanation_dict, include_long_text=include_long_text, lang=lang)


def build_batch_explanation_row_from_explanation(
    explanation_dict: Mapping[str, Any],
    *,
    include_long_text: bool = False,
    lang: str | None = None,
) -> dict[str, Any]:
    """Build compact localized ``xai_*`` columns from an explanation_dict."""
    lang = _i18n_normalize_language(lang or str(explanation_dict.get("language", "ru")))
    molecule = _as_mapping(explanation_dict.get("molecule"))
    model_outputs = _as_mapping(explanation_dict.get("model_outputs"))
    decision = _as_mapping(explanation_dict.get("decision_explanation"))
    matrix = _as_mapping(explanation_dict.get("bbb_pgp_matrix"))
    applicability = _as_mapping(explanation_dict.get("applicability_domain"))
    uncertainty = _as_mapping(explanation_dict.get("uncertainty"))
    final_class = str(decision.get("final_class") or model_outputs.get("final_cns_class") or "uncertain_or_borderline")
    if not bool(molecule.get("valid")):
        final_class = "invalid_or_error"
    final_label = str(decision.get("final_label") or decision.get("final_label_ru") or _i18n_t(f"final.{final_class}.label", lang))
    if final_label.startswith("final."):
        final_label = FINAL_LABEL_FALLBACK_RU.get(final_class, final_class)
    positive = _factor_names(explanation_dict, "positive")
    negative = _factor_names(explanation_dict, "negative")
    borderline = _factor_names(explanation_dict, "borderline")
    warnings = _join_warnings(molecule.get("warnings"))
    priority = _classify_xai_priority(explanation_dict, final_class) if final_class != "invalid_or_error" else "Invalid/error"
    localized_priority = _i18n_batch_priority_label(priority, lang)
    row: dict[str, Any] = {
        "xai_schema_version": BATCH_XAI_SCHEMA_VERSION,
        "xai_language": lang,
        "xai_valid_smiles": bool(molecule.get("valid")),
        "xai_canonical_smiles": molecule.get("canonical_smiles"),
        "xai_final_class": final_class,
        "xai_final_label": final_label,
        "xai_short_summary": _localized_short_summary(final_label, matrix, negative, borderline, lang),
        "xai_bbb_normalized_score": _round_or_none(model_outputs.get("bbb_classifier_probability")),
        "xai_bbb_gupta_score": _round_or_none(model_outputs.get("bbb_v2_score")),
        "xai_bbb_class": model_outputs.get("bbb_class"),
        "xai_pgp_probability": _round_or_none(model_outputs.get("pgp_probability")),
        "xai_pgp_class": model_outputs.get("pgp_class"),
        "xai_pka_pred": _round_or_none(model_outputs.get("pka_pred")),
        "xai_bbb_pgp_scenario": matrix_current_label(str(matrix.get("current_cell") or "insufficient_data"), lang),
        "xai_matrix_interpretation": matrix.get("current_interpretation"),
        "xai_positive_factors": _join_names(positive),
        "xai_negative_factors": _join_names(negative),
        "xai_borderline_factors": _join_names(borderline),
        "xai_applicability_level": applicability.get("level", "unknown"),
        "xai_uncertainty_level": uncertainty.get("level", "unknown"),
        "xai_warnings": warnings,
        "xai_review_reasons": _build_review_reasons(explanation_dict, final_class, warnings),
        "xai_batch_priority": priority,
        "xai_batch_priority_label": localized_priority,
    }
    if include_long_text:
        row.update(
            {
                "xai_decision_summary_long": decision.get("summary", ""),
                "xai_student_interpretation_long": decision.get("student_interpretation", ""),
                "xai_applicability_message_long": applicability.get("student_message", ""),
                "xai_uncertainty_message_long": uncertainty.get("student_message", ""),
            }
        )
    return row


def build_batch_export_dataframe(
    rows: Sequence[Mapping[str, Any]],
    *,
    smiles_key: str = "SMILES",
    include_long_text: bool = False,
    lang: str = "ru",
) -> pd.DataFrame:
    """Append localized ``xai_*`` columns to current-app batch rows."""
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        base = dict(row)
        smi = str(base.get(smiles_key) or base.get("SMILES") or base.get("smiles") or "").strip()
        mol = Chem.MolFromSmiles(smi) if smi else None
        xai = build_batch_explanation_row(
            smi,
            descriptors=base,
            results=base,
            mol=mol,
            include_long_text=include_long_text,
            lang=lang,
        )
        output_rows.append({**base, **xai})
    return pd.DataFrame(output_rows)


def summarize_batch_explanations(batch_df: pd.DataFrame | Sequence[Mapping[str, Any]], lang: str = "ru") -> dict[str, Any]:
    """Build localized summary statistics for a batch table with ``xai_*`` columns."""
    lang = _i18n_normalize_language(lang)
    df = _ensure_dataframe(batch_df)
    n_total = int(len(df))
    if n_total == 0:
        summary = _empty_summary()
        summary["summary_text"] = _i18n_t("batch.summary_empty", lang)
        summary["teaching_summary"] = summary["summary_text"]
        summary["language"] = lang
        summary["bbb_high_count"] = 0
        summary["pgp_high_count"] = 0
        return summary
    n_valid = int(df.get("xai_valid_smiles", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    n_invalid = n_total - n_valid
    final_class_counts = _value_counts(df, "xai_final_class")
    scenario_counts = _value_counts(df, "xai_bbb_pgp_scenario")
    priority_counts = _value_counts(df, "xai_batch_priority")
    uncertainty_counts = _value_counts(df, "xai_uncertainty_level")
    applicability_counts = _value_counts(df, "xai_applicability_level")
    top_negative_factors = _top_semicolon_values(df, "xai_negative_factors", item_key="factor")
    top_warnings = _top_semicolon_values(df, "xai_warnings", item_key="warning")
    bbb_high_count = int((pd.to_numeric(df.get("xai_bbb_normalized_score"), errors="coerce") >= 0.70).sum()) if "xai_bbb_normalized_score" in df else 0
    pgp_high_count = int((pd.to_numeric(df.get("xai_pgp_probability"), errors="coerce") >= 0.65).sum()) if "xai_pgp_probability" in df else 0
    cns_count = int(final_class_counts.get("likely_cns_active", 0))
    efflux_count = int(final_class_counts.get("peripheral_action_risk", 0))
    summary_text = _i18n_batch_summary_text(
        total=n_total,
        valid=n_valid,
        invalid=n_invalid,
        cns=cns_count,
        efflux=efflux_count,
        borderline=int(final_class_counts.get("uncertain_or_borderline", 0)),
        lang=lang,
    )
    return {
        "schema_version": BATCH_XAI_SCHEMA_VERSION,
        "language": lang,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "total": n_total,
        "valid_count": n_valid,
        "invalid_count": n_invalid,
        "bbb_high_count": bbb_high_count,
        "pgp_high_count": pgp_high_count,
        "final_class_counts": final_class_counts,
        "scenario_counts": scenario_counts,
        "bbb_pgp_scenario_counts": scenario_counts,
        "priority_counts": priority_counts,
        "uncertainty_counts": uncertainty_counts,
        "applicability_counts": applicability_counts,
        "top_negative_factors": top_negative_factors,
        "top_warnings": top_warnings,
        "summary_text": summary_text,
        "teaching_summary": summary_text,
        "teaching_summary_ru": summary_text,
        "recommended_next_steps": _build_recommended_next_steps(final_class_counts, priority_counts, n_invalid),
    }


def build_batch_teaching_summary(
    *,
    total: int,
    valid_count: int,
    invalid_count: int,
    final_class_counts: Mapping[str, int],
    priority_counts: Mapping[str, int] | None = None,
    pgp_high_count: int | None = None,
    warning_count: int | None = None,
    lang: str = "ru",
) -> str:
    lang = _i18n_normalize_language(lang)
    return _i18n_batch_summary_text(
        total=total,
        valid=valid_count,
        invalid=invalid_count,
        cns=int(final_class_counts.get("likely_cns_active", 0)),
        efflux=int(final_class_counts.get("peripheral_action_risk", 0)),
        borderline=int(final_class_counts.get("uncertain_or_borderline", 0)),
        lang=lang,
    )


def _localized_short_summary(final_label: str, matrix: Mapping[str, Any], negative: list[str], borderline: list[str], lang: str) -> str:
    pieces = [str(final_label)]
    if matrix.get("current_interpretation"):
        pieces.append(str(matrix.get("current_interpretation")))
    if negative:
        pieces.append(("Negative factors" if lang == "en" else ("Теріс факторлар" if lang == "kk" else "Негативные факторы")) + ": " + _join_names(negative[:3]))
    if borderline:
        pieces.append(("Borderline" if lang == "en" else ("Шекаралық" if lang == "kk" else "Пограничные")) + ": " + _join_names(borderline[:3]))
    return " | ".join(piece for piece in pieces if piece)

# ---------------------------------------------------------------------------
# Stage 6.2 localization overrides for batch review text
# ---------------------------------------------------------------------------

def _build_recommended_next_steps(
    final_class_counts: Mapping[str, int],
    priority_counts: Mapping[str, int],
    n_invalid: int,
    lang: str = "ru",
) -> list[str]:
    lang = _i18n_normalize_language(lang)
    steps: list[str] = []
    if priority_counts.get("CNS candidate") or priority_counts.get("CNS candidate / caution"):
        steps.append({
            "ru": "Начать ручной разбор с CNS-кандидатов, затем проверить предупреждения о домене применимости.",
            "kk": "Қолмен талдауды CNS-кандидаттарынан бастап, кейін қолданылу домені туралы ескертулерді тексеріңіз.",
            "en": "Start manual review with CNS candidates, then check applicability-domain warnings.",
        }[lang])
    if final_class_counts.get("peripheral_action_risk"):
        steps.append({
            "ru": "Отдельно разобрать молекулы ???????? ???/P-gp как примеры конфликта пассивной проницаемости и эффлюкса.",
            "kk": "???????? ???/P-gp молекулаларын пассивті өткізгіштік пен efflux қақтығысының мысалы ретінде бөлек талдаңыз.",
            "en": "Review ???????? ???/P-gp molecules separately as examples of passive permeability versus efflux conflict.",
        }[lang])
    if final_class_counts.get("uncertain_or_borderline"):
        steps.append({
            "ru": "Для пограничных молекул посмотреть TPSA, LogP, pKa и P-gp probability около порогов.",
            "kk": "Шекаралық молекулалар үшін TPSA, LogP, pKa және P-gp probability мәндерін шектер маңында тексеріңіз.",
            "en": "For borderline molecules, inspect TPSA, LogP, pKa and P-gp probability near the thresholds.",
        }[lang])
    if n_invalid:
        steps.append({
            "ru": "Исправить invalid SMILES до повторного запуска batch-анализа.",
            "kk": "Batch талдауын қайта іске қоспас бұрын invalid SMILES жазбаларын түзетіңіз.",
            "en": "Fix invalid SMILES before rerunning the batch analysis.",
        }[lang])
    if not steps:
        steps.append({
            "ru": "Использовать таблицу xai_* как учебный чек-лист факторов BBB/P-gp для каждой молекулы.",
            "kk": "Әр молекула үшін xai_* кестесін BBB/P-gp факторларының оқу чек-парағы ретінде қолданыңыз.",
            "en": "Use the xai_* table as an educational BBB/P-gp factor checklist for each molecule.",
        }[lang])
    return steps


def _teacher_note_from_xai_priority(label: str, lang: str = "ru") -> str:
    lang = _i18n_normalize_language(lang)
    flag = _priority_flag_from_xai_label(label)
    notes = {
        "ru": {
            "cns_candidate": "Хороший пример согласованного BBB/P-gp профиля для обсуждения CNS-доступности.",
            "cns_candidate_with_caution": "Кандидат интересен, но стоит разобрать предупреждения и пограничные признаки.",
            "pgp_efflux_risk": "Полезный учебный пример конфликта: пассивная проницаемость против активного эффлюкса.",
            "full_barrier": "Пример двойного ограничения: физико-химический BBB-барьер плюс P-gp.",
            "poor_passive_bbb": "Пример неблагоприятного физико-химического профиля для BBB.",
            "outside_domain": "Использовать для обсуждения домена применимости модели и ограничений in silico-прогноза.",
            "invalid_smiles": "Сначала исправить структуру SMILES; модельный вывод недоступен.",
            "unknown": "Пограничный случай для ручного разбора.",
        },
        "kk": {
            "cns_candidate": "CNS қолжетімділігін талқылауға арналған келісілген BBB/P-gp профилінің жақсы мысалы.",
            "cns_candidate_with_caution": "Кандидат қызықты, бірақ ескертулер мен шекаралық белгілерді талдау керек.",
            "pgp_efflux_risk": "Пайдалы оқу мысалы: пассивті өткізгіштік пен белсенді efflux арасындағы қақтығыс.",
            "full_barrier": "Қос шектеу мысалы: физика-химиялық BBB бөгеті және P-gp.",
            "poor_passive_bbb": "BBB үшін қолайсыз физика-химиялық профильдің мысалы.",
            "outside_domain": "Модельдің қолданылу домені мен in silico болжам шектеулерін талқылауға қолданыңыз.",
            "invalid_smiles": "Алдымен SMILES құрылымын түзету керек; модель қорытындысы қолжетімсіз.",
            "unknown": "Қолмен талдауға арналған шекаралық жағдай.",
        },
        "en": {
            "cns_candidate": "A good example of a consistent BBB/P-gp profile for discussing CNS exposure.",
            "cns_candidate_with_caution": "An interesting candidate, but warnings and borderline features should be reviewed.",
            "pgp_efflux_risk": "A useful teaching example of conflict: passive permeability versus active efflux.",
            "full_barrier": "An example of a double limitation: physicochemical BBB barrier plus P-gp.",
            "poor_passive_bbb": "An example of an unfavourable physicochemical profile for BBB.",
            "outside_domain": "Use it to discuss model applicability domain and limitations of in silico prediction.",
            "invalid_smiles": "Fix the SMILES structure first; model interpretation is unavailable.",
            "unknown": "A borderline case for manual review.",
        },
    }
    return notes[lang].get(flag, notes[lang]["unknown"])


def _build_review_reasons(explanation_dict: Mapping[str, Any], final_class: str, warnings: str, lang: str = "ru") -> str:
    lang = _i18n_normalize_language(lang or str(explanation_dict.get("language", "ru")))
    reasons: list[str] = []
    if final_class == "peripheral_action_risk":
        reasons.append({"ru": "конфликт ???????? ???/P-gp: пассивная проницаемость против активного эффлюкса", "kk": "???????? ???/P-gp қақтығысы: пассивті өткізгіштік пен белсенді efflux", "en": "???????? ???/P-gp conflict: passive permeability versus active efflux"}[lang])
    if final_class == "full_barrier":
        reasons.append({"ru": "двойной барьер: плохая пассивная BBB-проницаемость и P-gp efflux", "kk": "қос бөгет: төмен пассивті BBB өткізгіштігі және P-gp efflux", "en": "double barrier: poor passive BBB permeability plus P-gp efflux"}[lang])
    if final_class == "likely_not_bbb_penetrant":
        reasons.append({"ru": "основное ограничение - физико-химический профиль пассивной BBB-проницаемости", "kk": "негізгі шектеу - пассивті BBB өткізгіштігінің физика-химиялық профилі", "en": "main limitation: physicochemical profile for passive BBB permeability"}[lang])
    if final_class == "uncertain_or_borderline":
        reasons.append({"ru": "пограничные вероятности или противоречивые сигналы модели", "kk": "шекаралық ықтималдықтар немесе модельдің қарама-қайшы сигналдары", "en": "borderline probabilities or conflicting model signals"}[lang])
    applicability = _as_mapping(explanation_dict.get("applicability_domain"))
    if str(applicability.get("level") or "") in {"caution", "outside"}:
        reasons.append({"ru": "есть предупреждение о домене применимости", "kk": "қолданылу домені туралы ескерту бар", "en": "applicability-domain warning is present"}[lang])
    if warnings:
        reasons.append({"ru": "есть структурные/model warnings", "kk": "құрылымдық немесе модельдік ескертулер бар", "en": "structural or model warnings are present"}[lang])
    negative = _factor_names(explanation_dict, "negative")
    borderline = _factor_names(explanation_dict, "borderline")
    if negative:
        prefix = {"ru": "негативные факторы: ", "kk": "теріс факторлар: ", "en": "negative factors: "}[lang]
        reasons.append(prefix + ", ".join(negative[:3]))
    elif borderline:
        prefix = {"ru": "пограничные факторы: ", "kk": "шекаралық факторлар: ", "en": "borderline factors: "}[lang]
        reasons.append(prefix + ", ".join(borderline[:3]))
    return "; ".join(reasons)


_stage6_2_build_batch_explanation_row_from_explanation = build_batch_explanation_row_from_explanation


def build_batch_explanation_row_from_explanation(
    explanation_dict: Mapping[str, Any],
    *,
    include_long_text: bool = False,
    lang: str | None = None,
) -> dict[str, Any]:
    lang = _i18n_normalize_language(lang or str(explanation_dict.get("language", "ru")))
    row = _stage6_2_build_batch_explanation_row_from_explanation(explanation_dict, include_long_text=include_long_text, lang=lang)
    warnings = str(row.get("xai_warnings") or "")
    final_class = str(row.get("xai_final_class") or "uncertain_or_borderline")
    row["xai_review_reasons"] = _build_review_reasons(explanation_dict, final_class, warnings, lang=lang)
    if include_long_text and not bool(row.get("xai_valid_smiles", True)):
        row["xai_student_interpretation_long"] = {"ru": "Невалидная структура не должна интерпретироваться как реальный ADMET-прогноз.", "kk": "Валидті емес құрылым нақты ADMET болжамы ретінде түсіндірілмеуі керек.", "en": "An invalid structure must not be interpreted as a real ADMET prediction."}[lang]
        row["xai_applicability_message_long"] = {"ru": "Вне домена: структура не распознана.", "kk": "Доменнен тыс: құрылым танылмады.", "en": "Outside domain: the structure was not parsed."}[lang]
        row["xai_uncertainty_message_long"] = {"ru": "Неопределённость высокая, потому что молекулярный граф не построен.", "kk": "Белгісіздік жоғары, себебі молекулалық граф құрылмады.", "en": "Uncertainty is high because the molecular graph was not built."}[lang]
    return row


_stage6_2_summarize_batch_explanations = summarize_batch_explanations


def summarize_batch_explanations(batch_df: pd.DataFrame | Sequence[Mapping[str, Any]], lang: str = "ru") -> dict[str, Any]:
    lang = _i18n_normalize_language(lang)
    summary = _stage6_2_summarize_batch_explanations(batch_df, lang=lang)
    summary["recommended_next_steps"] = _build_recommended_next_steps(
        summary.get("final_class_counts") or {},
        summary.get("priority_counts") or {},
        int(summary.get("n_invalid", summary.get("invalid_count", 0)) or 0),
        lang=lang,
    )
    return summary

# Keep compatibility alias aligned with the final override.
summarize_batch_explainability = summarize_batch_explanations
