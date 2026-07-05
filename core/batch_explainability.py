"""Batch explainability helpers for BioSynth-EDU mass screening.

Mass screening needs compact, spreadsheet-friendly explanations. This module
turns current flat ADMET rows into ``xai_*`` columns, summarizes a whole batch,
and prepares multi-sheet Excel exports.

The implementation reuses the accepted explanation pipeline:

``current app row -> adapter -> explanation_dict -> compact xai row``

No ML model is trained here, no SMILES is modified, and no result is presented
as experimental validation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from core.explainability import build_explanation_dict
from core.explainability_adapter import build_pipeline_result_from_current_app, normalize_gupta_score_to_educational_bbb_score
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

COMPACT_BATCH_COLUMNS: tuple[str, ...] = (
    "input_smiles",
    "canonical_smiles",
    "valid_smiles",
    "MW",
    "TPSA",
    "LogP",
    "HBD",
    "HBA",
    "Aro_R",
    "HA",
    "RotatableBonds",
    "MWHBN_raw",
    "p_MWHBN",
    "FormalCharge",
    "pKa_pred",
    "pKa_source",
    "caco2_value",
    "caco2_class",
    "bbb_gupta_score_raw",
    "bbb_gupta_score_normalized",
    "bbb_gupta_threshold",
    "bbb_formula_version",
    "bbb_class",
    "bbb_rf_probability",
    "bbb_rf_class",
    "pgp_probability",
    "pgp_class",
    "pgp_status",
    "catmos_value",
    "catmos_class",
    "applicability_level",
    "uncertainty_level",
    "positive_factors",
    "borderline_factors",
    "negative_factors",
    "warning_flags",
    "review_flags",
    "final_screening_class",
    "batch_priority",
)

LONG_BATCH_TEXT_COLUMNS: set[str] = {
    "xai_short_summary",
    "xai_decision_summary_long",
    "xai_student_interpretation_long",
    "xai_applicability_message_long",
    "xai_uncertainty_message_long",
    "xai_matrix_interpretation",
}

FINAL_LABEL_FALLBACK_RU: dict[str, str] = {
    "likely_cns_active": "Вероятно ЦНС-активный профиль",
    "peripheral_action_risk": "Хорошая оценка ГЭБ, но есть риск активного выведения через P-gp",
    "likely_not_bbb_penetrant": "Вероятно слабое прохождение через ГЭБ",
    "full_barrier": "Два ограничения: слабое прохождение через ГЭБ и P-gp",
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


# ---------------------------------------------------------------------------
# Batch summaries and Excel sheets
# ---------------------------------------------------------------------------


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


def build_screening_interpretation_markdown(lang: str = "ru") -> str:
    """Return a standalone interpretation note for compact batch exports."""
    lang = normalize_language(lang)
    return {
        "ru": """# Интерпретация результатов группового ADMET/BBB-скрининга

Групповой отчёт предназначен для предварительного сравнения молекул по набору расчётных ADMET-дескрипторов. Результаты не являются экспериментальным доказательством проникновения через гематоэнцефалический барьер или биологической активности в ЦНС.

## Основные показатели

`Gupta indicator for BBB` — расчётный показатель, используемый для оценки профиля прохождения через гематоэнцефалический барьер. В текущей версии применяется модифицированная формула `gupta_fixed_v2_use_p_mwhbn`, которая использует предсказанный `pKa` и нормализованный компонент `p_MWHBN`.

`RF estimate of BBB passage` — вероятность прохождения через ГЭБ по Random Forest-модели. Это независимая модельная оценка, которую следует рассматривать вместе с Gupta indicator.

`P-gp risk estimate` — расчётная вероятность того, что молекула может быть связана с P-gp-опосредованным выведением. Высокое значение может снижать уверенность в доступности молекулы для ЦНС даже при благоприятном BBB-профиле.

`Caco-2` — расчётный показатель кишечной проницаемости. Он не равен BBB-проницаемости, но помогает оценить общий профиль мембранной проницаемости.

`pKa` — предсказанная кислотно-основная характеристика молекулы. Она важна, потому что степень ионизации при физиологическом pH влияет на пассивное прохождение через мембраны.

## Как читать итоговый класс

`likely_bbb_penetrant_profile` означает, что молекула имеет благоприятный расчётный профиль для прохождения через ГЭБ.

`borderline_bbb_profile` означает, что результат неоднозначен: часть дескрипторов благоприятна, но есть факторы риска или пограничные значения.

`likely_low_bbb_profile` означает, что расчётные признаки указывают на слабое пассивное прохождение через ГЭБ.

`high_pgp_risk` означает, что даже при приемлемом BBB-профиле возможен риск активного выведения через P-gp.

`outside_applicability_domain` означает, что молекула находится вне надёжной области применения модели, поэтому результат требует ручной проверки.

## Важные ограничения

Карбоновые кислоты, аминокислоты, цвиттер-ионные структуры, полифенолы, очень простые углеводороды и молекулы с экстремальным LogP требуют осторожной интерпретации.

Для таких соединений высокий численный BBB-score не должен автоматически трактоваться как доказательство хорошего проникновения в ЦНС.

Групповой скрининг следует использовать как инструмент ранжирования и отбора кандидатов для дальнейшей проверки, а не как окончательное заключение.
""",
        "kk": """# Топтық ADMET/BBB-скрининг нәтижелерін түсіндіру

Топтық есеп молекулаларды есептік ADMET-дескрипторлар бойынша алдын ала салыстыруға арналған. Бұл нәтижелер гематоэнцефалдық бөгеттен өтуін немесе ОЖЖ-дегі биологиялық белсенділігін эксперименттік түрде дәлелдемейді.

## Негізгі көрсеткіштер

`Gupta indicator for BBB` — гематоэнцефалдық бөгеттен өту профилін бағалауға арналған есептік көрсеткіш. Қазіргі нұсқада `gupta_fixed_v2_use_p_mwhbn` модификацияланған формуласы қолданылады. Ол болжанған `pKa` мәнін және нормаланған `p_MWHBN` компонентін пайдаланады.

`RF estimate of BBB passage` — Random Forest моделі бойынша ГЭБ-тен өту ықтималдығы. Бұл Gupta көрсеткішінен бөлек модельдік баға, сондықтан оны басқа дескрипторлармен бірге қарастыру керек.

`P-gp risk estimate` — молекуланың P-gp арқылы шығарылу қаупінің есептік ықтималдығы. Бұл көрсеткіш жоғары болса, BBB-профиль жақсы болғанымен, ОЖЖ-ге қолжетімділікке сенімділік төмендеуі мүмкін.

`Caco-2` — ішектік өткізгіштіктің есептік көрсеткіші. Ол BBB-өткізгіштікпен бірдей емес, бірақ мембраналық өткізгіштік профилін жалпы бағалауға көмектеседі.

`pKa` — молекуланың болжанған қышқыл-негіздік сипаттамасы. Бұл көрсеткіш маңызды, себебі физиологиялық pH жағдайындағы иондалу дәрежесі мембрана арқылы пассивті өтуге әсер етеді.

## Қорытынды класты қалай оқу керек

`likely_bbb_penetrant_profile` молекуланың ГЭБ-тен өтуге қолайлы есептік профилі бар екенін білдіреді.

`borderline_bbb_profile` нәтиженің бірмәнді емес екенін білдіреді: кейбір дескрипторлар қолайлы, бірақ қауіп факторлары немесе шекаралық мәндер бар.

`likely_low_bbb_profile` есептік белгілер ГЭБ арқылы пассивті өтудің әлсіз болуы мүмкін екенін көрсетеді.

`high_pgp_risk` BBB-профиль қолайлы болса да, P-gp арқылы белсенді шығарылу қаупі болуы мүмкін екенін білдіреді.

`outside_applicability_domain` молекула модельдің сенімді қолдану аймағынан тыс екенін білдіреді, сондықтан нәтижені қолмен тексеру қажет.

## Маңызды шектеулер

Карбон қышқылдары, аминқышқылдары, цвиттер-иондық құрылымдар, полифенолдар, өте қарапайым көмірсутектер және LogP мәні шектен тыс молекулалар сақтықпен түсіндірілуі керек.

Мұндай қосылыстар үшін жоғары BBB-score автоматты түрде ОЖЖ-ге жақсы өтеді деген қорытынды бермейді.

Топтық скринингті соңғы қорытынды ретінде емес, әрі қарай тексеруге арналған кандидаттарды алдын ала іріктеу және ранжирлеу құралы ретінде қолдану керек.
""",
        "en": """# Interpretation of Batch ADMET/BBB Screening Results

The batch report is intended for preliminary comparison of molecules using calculated ADMET descriptors. These results are not experimental proof of blood-brain barrier penetration or CNS biological activity.

## Main indicators

`Gupta indicator for BBB` is a calculated score used to estimate the blood-brain barrier penetration profile. The current version uses the modified formula `gupta_fixed_v2_use_p_mwhbn`, which includes predicted `pKa` and the normalized `p_MWHBN` component.

`RF estimate of BBB passage` is the predicted probability of BBB passage from a Random Forest model. It should be interpreted together with the Gupta indicator and other descriptors.

`P-gp risk estimate` is the estimated probability that the molecule may be affected by P-gp-mediated efflux. A high value can reduce confidence in CNS availability even when the BBB profile appears favorable.

`Caco-2` is a calculated intestinal permeability indicator. It is not equivalent to BBB permeability, but it helps describe the general membrane permeability profile.

`pKa` is the predicted acid-base property of the molecule. It is important because the ionization state at physiological pH affects passive membrane penetration.

## How to read the final class

`likely_bbb_penetrant_profile` means that the molecule has a favorable calculated profile for BBB penetration.

`borderline_bbb_profile` means that the result is uncertain: some descriptors are favorable, but risk factors or borderline values are present.

`likely_low_bbb_profile` means that the calculated properties suggest weak passive BBB penetration.

`high_pgp_risk` means that P-gp-mediated efflux may be a concern even if the passive BBB profile is acceptable.

`outside_applicability_domain` means that the molecule is outside the reliable applicability domain of the model and requires manual review.

## Important limitations

Carboxylic acids, amino acids, zwitterionic structures, polyphenols, very simple hydrocarbons, and molecules with extreme LogP values require cautious interpretation.

For such compounds, a high numerical BBB score should not automatically be interpreted as strong evidence of CNS penetration.

Batch screening should be used as a ranking and prioritization tool for further review, not as a final conclusion.
""",
    }[lang].strip() + "\n"


def build_batch_excel_sheets(batch_df: pd.DataFrame, summary: Mapping[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Prepare named DataFrames for multi-sheet Excel export."""
    summary = summary or summarize_batch_explanations(batch_df)
    return {
        "screening_results": batch_df,
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


def build_compact_screening_row(
    source: Mapping[str, Any],
    *,
    smiles_key: str = "SMILES",
    lang: str = "ru",
) -> dict[str, Any]:
    """Build one strict numeric screening row for CSV/XLSX export."""
    lang = normalize_language(lang)
    data = dict(source or {})
    input_smiles = str(data.get(smiles_key) or data.get("SMILES") or data.get("smiles") or "").strip()
    mol = Chem.MolFromSmiles(input_smiles) if input_smiles else None
    valid = mol is not None and not data.get("error")

    if not valid:
        row = {column: "N/A" for column in COMPACT_BATCH_COLUMNS}
        row.update(
            {
                "input_smiles": input_smiles or "N/A",
                "canonical_smiles": "N/A",
                "valid_smiles": False,
                "warning_flags": "invalid_structure",
                "review_flags": "invalid_structure",
                "final_screening_class": "invalid_structure",
                "batch_priority": "invalid_structure",
            }
        )
        return row

    assert mol is not None
    explanation = build_explanation_from_current_batch_row(data, smiles_key=smiles_key, lang=lang)
    model_outputs = _as_mapping(explanation.get("model_outputs"))
    applicability = _as_mapping(explanation.get("applicability_domain"))
    uncertainty = _as_mapping(explanation.get("uncertainty"))
    gupta = _as_mapping(data.get("gupta_components") or model_outputs.get("gupta_components"))

    mw = _first_value(data, "MW", "mw")
    tpsa = _first_value(data, "TPSA", "tpsa")
    logp = _first_value(data, "LogP", "logp")
    hbd = _first_value(data, "HBD", "h_donors")
    hba = _first_value(data, "HBA", "h_acceptors")
    aro_r = _first_value(data, "Aro_R", "aro_r")
    ha = _first_value(data, "HA", "ha")
    rotatable = Descriptors.NumRotatableBonds(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    pka_pred = _first_value(data, "pKa_predicted", "pka_pred", "pKa_pred")
    pka_source = _normalise_pka_source(_first_value(data, "pka_source"))
    gupta_raw = _first_value(gupta, "score_raw", "score")
    gupta_score = _first_value(data, "gupta_bbb_score", "gupta_corrected_score")
    gupta_norm = normalize_gupta_score_to_educational_bbb_score(gupta_score)
    gupta_threshold = _first_value(gupta, "threshold", default=3.0)
    bbb_rf_prob = _first_value(data, "bbb_rf_probability")
    pgp_prob = _first_value(data, "pgp_probability")
    caco2_value = _first_value(data, "caco2_logpapp")
    catmos_value = _first_value(data, "catmos_score", "catmos_ld50")

    structural_flags = _structural_warning_flags(mol, logp=logp, applicability_level=str(applicability.get("level") or "unknown"))
    positive, borderline, negative = _descriptor_factor_flags(
        mw=mw,
        tpsa=tpsa,
        logp=logp,
        hbd=hbd,
        hba=hba,
        gupta_score=gupta_score,
        gupta_threshold=gupta_threshold,
        pgp_probability=pgp_prob,
    )
    warning_flags = list(structural_flags)
    if pka_source == "default_8.81":
        warning_flags.append("default_pka_used")

    review_flags = _review_flags(
        warning_flags=warning_flags,
        applicability_level=str(applicability.get("level") or "unknown"),
        uncertainty_level=str(uncertainty.get("level") or "unknown"),
        gupta_score=gupta_score,
        gupta_threshold=gupta_threshold,
        pgp_probability=pgp_prob,
        bbb_rf_probability=bbb_rf_prob,
    )
    final_class = _final_screening_class(
        gupta_score=gupta_score,
        gupta_threshold=gupta_threshold,
        pgp_probability=pgp_prob,
        warning_flags=warning_flags,
        review_flags=review_flags,
        applicability_level=str(applicability.get("level") or "unknown"),
    )

    row = {
        "input_smiles": input_smiles,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "valid_smiles": True,
        "MW": _round_or_na(mw, 3),
        "TPSA": _round_or_na(tpsa, 3),
        "LogP": _round_or_na(logp, 3),
        "HBD": _na(hbd),
        "HBA": _na(hba),
        "Aro_R": _na(aro_r),
        "HA": _na(ha),
        "RotatableBonds": _na(rotatable),
        "MWHBN_raw": _round_or_na(_first_value(gupta, "mwhbn_raw", "MWHBN"), 6),
        "p_MWHBN": _round_or_na(_first_value(gupta, "p_mwhbn"), 6),
        "FormalCharge": _na(formal_charge),
        "pKa_pred": _round_or_na(pka_pred, 3),
        "pKa_source": pka_source,
        "caco2_value": _round_or_na(caco2_value, 3),
        "caco2_class": _na(_first_value(data, "caco2_status")),
        "bbb_gupta_score_raw": _round_or_na(gupta_raw, 6),
        "bbb_gupta_score_normalized": _round_or_na(gupta_norm, 3),
        "bbb_gupta_threshold": _round_or_na(gupta_threshold, 3),
        "bbb_formula_version": _na(_first_value(data, "gupta_formula_version") or _first_value(gupta, "formula_version")),
        "bbb_class": _bbb_class(gupta_score, gupta_threshold),
        "bbb_rf_probability": _round_or_na(bbb_rf_prob, 3),
        "bbb_rf_class": _na(_first_value(data, "bbb_rf_class")),
        "pgp_probability": _round_or_na(pgp_prob, 3),
        "pgp_class": _na(_first_value(data, "pgp_substrate_class", "pgp_class")),
        "pgp_status": _na(_first_value(data, "pgp_status")),
        "catmos_value": _round_or_na(catmos_value, 3),
        "catmos_class": _na(_first_value(data, "catmos_unit_status", "catmos_model_status")),
        "applicability_level": _na(applicability.get("level", "unknown")),
        "uncertainty_level": _na(uncertainty.get("level", "unknown")),
        "positive_factors": _join_flags(positive),
        "borderline_factors": _join_flags(borderline),
        "negative_factors": _join_flags(negative),
        "warning_flags": _join_flags(warning_flags),
        "review_flags": _join_flags(review_flags),
        "final_screening_class": final_class,
        "batch_priority": _batch_priority_from_final_class(final_class, review_flags),
    }
    return {column: _na(row.get(column)) for column in COMPACT_BATCH_COLUMNS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _na(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, float) and value != value:
        return "N/A"
    if isinstance(value, str) and value.strip() == "":
        return "N/A"
    return value


def _round_or_na(value: Any, digits: int = 3) -> Any:
    numeric = _safe_float(value)
    if numeric is None:
        return "N/A"
    return round(numeric, digits)


def _first_value(source: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source:
            value = source[key]
            if value is not None and not (isinstance(value, str) and value == ""):
                return value
    return default


def _join_flags(flags: Sequence[str] | None) -> str:
    unique = list(dict.fromkeys(str(flag) for flag in flags or [] if str(flag).strip()))
    return "; ".join(unique) if unique else "N/A"


def _normalise_pka_source(source: Any) -> str:
    raw = str(source or "").strip().lower()
    if raw in {"model", "predicted"}:
        return "predicted"
    if raw in {"default", "fallback", "fallback_error", "fallback_invalid_mol"}:
        return "default_8.81"
    if raw in {"user", "user_input"}:
        return "user_input"
    return "N/A"


def _has_substructure(mol: Chem.Mol, smarts: str) -> bool:
    pattern = Chem.MolFromSmarts(smarts)
    return bool(pattern is not None and mol.HasSubstructMatch(pattern))


def _structural_warning_flags(mol: Chem.Mol, *, logp: Any, applicability_level: str) -> list[str]:
    flags: list[str] = []
    if _has_substructure(mol, "[CX3](=O)[OX2H1,O-]"):
        flags.append("carboxylic_acid_flag")
    has_carboxyl = "carboxylic_acid_flag" in flags
    has_amine = _has_substructure(mol, "[NX3,NX4+]")
    if (has_carboxyl and has_amine) or (_has_substructure(mol, "[N+;!$([N+](=O)[O-])]" ) and _has_substructure(mol, "[O-]")):
        flags.append("amino_acid_or_zwitterion_flag")
    phenol_pattern = Chem.MolFromSmarts("c[OX2H]")
    if phenol_pattern is not None and len(mol.GetSubstructMatches(phenol_pattern)) >= 2:
        flags.append("polyphenol_flag")
    atom_nums = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    if atom_nums and atom_nums.issubset({6}) and mol.GetNumHeavyAtoms() <= 12:
        flags.append("simple_hydrocarbon_flag")
    logp_value = _safe_float(logp)
    if logp_value is not None and logp_value > 5:
        flags.append("very_high_logp_flag")
    if logp_value is not None and logp_value < 0:
        flags.append("very_low_logp_flag")
    if applicability_level == "outside":
        flags.append("outside_applicability_domain")
    return flags


def _descriptor_factor_flags(
    *,
    mw: Any,
    tpsa: Any,
    logp: Any,
    hbd: Any,
    hba: Any,
    gupta_score: Any,
    gupta_threshold: Any,
    pgp_probability: Any,
) -> tuple[list[str], list[str], list[str]]:
    positive: list[str] = []
    borderline: list[str] = []
    negative: list[str] = []
    mw_v = _safe_float(mw)
    tpsa_v = _safe_float(tpsa)
    logp_v = _safe_float(logp)
    hbd_v = _safe_float(hbd)
    hba_v = _safe_float(hba)
    gupta_v = _safe_float(gupta_score)
    threshold_v = _safe_float(gupta_threshold) or 3.0
    pgp_v = _safe_float(pgp_probability)

    if mw_v is not None:
        (positive if mw_v <= 450 else borderline if mw_v <= 550 else negative).append("acceptable_MW" if mw_v <= 450 else "borderline_MW" if mw_v <= 550 else "high_MW")
    if tpsa_v is not None:
        (positive if tpsa_v <= 70 else borderline if tpsa_v <= 90 else negative).append("low_TPSA" if tpsa_v <= 70 else "borderline_TPSA" if tpsa_v <= 90 else "high_TPSA")
    if logp_v is not None:
        (positive if 1 <= logp_v <= 4 else borderline if 0 <= logp_v <= 5 else negative).append("acceptable_LogP" if 1 <= logp_v <= 4 else "borderline_LogP" if 0 <= logp_v <= 5 else "extreme_LogP")
    if hbd_v is not None:
        (positive if hbd_v <= 1 else borderline if hbd_v <= 3 else negative).append("low_HBD" if hbd_v <= 1 else "borderline_HBD" if hbd_v <= 3 else "high_HBD")
    if hba_v is not None:
        (positive if hba_v <= 6 else borderline if hba_v <= 9 else negative).append("low_HBA" if hba_v <= 6 else "borderline_HBA" if hba_v <= 9 else "high_HBA")
    if gupta_v is not None:
        if gupta_v >= threshold_v + 0.5:
            positive.append("gupta_above_threshold")
        elif gupta_v >= threshold_v - 0.5:
            borderline.append("gupta_near_threshold")
        else:
            negative.append("gupta_below_threshold")
    if pgp_v is not None:
        if pgp_v < 0.35:
            positive.append("low_Pgp")
        elif pgp_v < 0.65:
            borderline.append("borderline_Pgp")
        else:
            negative.append("high_Pgp")
    return positive, borderline, negative


def _review_flags(
    *,
    warning_flags: Sequence[str],
    applicability_level: str,
    uncertainty_level: str,
    gupta_score: Any,
    gupta_threshold: Any,
    pgp_probability: Any,
    bbb_rf_probability: Any,
) -> list[str]:
    flags = list(warning_flags)
    if applicability_level in {"caution", "outside"}:
        flags.append(f"{applicability_level}_applicability")
    if uncertainty_level in {"medium", "high"}:
        flags.append(f"{uncertainty_level}_uncertainty")
    gupta = _safe_float(gupta_score)
    threshold = _safe_float(gupta_threshold) or 3.0
    pgp = _safe_float(pgp_probability)
    bbb_rf = _safe_float(bbb_rf_probability)
    if gupta is not None and abs(gupta - threshold) <= 0.5:
        flags.append("borderline_gupta_score")
    if pgp is not None and pgp >= 0.65:
        flags.append("high_pgp_risk")
    if gupta is not None and bbb_rf is not None and ((gupta >= threshold and bbb_rf < 0.5) or (gupta < threshold and bbb_rf >= 0.7)):
        flags.append("bbb_model_disagreement")
    return flags


def _bbb_class(gupta_score: Any, threshold: Any) -> str:
    score = _safe_float(gupta_score)
    limit = _safe_float(threshold) or 3.0
    if score is None:
        return "N/A"
    return "bbb_high" if score >= limit else "bbb_low"


def _final_screening_class(
    *,
    gupta_score: Any,
    gupta_threshold: Any,
    pgp_probability: Any,
    warning_flags: Sequence[str],
    review_flags: Sequence[str],
    applicability_level: str,
) -> str:
    if applicability_level == "outside" or "outside_applicability_domain" in warning_flags:
        return "outside_applicability_domain"
    gupta = _safe_float(gupta_score)
    threshold = _safe_float(gupta_threshold) or 3.0
    pgp = _safe_float(pgp_probability)
    if gupta is None:
        return "borderline_bbb_profile"
    if pgp is not None and pgp >= 0.65 and gupta >= threshold:
        return "high_pgp_risk"
    strict_warning_flags = [flag for flag in warning_flags if flag not in {"default_pka_used"}]
    if strict_warning_flags or review_flags:
        if gupta < threshold - 0.5:
            return "likely_low_bbb_profile"
        return "borderline_bbb_profile"
    if gupta >= threshold:
        return "likely_bbb_penetrant_profile"
    if gupta >= threshold - 0.5:
        return "borderline_bbb_profile"
    return "likely_low_bbb_profile"


def _batch_priority_from_final_class(final_class: str, review_flags: Sequence[str]) -> str:
    if final_class == "invalid_structure":
        return "invalid_structure"
    if final_class == "outside_applicability_domain":
        return "manual_review"
    if review_flags:
        return "review"
    if final_class == "likely_bbb_penetrant_profile":
        return "candidate"
    if final_class == "high_pgp_risk":
        return "pgp_risk_review"
    if final_class == "borderline_bbb_profile":
        return "borderline_review"
    return "low_priority"


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
        f"кандидаты для ЦНС: {cns}; конфликт ГЭБ/P-gp: {efflux}; "
        f"слабое прохождение через ГЭБ: {low_bbb}; два ограничения: {full_barrier}; "
        f"пограничных случаев: {review}."
    )


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
    columns = list(getattr(table, "columns", []))
    preferred = [
        "input_smiles",
        "canonical_smiles",
        "valid_smiles",
        "MW",
        "TPSA",
        "LogP",
        "pKa_pred",
        "pKa_source",
        "bbb_gupta_score_raw",
        "bbb_gupta_score_normalized",
        "bbb_formula_version",
        "bbb_class",
        "bbb_rf_probability",
        "pgp_probability",
        "pgp_status",
        "applicability_level",
        "uncertainty_level",
        "warning_flags",
        "review_flags",
        "final_screening_class",
        "batch_priority",
    ]
    selected = [name for name in preferred if name in columns]
    selected.extend([name for name in COMPACT_BATCH_COLUMNS if name in columns and name not in selected])
    return selected

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
        "xai_review_reasons": _build_review_reasons(explanation_dict, final_class, warnings, lang=lang),
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
    """Build strict compact screening rows for CSV/XLSX export."""
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        base = dict(row)
        compact = build_compact_screening_row(base, smiles_key=smiles_key, lang=lang)
        if include_long_text:
            smi = str(base.get(smiles_key) or base.get("SMILES") or base.get("smiles") or "").strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            xai = build_batch_explanation_row(smi, descriptors=base, results=base, mol=mol, include_long_text=True, lang=lang)
            compact.update({key: value for key, value in xai.items() if key not in LONG_BATCH_TEXT_COLUMNS})
        output_rows.append(compact)
    df = pd.DataFrame(output_rows)
    for column in COMPACT_BATCH_COLUMNS:
        if column not in df.columns:
            df[column] = "N/A"
    compact_columns = list(COMPACT_BATCH_COLUMNS)
    extra_columns = [column for column in df.columns if column not in compact_columns and column not in LONG_BATCH_TEXT_COLUMNS]
    return df[compact_columns + extra_columns].fillna("N/A")


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
    valid_column = "valid_smiles" if "valid_smiles" in df.columns else "xai_valid_smiles"
    n_valid = int(df.get(valid_column, pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    n_invalid = n_total - n_valid
    final_column = "final_screening_class" if "final_screening_class" in df.columns else "xai_final_class"
    priority_column = "batch_priority" if "batch_priority" in df.columns else "xai_batch_priority"
    uncertainty_column = "uncertainty_level" if "uncertainty_level" in df.columns else "xai_uncertainty_level"
    applicability_column = "applicability_level" if "applicability_level" in df.columns else "xai_applicability_level"
    final_class_counts = _value_counts(df, final_column)
    scenario_counts = {}
    priority_counts = _value_counts(df, priority_column)
    uncertainty_counts = _value_counts(df, uncertainty_column)
    applicability_counts = _value_counts(df, applicability_column)
    top_negative_factors = _top_semicolon_values(df, "negative_factors", item_key="factor")
    top_warnings = _top_semicolon_values(df, "warning_flags", item_key="warning")
    bbb_high_count = int((pd.to_numeric(df.get("bbb_gupta_score_normalized"), errors="coerce") >= 0.70).sum()) if "bbb_gupta_score_normalized" in df else 0
    pgp_high_count = int((pd.to_numeric(df.get("pgp_probability"), errors="coerce") >= 0.65).sum()) if "pgp_probability" in df else 0
    cns_count = int(final_class_counts.get("likely_bbb_penetrant_profile", 0))
    efflux_count = int(final_class_counts.get("high_pgp_risk", 0))
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
        "recommended_next_steps": _build_recommended_next_steps(final_class_counts, priority_counts, n_invalid, lang=lang),
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
            "ru": "Начать ручной разбор с кандидатов для ЦНС, затем проверить предупреждения о надёжности модели.",
            "kk": "Қолмен талдауды ОЖЖ кандидаттарынан бастап, кейін модель сенімділігі туралы ескертулерді тексеріңіз.",
            "en": "Start manual review with CNS candidates, then check applicability-domain warnings.",
        }[lang])
    if final_class_counts.get("peripheral_action_risk"):
        steps.append({
            "ru": "Отдельно разобрать молекулы с конфликтом ГЭБ/P-gp как примеры пассивного прохождения и активного выведения.",
            "kk": "Қан-ми тосқауылы/P-gp қайшылығы бар молекулаларды пассивті өту мен белсенді шығарылу мысалы ретінде бөлек талдаңыз.",
            "en": "Review BBB/P-gp conflict molecules separately as examples of passive permeability versus active removal.",
        }[lang])
    if final_class_counts.get("uncertain_or_borderline"):
        steps.append({
            "ru": "Для пограничных молекул посмотреть TPSA, LogP, pKa и оценку P-gp около порогов.",
            "kk": "Шекаралық молекулалар үшін TPSA, LogP, pKa және P-gp бағасын шектер маңында тексеріңіз.",
            "en": "For borderline molecules, inspect TPSA, LogP, pKa and P-gp probability near the thresholds.",
        }[lang])
    if n_invalid:
        steps.append({
            "ru": "Исправить некорректные SMILES до повторного запуска массового анализа.",
            "kk": "Жаппай талдауды қайта іске қоспас бұрын қате SMILES жазбаларын түзетіңіз.",
            "en": "Fix invalid SMILES before rerunning the batch analysis.",
        }[lang])
    if not steps:
        steps.append({
            "ru": "Использовать таблицу xai_* как учебный чек-лист факторов ГЭБ/P-gp для каждой молекулы.",
            "kk": "Әр молекула үшін xai_* кестесін қан-ми тосқауылы/P-gp факторларының оқу чек-парағы ретінде қолданыңыз.",
            "en": "Use the xai_* table as an educational BBB/P-gp factor checklist for each molecule.",
        }[lang])
    return steps


def _teacher_note_from_xai_priority(label: str, lang: str = "ru") -> str:
    lang = _i18n_normalize_language(lang)
    flag = _priority_flag_from_xai_label(label)
    notes = {
        "ru": {
            "cns_candidate": "Хороший пример согласованного ГЭБ/P-gp профиля для обсуждения доступности для ЦНС.",
            "cns_candidate_with_caution": "Кандидат интересен, но стоит разобрать предупреждения и пограничные признаки.",
            "pgp_efflux_risk": "Полезный учебный пример конфликта: пассивное прохождение против активного выведения.",
            "full_barrier": "Пример двойного ограничения: физико-химический барьер ГЭБ плюс P-gp.",
            "poor_passive_bbb": "Пример неблагоприятного физико-химического профиля для прохождения через ГЭБ.",
            "outside_domain": "Использовать для обсуждения надёжности модели и ограничений in silico-прогноза.",
            "invalid_smiles": "Сначала исправить структуру SMILES; модельный вывод недоступен.",
            "unknown": "Пограничный случай для ручного разбора.",
        },
        "kk": {
            "cns_candidate": "ОЖЖ қолжетімділігін талқылауға арналған келісілген қан-ми тосқауылы/P-gp профилінің жақсы мысалы.",
            "cns_candidate_with_caution": "Кандидат қызықты, бірақ ескертулер мен шекаралық белгілерді талдау керек.",
            "pgp_efflux_risk": "Пайдалы оқу мысалы: пассивті өту мен белсенді шығарылу арасындағы қайшылық.",
            "full_barrier": "Қос шектеу мысалы: қан-ми тосқауылының физика-химиялық шектеуі және P-gp.",
            "poor_passive_bbb": "Қан-ми тосқауылынан өту үшін қолайсыз физика-химиялық профильдің мысалы.",
            "outside_domain": "Модель сенімділігі мен in silico болжам шектеулерін талқылауға қолданыңыз.",
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
        reasons.append({"ru": "конфликт ГЭБ/P-gp: пассивное прохождение против активного выведения", "kk": "қан-ми тосқауылы/P-gp қайшылығы: пассивті өту мен белсенді шығарылу", "en": "BBB/P-gp conflict: passive permeability versus active efflux"}[lang])
    if final_class == "full_barrier":
        reasons.append({"ru": "двойное ограничение: слабое прохождение через ГЭБ и активное выведение через P-gp", "kk": "қос шектеу: қан-ми тосқауылынан әлсіз өту және P-gp арқылы белсенді шығарылу", "en": "double barrier: poor passive BBB permeability plus P-gp efflux"}[lang])
    if final_class == "likely_not_bbb_penetrant":
        reasons.append({"ru": "основное ограничение - физико-химический профиль прохождения через ГЭБ", "kk": "негізгі шектеу - қан-ми тосқауылынан өтудің физика-химиялық профилі", "en": "main limitation: physicochemical profile for passive BBB permeability"}[lang])
    if final_class == "uncertain_or_borderline":
        reasons.append({"ru": "пограничные вероятности или противоречивые сигналы модели", "kk": "шекаралық ықтималдықтар немесе модельдің қарама-қайшы сигналдары", "en": "borderline probabilities or conflicting model signals"}[lang])
    applicability = _as_mapping(explanation_dict.get("applicability_domain"))
    if str(applicability.get("level") or "") in {"caution", "outside"}:
        reasons.append({"ru": "есть предупреждение о надёжности модели для этой молекулы", "kk": "бұл молекула үшін модель сенімділігі туралы ескерту бар", "en": "applicability-domain warning is present"}[lang])
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


summarize_batch_explainability = summarize_batch_explanations
