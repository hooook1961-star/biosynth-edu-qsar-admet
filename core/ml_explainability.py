"""ML explainability helpers for BioSynth-EDU.

This module adds local feature-contribution explanations for selected runtime
ML models. Rule-based educational explanations remain the primary
student-facing interpretation.

The first supported runtime models are:
* ``rf_pgp_model``     -- selected P-gp active-removal classifier, usually hybrid_pgp.
* ``rf_bbb_model``     -- supplementary BBB RandomForest classifier, usually
                          morgan_2048.

Fingerprint bits are aggregated into educational feature groups.  Individual
Morgan/MACCS bits are still exposed in an advanced section, but the default
commentary talks about groups such as Morgan fingerprint, MACCS keys, LogP,
TPSA, HBD/HBA and Gasteiger charges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from core.features import (
    build_bbb_rf_features,
    build_pgp_features,
    get_positive_class_probability,
    validate_feature_vector,
)
from core.i18n import normalize_language
from core.runtime_models import (
    get_runtime_context,
    get_runtime_entry,
    get_runtime_model,
    get_runtime_threshold,
    is_runtime_disabled,
)

SUPPORTED_ML_EXPLAINABLE_MODELS = ("rf_pgp_model", "rf_bbb_model")

MODEL_META = {
    "rf_pgp_model": {
        "short_key": "pgp",
        "task": "classification",
        "feature_kind": "hybrid_pgp",
        "positive_label": 1,
    },
    "rf_bbb_model": {
        "short_key": "bbb_rf",
        "task": "classification",
        "feature_kind": "morgan_2048",
        "positive_label": 1,
    },
}

PHYS_LABELS = ["MW", "TPSA", "HBD", "HBA", "GasteigerMax", "GasteigerMin", "LogP"]

ML_TEXT = {
    "ru": {
        "model.rf_pgp_model": "P-gp RF v2",
        "model.rf_bbb_model": "BBB RF v2",
        "status.ok": "объяснение построено",
        "status.unavailable": "модель недоступна",
        "status.disabled": "модель отключена runtime-selection",
        "method.shap": "SHAP TreeExplainer",
        "method.feature_importance_fallback": "fallback: feature_importances × значение признака",
        "group.morgan": "Morgan fingerprint bits",
        "group.maccs": "MACCS keys",
        "group.MW": "Молекулярная масса",
        "group.TPSA": "TPSA",
        "group.HBD": "HBD",
        "group.HBA": "HBA",
        "group.GasteigerMax": "Gasteiger max",
        "group.GasteigerMin": "Gasteiger min",
        "group.LogP": "LogP",
        "group.other": "Другие признаки",
        "direction.positive": "усиливает текущий вывод модели",
        "direction.negative": "ослабляет текущий вывод модели",
        "direction.neutral": "вклад мал или нейтрален",
        "direction.importance_only": "важность без знака",
        "disclaimer": (
            "Локальное объяснение показывает, какие признаки сильнее повлияли на вывод модели. "
            "Для структурных отпечатков отдельные биты трудно напрямую связать с химическим смыслом, "
            "поэтому BioSynth-EDU объединяет их в учебные группы. Это объяснение не является "
            "экспериментальным доказательством."
        ),
        "fingerprint_note": (
            "Большая доля сигнала приходится на структурные отпечатки молекулы. Это означает, что модель "
            "использует фрагменты структуры, но отдельный технический бит не всегда имеет простое учебное название."
        ),
        "physchem_note": (
            "Существенный вклад внесли физико-химические признаки. Их можно сопоставлять с обычным "
            "учебным разбором: LogP, TPSA, MW, HBD/HBA и частичные заряды."
        ),
        "bbb_note": (
            "Это дополнительная статистическая проверка прохождения через ГЭБ. Основной учебный разбор "
            "лучше читать по дескрипторам и формуле Gupta."
        ),
        "pgp_note": (
            "P-gp рассматривается как риск активного выведения молекулы из клеток барьера. Высокий сигнал "
            "может снижать доступность для ЦНС."
        ),
        "unavailable_reason": "ML-разбор недоступен: модель не загружена, отключена или SMILES некорректен.",
    },
    "kk": {
        "model.rf_pgp_model": "P-gp RF v2",
        "model.rf_bbb_model": "BBB RF v2",
        "status.ok": "түсіндірме құрылды",
        "status.unavailable": "модель қолжетімсіз",
        "status.disabled": "модель runtime-selection арқылы өшірілген",
        "method.shap": "SHAP TreeExplainer",
        "method.feature_importance_fallback": "fallback: feature_importances × белгі мәні",
        "group.morgan": "Morgan fingerprint биттері",
        "group.maccs": "MACCS кілттері",
        "group.MW": "Молекулалық масса",
        "group.TPSA": "TPSA",
        "group.HBD": "HBD",
        "group.HBA": "HBA",
        "group.GasteigerMax": "Gasteiger max",
        "group.GasteigerMin": "Gasteiger min",
        "group.LogP": "LogP",
        "group.other": "Басқа белгілер",
        "direction.positive": "модельдің ағымдағы қорытындысын күшейтеді",
        "direction.negative": "модельдің ағымдағы қорытындысын әлсіретеді",
        "direction.neutral": "үлесі аз немесе бейтарап",
        "direction.importance_only": "таңбасыз маңыздылық",
        "disclaimer": (
            "Жергілікті түсіндіру модель қорытындысына қай белгілер көбірек әсер еткенін көрсетеді. "
            "Құрылымдық отпечатоктарда жеке биттерді химиялық мағынамен тікелей байланыстыру қиын, "
            "сондықтан BioSynth-EDU оларды оқу топтарына біріктіреді. Бұл түсіндірме эксперименттік дәлел емес."
        ),
        "fingerprint_note": (
            "Сигналдың үлкен бөлігі молекуланың құрылымдық отпечатоктарына тиесілі. Бұл модель құрылым "
            "фрагменттерін қолданатынын білдіреді, бірақ жеке техникалық биттің қарапайым оқу атауы бола бермейді."
        ),
        "physchem_note": (
            "Физика-химиялық белгілер елеулі үлес қосты. Оларды LogP, TPSA, MW, HBD/HBA және "
            "жартылай зарядтармен байланыстырып түсіндіруге болады."
        ),
        "bbb_note": (
            "Бұл қан-ми тосқауылынан өтуге арналған қосымша статистикалық тексеру. Негізгі оқу талдауын "
            "дескрипторлар және Gupta формуласы арқылы оқыған дұрыс."
        ),
        "pgp_note": (
            "P-gp тосқауыл жасушаларынан молекуланы белсенді шығару қаупі ретінде қаралады. Жоғары сигнал "
            "ОЖЖ үшін қолжетімділікті төмендетуі мүмкін."
        ),
        "unavailable_reason": "ML-талдау қолжетімсіз: модель жүктелмеген, өшірілген немесе SMILES қате.",
    },
    "en": {
        "model.rf_pgp_model": "P-gp RF v2",
        "model.rf_bbb_model": "BBB RF v2",
        "status.ok": "explanation generated",
        "status.unavailable": "model unavailable",
        "status.disabled": "model disabled by runtime selection",
        "method.shap": "SHAP TreeExplainer",
        "method.feature_importance_fallback": "fallback: feature_importances × feature value",
        "group.morgan": "Morgan fingerprint bits",
        "group.maccs": "MACCS keys",
        "group.MW": "Molecular weight",
        "group.TPSA": "TPSA",
        "group.HBD": "HBD",
        "group.HBA": "HBA",
        "group.GasteigerMax": "Gasteiger max",
        "group.GasteigerMin": "Gasteiger min",
        "group.LogP": "LogP",
        "group.other": "Other features",
        "direction.positive": "strengthens the current model conclusion",
        "direction.negative": "weakens the current model conclusion",
        "direction.neutral": "small or neutral contribution",
        "direction.importance_only": "unsigned importance",
        "disclaimer": (
            "The local explanation shows which features influenced the model conclusion most. For structural "
            "fingerprints, individual bits are difficult to map directly to chemical meaning, so BioSynth-EDU "
            "aggregates them into educational groups. This explanation is not experimental evidence."
        ),
        "fingerprint_note": (
            "A large share of the signal comes from structural fingerprints. This means the model uses "
            "structural fragments, but individual technical bits do not always have simple educational names."
        ),
        "physchem_note": (
            "Physicochemical features contributed noticeably. These can be linked to the usual educational "
            "breakdown: LogP, TPSA, MW, HBD/HBA and partial charges."
        ),
        "bbb_note": (
            "This is a supplementary statistical check for BBB passage. The main teaching interpretation is better "
            "read through descriptors and the Gupta formula."
        ),
        "pgp_note": (
            "P-gp is treated as a risk of active removal from barrier cells. A high signal can reduce CNS exposure."
        ),
        "unavailable_reason": "ML breakdown is unavailable: the model is not loaded, disabled, or the SMILES is invalid.",
    },
}


def ml_t(key: str, lang: str = "ru", **kwargs: Any) -> str:
    lang = normalize_language(lang)
    template = ML_TEXT.get(lang, ML_TEXT["ru"]).get(key, ML_TEXT["ru"].get(key, key))
    try:
        return str(template).format(**kwargs)
    except Exception:
        return str(template)


@dataclass(frozen=True)
class FeatureGroup:
    key: str
    label_key: str
    start: int
    stop: int
    kind: str = "group"  # group | scalar
    scalar_label: Optional[str] = None

    @property
    def length(self) -> int:
        return max(0, self.stop - self.start)


def build_feature_vector_for_model(mol: Chem.Mol, legacy_model_name: str) -> np.ndarray:
    if legacy_model_name == "rf_pgp_model":
        return build_pgp_features(mol)
    if legacy_model_name == "rf_bbb_model":
        return build_bbb_rf_features(mol)
    raise ValueError(f"Unsupported explainable model: {legacy_model_name}")


def feature_kind_for_model(legacy_model_name: str, runtime_entry: Mapping[str, Any] | None = None) -> str:
    if runtime_entry and runtime_entry.get("feature_kind"):
        return str(runtime_entry["feature_kind"])
    return MODEL_META.get(legacy_model_name, {}).get("feature_kind", "unknown")


def feature_groups(feature_kind: str) -> List[FeatureGroup]:
    if feature_kind == "hybrid_pgp":
        return [
            FeatureGroup("morgan", "group.morgan", 0, 2048, "group"),
            FeatureGroup("maccs", "group.maccs", 2048, 2215, "group"),
            FeatureGroup("MW", "group.MW", 2215, 2216, "scalar", "MW"),
            FeatureGroup("TPSA", "group.TPSA", 2216, 2217, "scalar", "TPSA"),
            FeatureGroup("HBD", "group.HBD", 2217, 2218, "scalar", "HBD"),
            FeatureGroup("HBA", "group.HBA", 2218, 2219, "scalar", "HBA"),
            FeatureGroup("GasteigerMax", "group.GasteigerMax", 2219, 2220, "scalar", "GasteigerMax"),
            FeatureGroup("GasteigerMin", "group.GasteigerMin", 2220, 2221, "scalar", "GasteigerMin"),
            FeatureGroup("LogP", "group.LogP", 2221, 2222, "scalar", "LogP"),
        ]
    if feature_kind == "morgan_2048":
        return [FeatureGroup("morgan", "group.morgan", 0, 2048, "group")]
    return [FeatureGroup("other", "group.other", 0, 0, "group")]


def _positive_class_index(model: Any, positive_label: int = 1) -> int:
    classes = list(getattr(model, "classes_", []))
    if positive_label in classes:
        return classes.index(positive_label)
    if len(classes) >= 2:
        return 1
    return 0


def _extract_class_shap_values(raw_values: Any, model: Any, n_features: int, positive_label: int = 1) -> np.ndarray:
    class_idx = _positive_class_index(model, positive_label=positive_label)

    if isinstance(raw_values, list):
        if not raw_values:
            raise ValueError("Empty SHAP values list")
        index = min(class_idx, len(raw_values) - 1)
        arr = np.asarray(raw_values[index])
        return np.asarray(arr[0], dtype=float).reshape(-1)

    arr = np.asarray(raw_values)
    if arr.ndim == 1:
        return arr.astype(float).reshape(-1)
    if arr.ndim == 2:
        return arr[0].astype(float).reshape(-1)
    if arr.ndim == 3:
        # Newer SHAP versions for sklearn classifiers often return
        # shape (n_samples, n_features, n_outputs).
        if arr.shape[0] == 1 and arr.shape[1] == n_features:
            index = min(class_idx, arr.shape[2] - 1)
            return arr[0, :, index].astype(float).reshape(-1)
        # Some older paths use (n_outputs, n_samples, n_features).
        if arr.shape[2] == n_features:
            index = min(class_idx, arr.shape[0] - 1)
            return arr[index, 0, :].astype(float).reshape(-1)

    raise ValueError(f"Unsupported SHAP values shape: {arr.shape}")


def compute_local_contributions(
    model: Any,
    x: np.ndarray,
    *,
    positive_label: int = 1,
    use_shap: bool = True,
) -> tuple[np.ndarray, str, Optional[str]]:
    """Return local feature contributions and the method used.

    If SHAP is unavailable or fails, the function falls back to unsigned
    ``feature_importances_ * feature_value``.  That fallback is useful for
    UI/debugging but is explicitly labelled as unsigned importance, not SHAP.
    """
    n_features = int(x.shape[1])

    if use_shap:
        try:
            import shap  # type: ignore

            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(x)
            values = _extract_class_shap_values(raw, model, n_features, positive_label=positive_label)
            if values.shape[0] != n_features:
                raise ValueError(f"SHAP length mismatch: {values.shape[0]} != {n_features}")
            return values.astype(float), "shap", None
        except Exception as exc:  # pragma: no cover - exercised by fallback tests indirectly
            fallback_error = f"{type(exc).__name__}: {exc}"
        
    else:
        fallback_error = None

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return np.zeros(n_features, dtype=float), "feature_importance_fallback", fallback_error or "model has no feature_importances_"

    values = np.asarray(importances, dtype=float).reshape(-1)
    if values.shape[0] != n_features:
        return np.zeros(n_features, dtype=float), "feature_importance_fallback", f"feature_importances_ length mismatch: {values.shape[0]} != {n_features}"

    # Keep only magnitude-like values.  This fallback is intentionally unsigned.
    return values * np.abs(x[0].astype(float)), "feature_importance_fallback", fallback_error


def _direction(value: float, *, signed: bool = True, eps: float = 1e-9) -> str:
    if not signed:
        return "importance_only"
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "neutral"


def _summarize_group_value(vector: np.ndarray, group: FeatureGroup) -> Any:
    segment = vector[group.start:group.stop]
    if group.kind == "scalar":
        return float(segment[0]) if segment.size else None
    if group.key in {"morgan", "maccs"}:
        return int(np.count_nonzero(segment))
    if segment.size:
        return float(np.sum(segment))
    return None


def aggregate_contributions(
    vector: np.ndarray,
    contributions: np.ndarray,
    feature_kind: str,
    *,
    lang: str = "ru",
    signed: bool = True,
) -> List[Dict[str, Any]]:
    lang = normalize_language(lang)
    rows: List[Dict[str, Any]] = []
    for group in feature_groups(feature_kind):
        if group.stop > len(vector):
            continue
        contrib_segment = contributions[group.start:group.stop]
        value_segment = vector[group.start:group.stop]
        total = float(np.sum(contrib_segment)) if contrib_segment.size else 0.0
        abs_total = float(np.sum(np.abs(contrib_segment))) if contrib_segment.size else 0.0
        rows.append(
            {
                "group_key": group.key,
                "group_label": ml_t(group.label_key, lang),
                "feature_kind": feature_kind,
                "kind": group.kind,
                "start": group.start,
                "stop": group.stop,
                "n_features": group.length,
                "active_or_value": _summarize_group_value(vector, group),
                "contribution": total,
                "abs_contribution": abs_total,
                "direction": _direction(total, signed=signed),
                "direction_label": ml_t(f"direction.{_direction(total, signed=signed)}", lang),
            }
        )
    rows.sort(key=lambda item: float(item.get("abs_contribution") or 0.0), reverse=True)
    return rows


def top_feature_contributions(
    vector: np.ndarray,
    contributions: np.ndarray,
    feature_kind: str,
    *,
    lang: str = "ru",
    max_items: int = 12,
    signed: bool = True,
) -> List[Dict[str, Any]]:
    groups = feature_groups(feature_kind)
    feature_to_group: Dict[int, FeatureGroup] = {}
    for group in groups:
        for idx in range(group.start, min(group.stop, len(vector))):
            feature_to_group[idx] = group

    order = np.argsort(-np.abs(contributions))[: max(0, int(max_items))]
    rows: List[Dict[str, Any]] = []
    for idx in order:
        value = float(vector[idx])
        contrib = float(contributions[idx])
        group = feature_to_group.get(int(idx))
        if group is None:
            group_label = ml_t("group.other", lang)
            feature_label = f"feature_{idx}"
        elif group.kind == "scalar":
            group_label = ml_t(group.label_key, lang)
            feature_label = group.scalar_label or group.key
        elif group.key == "morgan":
            group_label = ml_t(group.label_key, lang)
            feature_label = f"Morgan bit {idx - group.start}"
        elif group.key == "maccs":
            group_label = ml_t(group.label_key, lang)
            feature_label = f"MACCS key {idx - group.start}"
        else:
            group_label = ml_t(group.label_key, lang)
            feature_label = f"feature_{idx}"
        rows.append(
            {
                "feature_index": int(idx),
                "feature_label": feature_label,
                "group_label": group_label,
                "value": value,
                "contribution": contrib,
                "abs_contribution": abs(contrib),
                "direction": _direction(contrib, signed=signed),
                "direction_label": ml_t(f"direction.{_direction(contrib, signed=signed)}", lang),
            }
        )
    return rows


def _predict_classifier(model: Any, x: np.ndarray, legacy_model_name: str, threshold: Optional[float]) -> Dict[str, Any]:
    prob = get_positive_class_probability(model, x, positive_label=1)
    prob = float(max(0.0, min(1.0, prob)))
    if threshold is None:
        pred_class = int(model.predict(x)[0])
    else:
        pred_class = int(prob >= float(threshold))
    return {"probability": round(prob, 4), "class": pred_class, "threshold": threshold}


def _commentary(legacy_model_name: str, groups: Sequence[Mapping[str, Any]], *, lang: str = "ru") -> List[str]:
    notes: List[str] = []
    top = list(groups[:3])
    top_keys = {str(item.get("group_key")) for item in top}
    if "morgan" in top_keys or "maccs" in top_keys:
        notes.append(ml_t("fingerprint_note", lang))
    if any(key in top_keys for key in {"MW", "TPSA", "HBD", "HBA", "GasteigerMax", "GasteigerMin", "LogP"}):
        notes.append(ml_t("physchem_note", lang))
    if legacy_model_name == "rf_bbb_model":
        notes.append(ml_t("bbb_note", lang))
    if legacy_model_name == "rf_pgp_model":
        notes.append(ml_t("pgp_note", lang))
    return notes


def unavailable_explanation(legacy_model_name: str, *, lang: str = "ru", reason: str | None = None) -> Dict[str, Any]:
    lang = normalize_language(lang)
    return {
        "schema_version": "ml_explanation_v1.0",
        "language": lang,
        "legacy_model_name": legacy_model_name,
        "model_label": ml_t(f"model.{legacy_model_name}", lang),
        "status": "unavailable",
        "status_label": ml_t("status.unavailable", lang),
        "reason": reason or ml_t("unavailable_reason", lang),
        "prediction": {},
        "method": None,
        "method_label": None,
        "group_contributions": [],
        "top_features": [],
        "commentary": [ml_t("unavailable_reason", lang)],
    }


def explain_tree_model_for_mol(
    mol: Chem.Mol,
    legacy_model_name: str,
    *,
    model: Any | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    lang: str = "ru",
    use_shap: bool = True,
    max_features: int = 12,
) -> Dict[str, Any]:
    """Explain one selected runtime tree model for one RDKit molecule."""
    lang = normalize_language(lang)
    if legacy_model_name not in SUPPORTED_ML_EXPLAINABLE_MODELS:
        return unavailable_explanation(legacy_model_name, lang=lang, reason=f"Unsupported model: {legacy_model_name}")

    if model is None:
        if runtime_context is None:
            runtime_context = get_runtime_context()
        if is_runtime_disabled(legacy_model_name, context=runtime_context):
            return unavailable_explanation(legacy_model_name, lang=lang, reason=ml_t("status.disabled", lang))
        model = get_runtime_model(legacy_model_name, runtime_context)
        if model is None:
            return unavailable_explanation(legacy_model_name, lang=lang)
        runtime_entry = get_runtime_entry(legacy_model_name, runtime_context)
        threshold = get_runtime_threshold(legacy_model_name, runtime_context)
    else:
        runtime_entry = {}
        threshold = None

    try:
        vector = build_feature_vector_for_model(mol, legacy_model_name)
        x = validate_feature_vector(vector, expected_dim=getattr(model, "n_features_in_", None), name=legacy_model_name).reshape(1, -1)
        feature_kind = feature_kind_for_model(legacy_model_name, runtime_entry)
        prediction = _predict_classifier(model, x, legacy_model_name, threshold)
        positive_label = int(MODEL_META[legacy_model_name].get("positive_label", 1))
        values, method, fallback_error = compute_local_contributions(model, x, positive_label=positive_label, use_shap=use_shap)
        signed = method == "shap"
        groups = aggregate_contributions(x[0], values, feature_kind, lang=lang, signed=signed)
        top = top_feature_contributions(x[0], values, feature_kind, lang=lang, max_items=max_features, signed=signed)
        return {
            "schema_version": "ml_explanation_v1.0",
            "language": lang,
            "legacy_model_name": legacy_model_name,
            "model_label": ml_t(f"model.{legacy_model_name}", lang),
            "status": "ok",
            "status_label": ml_t("status.ok", lang),
            "runtime_entry": dict(runtime_entry),
            "feature_kind": feature_kind,
            "prediction": prediction,
            "method": method,
            "method_label": ml_t(f"method.{method}", lang),
            "fallback_error": fallback_error,
            "group_contributions": groups,
            "top_features": top,
            "commentary": _commentary(legacy_model_name, groups, lang=lang),
            "disclaimer": ml_t("disclaimer", lang),
        }
    except Exception as exc:
        return unavailable_explanation(legacy_model_name, lang=lang, reason=f"{type(exc).__name__}: {exc}")


def explain_selected_runtime_models_for_smiles(
    smiles: str,
    *,
    lang: str = "ru",
    models: Sequence[str] = SUPPORTED_ML_EXPLAINABLE_MODELS,
    runtime_context: Mapping[str, Any] | None = None,
    use_shap: bool = True,
    max_features: int = 12,
) -> Dict[str, Any]:
    """Build ML explanations for selected runtime models for a SMILES string."""
    lang = normalize_language(lang)
    mol = Chem.MolFromSmiles(str(smiles).strip()) if smiles else None
    if mol is None:
        return {
            "schema_version": "ml_explanations_v1.0",
            "language": lang,
            "input_smiles": smiles,
            "valid": False,
            "disclaimer": ml_t("disclaimer", lang),
            "models": {name: unavailable_explanation(name, lang=lang) for name in models},
        }

    if runtime_context is None:
        runtime_context = get_runtime_context()

    return {
        "schema_version": "ml_explanations_v1.0",
        "language": lang,
        "input_smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "valid": True,
        "disclaimer": ml_t("disclaimer", lang),
        "models": {
            name: explain_tree_model_for_mol(
                mol,
                name,
                runtime_context=runtime_context,
                lang=lang,
                use_shap=use_shap,
                max_features=max_features,
            )
            for name in models
        },
    }


def build_ml_explanation_summary(ml_explanations: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact summary useful for reports or tests."""
    rows: List[Dict[str, Any]] = []
    for name, item in (ml_explanations.get("models") or {}).items():
        if not isinstance(item, Mapping):
            continue
        prediction = item.get("prediction") if isinstance(item.get("prediction"), Mapping) else {}
        top_groups = item.get("group_contributions") if isinstance(item.get("group_contributions"), list) else []
        rows.append(
            {
                "model": name,
                "status": item.get("status"),
                "method": item.get("method"),
                "probability": prediction.get("probability"),
                "class": prediction.get("class"),
                "threshold": prediction.get("threshold"),
                "top_group": top_groups[0].get("group_label") if top_groups else None,
                "top_group_contribution": top_groups[0].get("contribution") if top_groups else None,
            }
        )
    return {"schema_version": "ml_explanation_summary_v1.0", "rows": rows}
