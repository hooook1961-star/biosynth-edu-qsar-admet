"""Educational What-if simulation for BioSynth-EDU.

The What-if lab changes descriptor values, not the molecular structure. It is a
teaching sensitivity analysis over MW, LogP, TPSA, HBD, HBA, pKa and P-gp.
It does not call the real ML models and must not be presented as a prediction
for a new molecule.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from core.explainability import canonical_descriptor_name, classify_descriptor_zone
from core.teaching_templates import WHAT_IF_DISCLAIMER_RU
from core.i18n import normalize_language, disclaimer, what_if_text


WHAT_IF_WEIGHTS: dict[str, float] = {
    "LogP": 0.22,
    "TPSA": 0.22,
    "MW": 0.16,
    "HBD": 0.12,
    "HBA": 0.10,
    "pKa_pred": 0.12,
    "FormalCharge": 0.06,
}

DEFAULT_WHAT_IF_DESCRIPTORS: tuple[str, ...] = (
    "LogP",
    "TPSA",
    "MW",
    "pKa_pred",
    "HBD",
    "HBA",
    "Pgp_probability",
)

WHAT_IF_SLIDER_CONFIG: dict[str, dict[str, Any]] = {
    "LogP": {"label": "LogP", "min": -2.0, "max": 7.0, "step": 0.1},
    "TPSA": {"label": "TPSA", "min": 0.0, "max": 220.0, "step": 1.0},
    "MW": {"label": "MW", "min": 50.0, "max": 800.0, "step": 1.0},
    "pKa_pred": {"label": "pKa", "min": 0.0, "max": 14.0, "step": 0.1},
    "HBD": {"label": "HBD", "min": 0, "max": 8, "step": 1},
    "HBA": {"label": "HBA", "min": 0, "max": 15, "step": 1},
    "Pgp_probability": {"label": "P-gp", "min": 0.0, "max": 1.0, "step": 0.01},
}

DISPLAY_NAMES: dict[str, dict[str, str]] = {
    "ru": {
        "MW": "Молекулярная масса",
        "LogP": "LogP",
        "TPSA": "TPSA",
        "HBD": "Доноры H-связей",
        "HBA": "Акцепторы H-связей",
        "RotatableBonds": "Вращаемые связи",
        "AromaticRings": "Ароматические кольца",
        "pKa_pred": "pKa",
        "FormalCharge": "Формальный заряд",
        "Pgp_probability": "Риск P-gp",
    },
    "kk": {
        "MW": "Молекулалық масса",
        "LogP": "LogP",
        "TPSA": "TPSA",
        "HBD": "H-байланыс донорлары",
        "HBA": "H-байланыс акцепторлары",
        "RotatableBonds": "Айналмалы байланыстар",
        "AromaticRings": "Ароматты сақиналар",
        "pKa_pred": "pKa",
        "FormalCharge": "Формальды заряд",
        "Pgp_probability": "P-gp қаупі",
    },
    "en": {
        "MW": "Molecular weight",
        "LogP": "LogP",
        "TPSA": "TPSA",
        "HBD": "H-bond donors",
        "HBA": "H-bond acceptors",
        "RotatableBonds": "Rotatable bonds",
        "AromaticRings": "Aromatic rings",
        "pKa_pred": "pKa",
        "FormalCharge": "Formal charge",
        "Pgp_probability": "P-gp risk",
    },
}

SCORE_CLASS_LABELS_RU = {
    "high": "благоприятная",
    "borderline": "пограничная",
    "low": "неблагоприятная",
}

DESCRIPTOR_DIRECTION_TEXT_RU = {
    "LogP": (
        "Умеренный LogP обычно помогает пассивному прохождению через ГЭБ. Слишком низкий LogP делает молекулу "
        "слишком гидрофильной, а слишком высокий может ухудшать растворимость."
    ),
    "TPSA": (
        "Рост TPSA обычно ухудшает прохождение через ГЭБ, потому что молекула становится более полярной."
    ),
    "MW": "Рост молекулярной массы обычно усложняет пассивное прохождение через ГЭБ.",
    "HBD": "Рост HBD усиливает взаимодействие с водой и часто мешает пассивной диффузии.",
    "HBA": "Рост HBA часто повышает полярность и ухудшает оценку прохождения через ГЭБ.",
    "pKa_pred": (
        "pKa влияет на долю ионизированной формы при pH 7.4. Более ионизированные формы "
        "обычно хуже проходят через липидные мембраны."
    ),
    "Pgp_probability": (
        "P-gp не меняет пассивное прохождение через ГЭБ, но может снижать доступность для ЦНС за счёт активного выведения."
    ),
}

DESCRIPTOR_DIRECTION_TEXTS = {
    "ru": DESCRIPTOR_DIRECTION_TEXT_RU,
    "kk": {
        "LogP": "Орташа LogP қан-ми тосқауылынан пассивті өтуге көмектеседі; тым төмен немесе тым жоғары LogP профильді нашарлатуы мүмкін.",
        "TPSA": "TPSA өсуі молекуланың полярлығын арттырып, қан-ми тосқауылынан өтуді қиындатады.",
        "MW": "Молекулалық массаның өсуі қан-ми тосқауылынан өтуді қиындатуы мүмкін.",
        "HBD": "HBD санының өсуі сумен әрекеттесуді күшейтіп, пассивті диффузияға кедергі болуы мүмкін.",
        "HBA": "HBA санының өсуі полярлықты арттырып, қан-ми тосқауылынан өту бағасын төмендетуі мүмкін.",
        "pKa_pred": "pKa pH 7.4 кезіндегі иондану үлесіне әсер етеді; иондалған түрлер мембранадан нашар өтеді.",
        "Pgp_probability": "P-gp қан-ми тосқауылынан пассивті өтуді өзгертпейді, бірақ белсенді шығару арқылы ОЖЖ қолжетімділігін төмендетуі мүмкін.",
    },
    "en": {
        "LogP": "Moderate LogP usually supports passive diffusion; very low or very high LogP can worsen the profile.",
        "TPSA": "Increasing TPSA makes the molecule more polar and usually reduces passive BBB passage.",
        "MW": "Increasing molecular weight can make passive BBB passage more difficult.",
        "HBD": "Increasing HBD strengthens interactions with water and can oppose passive diffusion.",
        "HBA": "Increasing HBA often raises polarity and can make BBB passage less favorable.",
        "pKa_pred": "pKa affects the ionised fraction at pH 7.4; more ionised forms usually cross membranes less efficiently.",
        "Pgp_probability": "P-gp does not change passive BBB passage, but it can lower CNS exposure by actively removing the molecule.",
    },
}

SCORE_CLASS_LABELS = {
    "ru": SCORE_CLASS_LABELS_RU,
    "kk": {"high": "қолайлы", "borderline": "шекаралық", "low": "қолайсыз"},
    "en": {"high": "high", "borderline": "borderline", "low": "low"},
}

ZONE_LABELS = {
    "ru": {"green": "зелёная", "yellow": "жёлтая", "red": "красная", "gray": "серая"},
    "kk": {"green": "жасыл", "yellow": "сары", "red": "қызыл", "gray": "сұр"},
    "en": {"green": "green", "yellow": "yellow", "red": "red", "gray": "gray"},
}


WhatIfDict = dict[str, Any]


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def calculate_descriptor_desirability(
    name: str,
    value: Any,
    context: Mapping[str, Any] | None = None,
) -> float:
    """Return an educational descriptor desirability score from 0 to 1."""
    canonical = canonical_descriptor_name(name)
    value_f = _as_float(value)
    if value_f is None:
        return 0.5

    if canonical == "LogP":
        return _plateau_score(value_f, low=0.5, opt_low=1.5, opt_high=3.5, high=5.0, floor=0.05)

    if canonical == "TPSA":
        if value_f <= 70:
            return 1.0
        if value_f <= 90:
            return _linear(value_f, 70, 90, 0.95, 0.65)
        if value_f <= 120:
            return _linear(value_f, 90, 120, 0.65, 0.25)
        if value_f <= 180:
            return _linear(value_f, 120, 180, 0.25, 0.05)
        return 0.03

    if canonical == "MW":
        if 150 <= value_f <= 450:
            return 1.0
        if 100 <= value_f < 150:
            return _linear(value_f, 100, 150, 0.55, 1.0)
        if value_f < 100:
            return 0.45
        if value_f <= 500:
            return _linear(value_f, 450, 500, 1.0, 0.6)
        if value_f <= 700:
            return _linear(value_f, 500, 700, 0.6, 0.08)
        return 0.05

    if canonical == "HBD":
        if value_f <= 1:
            return 1.0
        if value_f <= 2:
            return 0.65
        if value_f <= 3:
            return 0.35
        return 0.12

    if canonical == "HBA":
        if value_f <= 5:
            return 1.0
        if value_f <= 7:
            return 0.65
        if value_f <= 10:
            return _linear(value_f, 8, 10, 0.35, 0.15)
        return 0.08

    if canonical == "RotatableBonds":
        if value_f <= 5:
            return 1.0
        if value_f <= 8:
            return 0.65
        return 0.25

    if canonical == "pKa_pred":
        return _pka_desirability(value_f, context or {})

    if canonical == "FormalCharge":
        abs_charge = abs(value_f)
        if abs_charge < 0.5:
            return 1.0
        if abs_charge <= 1.0:
            return 0.45
        return 0.12

    if canonical == "GasteigerAbsMax":
        if value_f < 0.45:
            return 0.9
        if value_f <= 0.75:
            return 0.55
        return 0.25

    if canonical == "Pgp_probability":
        return clamp(1.0 - value_f)

    return 0.5


def calculate_simplified_bbb_score(
    descriptors: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> float:
    """Calculate an educational passive BBB score from descriptor desirability."""
    canonical = _canonicalize_descriptors(descriptors)
    total_weight = 0.0
    weighted_sum = 0.0
    full_context = {**canonical, **(context or {})}

    for name, weight in WHAT_IF_WEIGHTS.items():
        if name not in canonical:
            continue
        desirability = calculate_descriptor_desirability(name, canonical[name], context=full_context)
        weighted_sum += weight * desirability
        total_weight += weight

    if total_weight <= 0:
        return 0.5
    return round(clamp(weighted_sum / total_weight), 3)


def calculate_simplified_cns_score(passive_bbb_score: Any, pgp_probability: Any) -> float:
    """Apply an educational P-gp efflux penalty to passive BBB score."""
    bbb = _as_float(passive_bbb_score, default=0.5)
    pgp = _as_float(pgp_probability, default=0.5)
    if bbb is None:
        bbb = 0.5
    if pgp is None:
        pgp = 0.5
    pgp = clamp(pgp)
    penalty = 1.0 - 0.55 * pgp
    return round(clamp(bbb * penalty), 3)


def classify_educational_score(score: Any) -> str:
    value = _as_float(score, default=0.5)
    if value is None:
        value = 0.5
    if value >= 0.70:
        return "high"
    if value >= 0.40:
        return "borderline"
    return "low"


def classify_what_if_zone(name: str, value: Any, *, context: Mapping[str, Any] | None = None) -> str:
    """Return green/yellow/red/gray for a descriptor in What-if mode."""
    return str(classify_descriptor_zone(name, value, context=context or {}).get("zone", "gray"))


def simulate_descriptor_change(
    base_descriptors: Mapping[str, Any],
    changed_values: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    lang: str = "ru",
) -> WhatIfDict:
    """Simulate changed descriptor values and return educational score deltas."""
    lang = normalize_language(lang)
    base = _canonicalize_descriptors(base_descriptors)
    changed = _canonicalize_descriptors(changed_values or {})
    modified = dict(base)
    modified.update(changed)

    base_passive = calculate_simplified_bbb_score(base, context=context)
    modified_passive = calculate_simplified_bbb_score(modified, context=context)

    base_pgp = _as_float(base.get("Pgp_probability"), default=0.5)
    modified_pgp = _as_float(modified.get("Pgp_probability"), default=base_pgp)
    if base_pgp is None:
        base_pgp = 0.5
    if modified_pgp is None:
        modified_pgp = base_pgp

    base_cns = calculate_simplified_cns_score(base_passive, base_pgp)
    modified_cns = calculate_simplified_cns_score(modified_passive, modified_pgp)

    zone_changes = compare_descriptor_zones(base, modified, changed.keys(), context=context, lang=lang)
    score_components = _build_score_components(base, modified, context=context, lang=lang)

    base_scores = {
        "passive_bbb_score": base_passive,
        "cns_score": base_cns,
        "passive_class": classify_educational_score(base_passive),
        "cns_class": classify_educational_score(base_cns),
        "pgp_probability": round(float(base_pgp), 3),
    }
    modified_scores = {
        "passive_bbb_score": modified_passive,
        "cns_score": modified_cns,
        "passive_class": classify_educational_score(modified_passive),
        "cns_class": classify_educational_score(modified_cns),
        "pgp_probability": round(float(modified_pgp), 3),
    }
    score_delta = {
        "passive_bbb_score": round(modified_passive - base_passive, 3),
        "cns_score": round(modified_cns - base_cns, 3),
        "pgp_probability": round(float(modified_pgp) - float(base_pgp), 3),
    }

    result: WhatIfDict = {
        "available": True,
        "mode": "educational_heuristic_v1",
        "is_real_model_prediction": False,
        "language": lang,
        "disclaimer": disclaimer("what_if", lang),
        "base_descriptors": base,
        "changed_values": changed,
        "modified_descriptors": modified,
        "base": {"descriptors": base, **base_scores},
        "modified": {"descriptors": modified, **modified_scores},
        "delta": score_delta,
        "score_delta": score_delta,
        "base_scores": base_scores,
        "modified_scores": modified_scores,
        "descriptor_changes": zone_changes,
        "zone_changes": zone_changes,
        "score_components": score_components,
        "descriptor_scores": {
            "base": {item["name"]: item for item in score_components},
            "modified": {item["name"]: item for item in score_components},
        },
    }
    result["commentary"] = generate_what_if_commentary(result, lang=lang)
    return result


def compare_descriptor_zones(
    base_descriptors: Mapping[str, Any],
    modified_descriptors: Mapping[str, Any],
    changed_keys: Iterable[str] | None = None,
    context: Mapping[str, Any] | None = None,
    lang: str = "ru",
) -> list[dict[str, Any]]:
    """Compare descriptor zones between two descriptor sets."""
    lang = normalize_language(lang)
    base = _canonicalize_descriptors(base_descriptors)
    modified = _canonicalize_descriptors(modified_descriptors)
    if changed_keys is None:
        keys = _ordered_keys(set(base) | set(modified))
    else:
        keys = _ordered_keys({canonical_descriptor_name(str(key)) for key in changed_keys})

    rows: list[dict[str, Any]] = []
    for key in keys:
        old_value = base.get(key)
        new_value = modified.get(key)
        if old_value == new_value:
            continue
        old_zone = classify_descriptor_zone(key, old_value, context={**base, **(context or {})}).get("zone", "gray")
        new_zone = classify_descriptor_zone(key, new_value, context={**modified, **(context or {})}).get("zone", "gray")
        old_des = calculate_descriptor_desirability(key, old_value, context={**base, **(context or {})})
        new_des = calculate_descriptor_desirability(key, new_value, context={**modified, **(context or {})})
        rows.append(
            {
                "name": key,
                "display_name": _display_name(key, lang),
                "old_value": old_value,
                "new_value": new_value,
                "base_value": old_value,
                "modified_value": new_value,
                "before_value": old_value,
                "after_value": new_value,
                "value_delta": _numeric_delta(new_value, old_value),
                "old_zone": old_zone,
                "new_zone": new_zone,
                "from_zone": old_zone,
                "to_zone": new_zone,
                "before_zone": old_zone,
                "after_zone": new_zone,
                "from_zone_label": _zone_label(old_zone, lang),
                "to_zone_label": _zone_label(new_zone, lang),
                "desirability_delta": round(new_des - old_des, 3),
                "direction": _direction_from_delta(new_des - old_des),
                "teaching_note": _direction_text(key, lang),
            }
        )
    return rows


def generate_what_if_commentary(simulation_result: Mapping[str, Any], lang: str = "ru") -> dict[str, Any]:
    """Generate localized commentary for a What-if simulation result."""
    lang = normalize_language(lang)
    delta = simulation_result.get("delta", {}) if isinstance(simulation_result, Mapping) else {}
    base = simulation_result.get("base", {}) if isinstance(simulation_result, Mapping) else {}
    modified = simulation_result.get("modified", {}) if isinstance(simulation_result, Mapping) else {}

    passive_delta = _as_float(delta.get("passive_bbb_score"), default=0.0) or 0.0
    cns_delta = _as_float(delta.get("cns_score"), default=0.0) or 0.0
    pgp_delta = _as_float(delta.get("pgp_probability"), default=0.0) or 0.0

    messages: list[str] = []
    if passive_delta >= 0.05:
        summary = what_if_text("passive_up", lang)
        messages.append(what_if_text("passive_up", lang))
    elif passive_delta <= -0.05:
        summary = what_if_text("passive_down", lang)
        messages.append(what_if_text("passive_down", lang))
    else:
        summary = what_if_text("passive_same", lang)

    if pgp_delta >= 0.05:
        messages.append(what_if_text("pgp_up", lang))
    elif pgp_delta <= -0.05:
        messages.append(what_if_text("pgp_down", lang))

    if cns_delta >= 0.05:
        messages.append(what_if_text("cns_up", lang))
    elif cns_delta <= -0.05:
        messages.append(what_if_text("cns_down", lang))

    if cns_delta <= -0.05 and passive_delta > -0.05:
        messages.append(what_if_text("pgp_up", lang))

    if not messages:
        messages.append(what_if_text("same", lang))

    components = simulation_result.get("score_components", []) if isinstance(simulation_result, Mapping) else []
    improvements = []
    worsenings = []
    for item in components:
        weighted_delta = _as_float(item.get("weighted_delta"), default=0.0) or 0.0
        if weighted_delta >= 0.01:
            improvements.append(f"{item.get('display_name')}: {item.get('interpretation', '')}")
        elif weighted_delta <= -0.01:
            worsenings.append(f"{item.get('display_name')}: {item.get('interpretation', '')}")

    zone_texts = []
    for item in simulation_result.get("zone_changes", []) if isinstance(simulation_result, Mapping) else []:
        zone_texts.append(
            f"{item.get('display_name')}: {item.get('from_zone')} -> {item.get('to_zone')}"
        )

    pgp_note = ""
    if abs(pgp_delta) >= 0.05:
        pgp_note = _direction_text("Pgp_probability", lang)

    return {
        "summary": summary,
        "messages": messages,
        "details": messages,
        "improvements": improvements,
        "worsenings": worsenings,
        "zone_changes": zone_texts,
        "pgp_note": pgp_note,
        "teaching_note": disclaimer("what_if", lang),
        "base_passive_class": _score_class_label(str(base.get("passive_class")), lang),
        "modified_passive_class": _score_class_label(str(modified.get("passive_class")), lang),
        "base_cns_class": _score_class_label(str(base.get("cns_class")), lang),
        "modified_cns_class": _score_class_label(str(modified.get("cns_class")), lang),
    }


def build_base_descriptors_from_explanation(explanation_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the descriptors used by the What-if lab from explanation_dict."""
    return extract_what_if_base_descriptors(explanation_dict)


def extract_what_if_base_descriptors(explanation_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the descriptor set for What-if mode from explanation_dict."""
    if not isinstance(explanation_dict, Mapping):
        return {}

    what_if_base = explanation_dict.get("what_if_base", {})
    base: dict[str, Any] = {}
    if isinstance(what_if_base, Mapping):
        base.update(_canonicalize_descriptors(what_if_base.get("base_descriptors") or {}))

    descriptors = explanation_dict.get("descriptors", {})
    if isinstance(descriptors, Mapping):
        for key, item in descriptors.items():
            canonical = canonical_descriptor_name(str(key))
            if canonical not in DEFAULT_WHAT_IF_DESCRIPTORS and canonical != "FormalCharge":
                continue
            if canonical in base:
                continue
            if isinstance(item, Mapping) and "value" in item:
                base[canonical] = item.get("value")
            else:
                base[canonical] = item

    model_outputs = explanation_dict.get("model_outputs", {})
    if isinstance(model_outputs, Mapping):
        if "Pgp_probability" not in base and model_outputs.get("pgp_probability") is not None:
            base["Pgp_probability"] = model_outputs.get("pgp_probability")
        if "pKa_pred" not in base and model_outputs.get("pka_pred") is not None:
            base["pKa_pred"] = model_outputs.get("pka_pred")

    return {key: value for key, value in base.items() if key in DEFAULT_WHAT_IF_DESCRIPTORS or key == "FormalCharge"}


def build_what_if_payload(
    explanation_dict: Mapping[str, Any],
    changed_values: Mapping[str, Any] | None = None,
    lang: str | None = None,
) -> WhatIfDict:
    """Build a What-if payload directly from explanation_dict."""
    selected_lang = normalize_language(lang or (explanation_dict.get("language") if isinstance(explanation_dict, Mapping) else None))
    base = extract_what_if_base_descriptors(explanation_dict)
    payload = simulate_descriptor_change(base, changed_values or {}, lang=selected_lang)
    payload["is_real_model_prediction"] = False
    return payload


def _build_score_components(
    base_descriptors: Mapping[str, Any],
    modified_descriptors: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
    lang: str = "ru",
) -> list[dict[str, Any]]:
    lang = normalize_language(lang)
    base = _canonicalize_descriptors(base_descriptors)
    modified = _canonicalize_descriptors(modified_descriptors)
    keys = _ordered_keys(set(base) | set(modified) | set(WHAT_IF_WEIGHTS) | {"Pgp_probability"})
    rows: list[dict[str, Any]] = []

    for key in keys:
        if key not in base and key not in modified:
            continue
        old_value = base.get(key)
        new_value = modified.get(key)

        if key == "Pgp_probability":
            old_pgp = _as_float(old_value, default=0.5) or 0.5
            new_pgp = _as_float(new_value, default=old_pgp) or old_pgp
            desirability_delta = (1.0 - new_pgp) - (1.0 - old_pgp)
            weighted_delta = -0.55 * (new_pgp - old_pgp)
            role = "active_efflux_penalty"
        else:
            if key not in WHAT_IF_WEIGHTS:
                continue
            old_des = calculate_descriptor_desirability(key, old_value, context={**base, **(context or {})})
            new_des = calculate_descriptor_desirability(key, new_value, context={**modified, **(context or {})})
            desirability_delta = new_des - old_des
            weighted_delta = WHAT_IF_WEIGHTS[key] * desirability_delta
            role = "passive_bbb_component"

        rows.append(
            {
                "name": key,
                "display_name": _display_name(key, lang),
                "base_value": old_value,
                "modified_value": new_value,
                "old_value": old_value,
                "new_value": new_value,
                "value_delta": _numeric_delta(new_value, old_value),
                "base_desirability": round(calculate_descriptor_desirability(key, old_value, context={**base, **(context or {})}), 3),
                "modified_desirability": round(calculate_descriptor_desirability(key, new_value, context={**modified, **(context or {})}), 3),
                "desirability_delta": round(desirability_delta, 3),
                "weight": WHAT_IF_WEIGHTS.get(key, 0.0),
                "weighted_delta": round(weighted_delta, 3),
                "role": role,
                "interpretation": _component_interpretation(key, old_value, new_value, weighted_delta, lang=lang),
            }
        )
    return rows


def _canonicalize_descriptors(descriptors: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(descriptors, Mapping):
        return out
    for key, value in descriptors.items():
        canonical = canonical_descriptor_name(str(key))
        out[canonical] = _normalize_number(value)
    return out


def _pka_desirability(pka: float, context: Mapping[str, Any]) -> float:
    pka_type = str(context.get("pKa_type") or context.get("pka_type") or "").lower()
    if pka_type == "base":
        neutral_fraction = 1.0 / (1.0 + 10.0 ** (pka - 7.4))
        return clamp(neutral_fraction)
    if pka_type == "acid":
        neutral_fraction = 1.0 / (1.0 + 10.0 ** (7.4 - pka))
        return clamp(neutral_fraction)

    # Without acid/base type, keep the interpretation conservative.
    if pka < 4.0:
        return 0.60
    if pka <= 6.5:
        return 0.85
    if pka <= 8.5:
        return 0.62
    if pka <= 10.5:
        return _linear(pka, 8.5, 10.5, 0.62, 0.32)
    return 0.20


def _plateau_score(value: float, low: float, opt_low: float, opt_high: float, high: float, floor: float) -> float:
    if opt_low <= value <= opt_high:
        return 1.0
    if low <= value < opt_low:
        return _linear(value, low, opt_low, 0.55, 1.0)
    if opt_high < value <= high:
        return _linear(value, opt_high, high, 1.0, 0.45)
    if value < low:
        return max(floor, _linear(value, low - 2.0, low, floor, 0.55))
    return max(floor, _linear(value, high, high + 3.0, 0.45, floor))


def _linear(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return clamp(y1)
    t = (x - x0) / (x1 - x0)
    return clamp(y0 + clamp(t) * (y1 - y0))


def _direction_from_delta(delta: float) -> str:
    if delta >= 0.05:
        return "improved"
    if delta <= -0.05:
        return "worsened"
    return "similar"


def _component_interpretation(key: str, old_value: Any, new_value: Any, weighted_delta: float, lang: str = "ru") -> str:
    lang = normalize_language(lang)
    direction_map = {
        "ru": {"up": "улучшил", "down": "ухудшил", "same": "почти не изменил"},
        "kk": {"up": "жақсартты", "down": "нашарлатты", "same": "айтарлықтай өзгертпеді"},
        "en": {"up": "improved", "down": "worsened", "same": "changed only slightly"},
    }
    key_dir = "up" if weighted_delta > 0 else "down" if weighted_delta < 0 else "same"
    direction = direction_map[lang][key_dir]
    base_text = _direction_text(key, lang)
    if lang == "en":
        return f"Changing {_fmt(old_value)} -> {_fmt(new_value)} {direction} the teaching estimate. {base_text}"
    if lang == "kk":
        return f"{_fmt(old_value)} -> {_fmt(new_value)} ауысуы оқу бағасын {direction}. {base_text}"
    return f"Переход {_fmt(old_value)} -> {_fmt(new_value)} {direction} учебную оценку. {base_text}"


def _ordered_keys(keys: set[str]) -> list[str]:
    order = ["MW", "LogP", "TPSA", "HBD", "HBA", "pKa_pred", "FormalCharge", "Pgp_probability", "RotatableBonds"]
    return [key for key in order if key in keys] + sorted(key for key in keys if key not in order)


def _display_name(key: str, lang: str = "ru") -> str:
    lang = normalize_language(lang)
    labels = DISPLAY_NAMES.get(lang, DISPLAY_NAMES["ru"])
    return labels.get(key, DISPLAY_NAMES["ru"].get(key, key))


def _zone_label(zone: str, lang: str = "ru") -> str:
    lang = normalize_language(lang)
    return ZONE_LABELS.get(lang, ZONE_LABELS["ru"]).get(str(zone), str(zone))


def _direction_text(key: str, lang: str = "ru") -> str:
    lang = normalize_language(lang)
    return DESCRIPTOR_DIRECTION_TEXTS.get(lang, DESCRIPTOR_DIRECTION_TEXTS["ru"]).get(key, DESCRIPTOR_DIRECTION_TEXT_RU.get(key, "Этот фактор имеет контекстное влияние."))


def _score_class_label(score_class: str, lang: str = "ru") -> str:
    lang = normalize_language(lang)
    return SCORE_CLASS_LABELS.get(lang, SCORE_CLASS_LABELS_RU).get(score_class, score_class)


def _numeric_delta(new_value: Any, old_value: Any) -> float | None:
    new_f = _as_float(new_value)
    old_f = _as_float(old_value)
    if new_f is None or old_f is None:
        return None
    return round(new_f - old_f, 3)


def _normalize_number(value: Any) -> Any:
    numeric = _as_float(value)
    if numeric is None:
        return value
    if abs(numeric - round(numeric)) < 1e-9 and abs(numeric) < 1000:
        return int(round(numeric))
    return round(numeric, 3)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric:  # NaN
        return default
    return numeric


def _fmt(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return str(value)
    if abs(numeric) < 100:
        return f"{numeric:.3g}"
    return f"{numeric:.1f}"

# Backward-compatible name kept for existing imports.
def build_base_descriptors_from_explanation(explanation_dict: Mapping[str, Any]) -> dict[str, float]:
    return extract_what_if_base_descriptors(explanation_dict)
