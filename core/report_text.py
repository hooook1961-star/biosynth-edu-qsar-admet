"""Single source of student-report text.

The report renderer should import labels and educational wording from here
instead of hard-coding copy in ``core.reporting`` or Streamlit components.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


REPORT_TEXT: dict[str, dict[str, Any]] = {
    "ru": {
        "title": "BioSynth-EDU: учебный отчет ADMET, ГЭБ и ЦНС",
        "generated": "Дата генерации UTC",
        "molecule": "Молекула",
        "input_smiles": "Исходный SMILES",
        "canonical_smiles": "Канонический SMILES",
        "validity": "Корректность структуры",
        "valid": "структура распознана",
        "invalid": "структура не распознана",
        "warnings": "Предупреждения",
        "none": "нет",
        "summary": "Итоговый учебный вывод",
        "uncertainty": "Уровень неопределенности",
        "uncertainty_levels": {
            "low": "низкий",
            "medium": "средний",
            "high": "высокий",
            "unknown": "не определен",
        },
        "scores": "Основные показатели",
        "indicator": "Показатель",
        "value": "Значение",
        "descriptors": "Дескрипторы и объяснения",
        "descriptor": "Дескриптор",
        "zone": "Зона",
        "effect": "Влияние",
        "short_expl": "Короткое объяснение",
        "factors": "Факторы за и против",
        "positive": "Поддерживают прохождение через ГЭБ и доступность для ЦНС",
        "negative": "Мешают прохождению через ГЭБ или доступности для ЦНС",
        "borderline": "Пограничные факторы",
        "no_factors": "Нет явных факторов в этой группе.",
        "matrix": "Матрица ГЭБ (BBB) × P-gp",
        "current": "Текущий сценарий",
        "interpretation": "Интерпретация",
        "steps": "Пошаговое решение модели",
        "step": "Шаг",
        "status": "Статус",
        "methodology": "Методология",
        "limitations": "Ограничения",
        "questions": "Вопросы для студента",
        "disclaimers": "Оговорки",
        "markdown": "Markdown-версия отчета",
        "in_silico_label": "Расчетный прогноз",
        "what_if_label": "What-if лаборатория",
        "scores_table": [
            ("Оценка прохождения через ГЭБ (BBB), нормированная", "bbb_normalized_score"),
            ("Оценка по формуле Gupta для ГЭБ (BBB)", "gupta_bbb_score"),
            ("Оценка Gupta V1, если доступна", "gupta_v1_score"),
            ("Вероятность активного выведения через P-gp", "pgp_probability"),
            ("pKa", "pka_pred"),
            ("Риск метаболического клиренса Clint", "clint_risk"),
            ("Показатель токсичности CATMoS", "catmos_ld50"),
            ("Итоговый учебный класс для ЦНС", "final_cns_class"),
        ],
        "decision_text": {
            "likely_cns_active": {
                "label": "Вероятный профиль с доступностью для ЦНС",
                "summary": "Модель оценивает прохождение через гематоэнцефалический барьер (ГЭБ, BBB) как благоприятное и не видит выраженного риска активного выведения через P-gp.",
                "interpretation": "Такой профиль поддерживает гипотезу о потенциальной доступности для ЦНС, но не доказывает проникновение экспериментально.",
            },
            "peripheral_action_risk": {
                "label": "Есть риск сниженной доступности для ЦНС",
                "summary": "Прохождение через ГЭБ (BBB) выглядит благоприятным, но модель видит риск активного выведения через P-gp.",
                "interpretation": "Молекула может проходить барьер, но затем частично выводиться транспортером P-gp. Поэтому вывод о действии в ЦНС требует осторожности.",
            },
            "likely_not_bbb_penetrant": {
                "label": "Вероятно низкое прохождение через ГЭБ",
                "summary": "P-gp не выглядит главным ограничением, но физико-химический профиль плохо поддерживает пассивное прохождение через ГЭБ (BBB).",
                "interpretation": "Главное ограничение здесь связано не с активным выведением, а со свойствами молекулы, которые мешают пассивной диффузии.",
            },
            "full_barrier": {
                "label": "Двойное ограничение для ЦНС",
                "summary": "Молекула сочетает неблагоприятное пассивное прохождение через ГЭБ (BBB) и высокий риск активного выведения через P-gp.",
                "interpretation": "Это наиболее неблагоприятный учебный сценарий для доступности в ЦНС: молекуле мешают и свойства для пассивной диффузии, и транспортер P-gp.",
            },
            "uncertain_or_borderline": {
                "label": "Неопределенный или пограничный результат",
                "summary": "Часть показателей находится рядом с порогами или разные блоки модели дают не полностью согласованные сигналы.",
                "interpretation": "Такой результат лучше использовать как повод для разбора дескрипторов и ограничений модели, а не как окончательный вывод.",
            },
            "insufficient_data": {
                "label": "Недостаточно данных для вывода",
                "summary": "Не хватает числовой оценки прохождения через ГЭБ (BBB) или оценки P-gp.",
                "interpretation": "Проверьте корректность SMILES и то, что расчетные модели вернули числовые значения.",
            },
        },
        "uncertainty_messages": {
            "low": "Основные сигналы модели согласованы; неопределенность в рамках учебного объяснения низкая.",
            "medium": "Прогноз нужно трактовать осторожно: отдельные показатели находятся рядом с порогами или дают не полностью согласованные сигналы.",
            "high": "Прогноз нужно трактовать очень осторожно: есть ошибки расчета или признаки выхода за область применимости модели.",
            "unknown": "Уровень неопределенности не определен.",
        },
    },
    "kk": {
        "title": "BioSynth-EDU: ADMET, BBB және ОЖЖ оқу есебі",
        "generated": "UTC генерация уақыты",
        "molecule": "Молекула",
        "input_smiles": "Бастапқы SMILES",
        "canonical_smiles": "Канондық SMILES",
        "validity": "Құрылымның дұрыстығы",
        "valid": "құрылым танылды",
        "invalid": "құрылым танылмады",
        "warnings": "Ескертулер",
        "none": "жоқ",
        "summary": "Қорытынды оқу тұжырымы",
        "uncertainty": "Белгісіздік деңгейі",
        "uncertainty_levels": {"low": "төмен", "medium": "орташа", "high": "жоғары", "unknown": "анық емес"},
        "scores": "Негізгі көрсеткіштер",
        "indicator": "Көрсеткіш",
        "value": "Мәні",
        "descriptors": "Дескрипторлар және түсіндірмелер",
        "descriptor": "Дескриптор",
        "zone": "Аймақ",
        "effect": "Әсері",
        "short_expl": "Қысқа түсіндірме",
        "factors": "Қолдайтын және шектейтін факторлар",
        "positive": "BBB арқылы өту мен ОЖЖ қолжетімділігін қолдайды",
        "negative": "BBB арқылы өтуге немесе ОЖЖ қолжетімділігіне кедергі жасайды",
        "borderline": "Шекаралық факторлар",
        "no_factors": "Бұл топта айқын фактор жоқ.",
        "matrix": "BBB × P-gp матрицасы",
        "current": "Ағымдағы сценарий",
        "interpretation": "Түсіндірме",
        "steps": "Модель шешімінің қадамдары",
        "step": "Қадам",
        "status": "Статус",
        "methodology": "Әдістеме",
        "limitations": "Шектеулер",
        "questions": "Студентке сұрақтар",
        "disclaimers": "Ескертпелер",
        "markdown": "Есептің Markdown нұсқасы",
        "in_silico_label": "Есептік болжам",
        "what_if_label": "What-if зертханасы",
        "scores_table": [
            ("BBB арқылы өту бағасы", "bbb_normalized_score"),
            ("BBB үшін Gupta формуласының бағасы", "gupta_bbb_score"),
            ("Gupta V1 бағасы, егер қолжетімді болса", "gupta_v1_score"),
            ("P-gp арқылы белсенді шығарылу ықтималдығы", "pgp_probability"),
            ("pKa", "pka_pred"),
            ("Clint метаболикалық клиренс қаупі", "clint_risk"),
            ("CATMoS токсикологиялық көрсеткіші", "catmos_ld50"),
            ("ОЖЖ үшін қорытынды оқу класы", "final_cns_class"),
        ],
        "decision_text": {
            "likely_cns_active": {
                "label": "ОЖЖ үшін қолжетімді профиль ықтимал",
                "summary": "Модель BBB арқылы өтуді қолайлы деп бағалайды және P-gp арқылы белсенді шығарылу қаупін жоғары көрмейді.",
                "interpretation": "Бұл профиль ОЖЖ үшін ықтимал қолжетімділікті қолдайды, бірақ эксперименттік дәлел болып саналмайды.",
            }
        },
        "uncertainty_messages": {
            "low": "Модельдің негізгі сигналдары келісілген; оқу түсіндірмесі шеңберінде белгісіздік төмен.",
            "medium": "Болжамды сақ түсіндіру керек: кейбір көрсеткіштер шектерге жақын немесе сигналдар толық сәйкес емес.",
            "high": "Болжамды өте сақ түсіндіру керек: есептеу қатесі немесе модельдің қолданылу аймағынан шығу белгілері бар.",
            "unknown": "Белгісіздік деңгейі анық емес.",
        },
    },
    "en": {
        "title": "BioSynth-EDU: ADMET, BBB and CNS student report",
        "generated": "Generated at UTC",
        "molecule": "Molecule",
        "input_smiles": "Input SMILES",
        "canonical_smiles": "Canonical SMILES",
        "validity": "Structure validity",
        "valid": "structure parsed",
        "invalid": "structure not parsed",
        "warnings": "Warnings",
        "none": "none",
        "summary": "Final educational conclusion",
        "uncertainty": "Uncertainty level",
        "uncertainty_levels": {"low": "low", "medium": "medium", "high": "high", "unknown": "unknown"},
        "scores": "Main indicators",
        "indicator": "Indicator",
        "value": "Value",
        "descriptors": "Descriptors and explanations",
        "descriptor": "Descriptor",
        "zone": "Zone",
        "effect": "Effect",
        "short_expl": "Short explanation",
        "factors": "Factors for and against",
        "positive": "Support BBB passage and CNS exposure",
        "negative": "Oppose BBB passage or CNS exposure",
        "borderline": "Borderline factors",
        "no_factors": "No clear factors in this group.",
        "matrix": "BBB × P-gp matrix",
        "current": "Current scenario",
        "interpretation": "Interpretation",
        "steps": "Stepwise model trace",
        "step": "Step",
        "status": "Status",
        "methodology": "Methodology",
        "limitations": "Limitations",
        "questions": "Questions for students",
        "disclaimers": "Disclaimers",
        "markdown": "Markdown version of the report",
        "in_silico_label": "Computational prediction",
        "what_if_label": "What-if lab",
        "scores_table": [
            ("Normalized BBB passage indicator", "bbb_normalized_score"),
            ("Gupta BBB indicator", "gupta_bbb_score"),
            ("Gupta V1 indicator, if available", "gupta_v1_score"),
            ("P-gp efflux probability", "pgp_probability"),
            ("pKa", "pka_pred"),
            ("Clint metabolic clearance risk", "clint_risk"),
            ("CATMoS toxicology indicator", "catmos_ld50"),
            ("Final educational CNS class", "final_cns_class"),
        ],
        "decision_text": {},
        "uncertainty_messages": {},
    },
}


def report_labels(lang: str) -> dict[str, Any]:
    return deepcopy(REPORT_TEXT.get(lang) or REPORT_TEXT["ru"])


def score_rows(lang: str) -> list[tuple[str, str]]:
    return list(report_labels(lang)["scores_table"])


def normalize_report_text(report: Mapping[str, Any], lang: str) -> dict[str, Any]:
    labels = report_labels(lang)
    normalized = deepcopy(dict(report or {}))
    normalized["title"] = labels["title"]
    summary = normalized.get("executive_summary") or {}
    scores = normalized.get("scores") or {}
    final_class = str(scores.get("final_cns_class") or summary.get("final_class") or "")
    decision_text = labels.get("decision_text", {}).get(final_class)
    if decision_text:
        summary["final_label_ru"] = decision_text["label"]
        summary["summary"] = decision_text["summary"]
        summary["student_interpretation"] = decision_text["interpretation"]
    level = str(summary.get("uncertainty_level") or "unknown")
    summary["uncertainty_level_label"] = labels.get("uncertainty_levels", {}).get(level, level)
    summary["uncertainty_message"] = labels.get("uncertainty_messages", {}).get(level, summary.get("uncertainty_message", ""))
    normalized["executive_summary"] = summary
    return normalized
