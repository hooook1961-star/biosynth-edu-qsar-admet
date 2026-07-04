"""Single source of student-report text."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


COMMON_DECISION_TEXT_RU = {
    "likely_cns_active": {
        "label": "Вероятный профиль с доступностью для ЦНС",
        "summary": "Модель оценивает прохождение через гематоэнцефалический барьер (ГЭБ, BBB) как благоприятное и не видит выраженного риска активного выведения через P-gp.",
        "interpretation": "Такой профиль поддерживает гипотезу о потенциальной доступности для ЦНС, но не доказывает проникновение экспериментально.",
    },
    "peripheral_action_risk": {
        "label": "Есть риск сниженной доступности для ЦНС",
        "summary": "Прохождение через ГЭБ выглядит благоприятным, но модель видит риск активного выведения через P-gp.",
        "interpretation": "Молекула может проходить барьер, но затем частично выводиться транспортером P-gp. Поэтому вывод о действии в ЦНС требует осторожности.",
    },
    "likely_not_bbb_penetrant": {
        "label": "Вероятно низкое прохождение через ГЭБ",
        "summary": "P-gp не выглядит главным ограничением, но физико-химический профиль плохо поддерживает пассивное прохождение через ГЭБ.",
        "interpretation": "Главное ограничение связано со свойствами молекулы, которые мешают пассивной диффузии.",
    },
    "full_barrier": {
        "label": "Двойное ограничение для ЦНС",
        "summary": "Молекула сочетает неблагоприятное пассивное прохождение через ГЭБ и высокий риск активного выведения через P-gp.",
        "interpretation": "Это наиболее неблагоприятный учебный сценарий для доступности в ЦНС.",
    },
    "uncertain_or_borderline": {
        "label": "Неопределённый или пограничный результат",
        "summary": "Часть показателей находится рядом с порогами или разные блоки модели дают не полностью согласованные сигналы.",
        "interpretation": "Такой результат лучше использовать как повод для разбора дескрипторов и ограничений модели, а не как окончательный вывод.",
    },
    "insufficient_data": {
        "label": "Недостаточно данных для вывода",
        "summary": "Не хватает числовой оценки прохождения через ГЭБ или оценки P-gp.",
        "interpretation": "Проверьте корректность SMILES и то, что расчётные модели вернули числовые значения.",
    },
}


REPORT_TEXT: dict[str, dict[str, Any]] = {
    "ru": {
        "title": "BioSynth-EDU: учебный отчёт ADMET, ГЭБ и ЦНС",
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
        "uncertainty": "Уровень неопределённости",
        "uncertainty_levels": {"low": "низкий", "medium": "средний", "high": "высокий", "unknown": "не определён"},
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
        "matrix": "Матрица ГЭБ (BBB) x P-gp",
        "current": "Текущий сценарий",
        "interpretation": "Интерпретация",
        "steps": "Пошаговое решение модели",
        "step": "Шаг",
        "status": "Статус",
        "methodology": "Методология",
        "limitations": "Ограничения",
        "questions": "Вопросы для студента",
        "disclaimers": "Оговорки",
        "markdown": "Markdown-версия отчёта",
        "in_silico_label": "Расчётный прогноз",
        "what_if_label": "Учебная What-if лаборатория",
        "scores_table": [
            ("Оценка прохождения через гематоэнцефалический барьер (ГЭБ, BBB)", "bbb_normalized_score"),
            ("Показатель по формуле Gupta для ГЭБ", "gupta_bbb_score"),
            ("Показатель Gupta V1, если доступен", "gupta_v1_score"),
            ("Оценка риска активного выведения через P-gp", "pgp_probability"),
            ("pKa", "pka_pred"),
            ("Риск метаболического клиренса Clint", "clint_risk"),
            ("Показатель токсикологического блока CATMoS", "catmos_ld50"),
            ("Итоговый учебный класс для ЦНС", "final_cns_class"),
        ],
        "decision_text": COMMON_DECISION_TEXT_RU,
        "uncertainty_messages": {
            "low": "Основные сигналы модели согласованы; неопределённость в рамках учебного объяснения низкая.",
            "medium": "Прогноз нужно трактовать осторожно: отдельные показатели находятся рядом с порогами или дают не полностью согласованные сигналы.",
            "high": "Прогноз нужно трактовать очень осторожно: есть ошибки расчёта или признаки, что молекула плохо похожа на структуры, на которых модель обычно работает надёжно.",
            "unknown": "Уровень неопределённости не определён.",
        },
        "methodology_sections": [
            {"title": "1. Проверка структуры", "text": "SMILES преобразуется в молекулярный граф. Если структуру не удаётся распознать, дескрипторы и учебное объяснение не строятся."},
            {"title": "2. Физико-химические дескрипторы", "text": "Для учебного объяснения используются молекулярная масса, LogP, TPSA, доноры и акцепторы водородных связей, pKa, заряд и оценка P-gp."},
            {"title": "3. Прохождение через ГЭБ", "text": "ГЭБ означает гематоэнцефалический барьер. Этот блок оценивает, насколько свойства молекулы совместимы с пассивным прохождением через такой барьер."},
            {"title": "4. P-gp", "text": "P-gp рассматривается как отдельный механизм активного выведения молекулы обратно в кровь. Он может снижать доступность для ЦНС даже при благоприятных свойствах для прохождения через ГЭБ."},
            {"title": "5. Учебное объяснение", "text": "Вывод строится как учебная интерпретация дескрипторов и модельных оценок. Он помогает разобрать ход рассуждения, но не заменяет экспериментальную проверку."},
        ],
        "limitations_list": [
            "BioSynth-EDU даёт расчётный прогноз, а не медицинскую рекомендацию.",
            "Прогноз не является экспериментальным доказательством прохождения через ГЭБ, токсичности или эффективности.",
            "Вероятности и показатели моделей следует читать как расчётные оценки, а не как абсолютную биологическую истину.",
            "What-if режим меняет дескрипторы без изменения химической структуры, поэтому это учебная симуляция, а не прогноз новой молекулы.",
            "Для солей, смесей, полифенолов, гликозидов и очень крупных молекул важно проверять, похожа ли молекула на те структуры, на которых модель обычно работает надёжно.",
        ],
        "student_questions_list": [
            "Какие дескрипторы сильнее всего поддерживают прохождение через ГЭБ?",
            "Какой фактор сильнее всего ограничивает доступность для ЦНС?",
            "Есть ли конфликт между пассивным прохождением через ГЭБ и активным выведением через P-gp?",
            "Как изменился бы учебный вывод, если TPSA или оценка P-gp стали выше?",
        ],
        "disclaimer_text": {
            "in_silico": "Это расчётный прогноз, а не экспериментальное доказательство. Интерпретируйте показатели вместе с дескрипторами и ограничениями модели.",
            "what_if": "What-if лаборатория является учебной симуляцией изменения дескрипторов, а не расчётом новой химической структуры.",
        },
    },
    "kk": {
        "title": "BioSynth-EDU: ADMET, BBB және ОЖЖ бойынша оқу есебі",
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
        "matrix": "BBB x P-gp матрицасы",
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
        "what_if_label": "Оқу What-if зертханасы",
        "scores_table": [
            ("Қан-ми тосқауылынан (BBB) өту бағасы", "bbb_normalized_score"),
            ("BBB үшін Gupta формуласы бойынша көрсеткіш", "gupta_bbb_score"),
            ("Gupta V1 көрсеткіші, егер қолжетімді болса", "gupta_v1_score"),
            ("P-gp арқылы белсенді шығарылу қаупінің бағасы", "pgp_probability"),
            ("pKa", "pka_pred"),
            ("Clint метаболикалық клиренс қаупі", "clint_risk"),
            ("CATMoS токсикологиялық блогының көрсеткіші", "catmos_ld50"),
            ("ОЖЖ үшін қорытынды оқу класы", "final_cns_class"),
        ],
        "decision_text": {},
        "uncertainty_messages": {
            "low": "Модельдің негізгі сигналдары келісілген; оқу түсіндірмесі шеңберінде белгісіздік төмен.",
            "medium": "Болжамды сақ түсіндіру керек: кейбір көрсеткіштер шекке жақын немесе сигналдар толық сәйкес емес.",
            "high": "Болжамды өте сақ түсіндіру керек: есептеу қатесі болуы мүмкін немесе молекула модель сенімді жұмыс істейтін құрылымдарға ұқсамауы мүмкін.",
            "unknown": "Белгісіздік деңгейі анық емес.",
        },
        "methodology_sections": [
            {"title": "1. Құрылымды тексеру", "text": "SMILES молекулалық графқа түрлендіріледі. Құрылым танылмаса, дескрипторлар мен оқу түсіндірмесі құрылмайды."},
            {"title": "2. Физика-химиялық дескрипторлар", "text": "Оқу түсіндірмесінде молекулалық масса, LogP, TPSA, сутектік байланыс донорлары мен акцепторлары, pKa, заряд және P-gp бағасы қолданылады."},
            {"title": "3. BBB арқылы өту", "text": "BBB қан-ми тосқауылын білдіреді. Бұл блок молекула қасиеттерінің осы тосқауылдан пассивті өтуге қаншалықты сәйкес келетінін бағалайды."},
            {"title": "4. P-gp", "text": "P-gp молекуланы қайтадан қанға белсенді шығара алатын бөлек механизм ретінде қарастырылады. Сондықтан BBB арқылы өту қолайлы болса да, ОЖЖ қолжетімділігі төмендеуі мүмкін."},
            {"title": "5. Оқу түсіндірмесі", "text": "Қорытынды дескрипторлар мен модельдік бағалардың оқу интерпретациясы ретінде беріледі. Ол ойлау логикасын түсіндіреді, бірақ эксперименттік тексеруді алмастырмайды."},
        ],
        "limitations_list": [
            "BioSynth-EDU есептік болжам береді, медициналық ұсыныс емес.",
            "Болжам BBB арқылы өту, уыттылық немесе тиімділік бойынша эксперименттік дәлел емес.",
            "Модель ықтималдықтары мен көрсеткіштерін абсолютті биологиялық шындық емес, есептік бағалар ретінде түсіндіру керек.",
            "What-if режимі химиялық құрылымды өзгертпей дескрипторларды өзгертеді, сондықтан бұл жаңа молекула болжамы емес, оқу симуляциясы.",
            "Тұздар, қоспалар, полифенолдар, гликозидтер және өте ірі молекулалар үшін молекула модель сенімді жұмыс істейтін құрылымдарға ұқсай ма, соны бөлек тексеру маңызды.",
        ],
        "student_questions_list": [
            "BBB арқылы өтуді ең көп қолдайтын дескрипторлар қайсы?",
            "ОЖЖ қолжетімділігін ең көп шектейтін фактор қандай?",
            "BBB арқылы пассивті өту мен P-gp арқылы белсенді шығарылу арасында қайшылық бар ма?",
            "TPSA немесе P-gp бағасы жоғарыласа, оқу қорытындысы қалай өзгерер еді?",
        ],
        "disclaimer_text": {
            "in_silico": "Бұл есептік болжам, эксперименттік дәлел емес. Көрсеткіштерді дескрипторлармен және модель шектеулерімен бірге түсіндіріңіз.",
            "what_if": "What-if зертханасы дескрипторларды өзгертуге арналған оқу симуляциясы; бұл жаңа химиялық құрылымды есептеу емес.",
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
        "summary": "Final teaching conclusion",
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
        "factors": "Supporting and opposing factors",
        "positive": "Support BBB passage and CNS exposure",
        "negative": "Oppose BBB passage or CNS exposure",
        "borderline": "Borderline factors",
        "no_factors": "No clear factors in this group.",
        "matrix": "BBB x P-gp matrix",
        "current": "Current scenario",
        "interpretation": "Interpretation",
        "steps": "Stepwise model trace",
        "step": "Step",
        "status": "Status",
        "methodology": "Methodology",
        "limitations": "Limitations",
        "questions": "Questions for students",
        "disclaimers": "Notes",
        "markdown": "Markdown report",
        "in_silico_label": "Computational prediction",
        "what_if_label": "Teaching What-if lab",
        "scores_table": [
            ("Blood-brain barrier (BBB) passage estimate", "bbb_normalized_score"),
            ("Gupta indicator for BBB", "gupta_bbb_score"),
            ("Gupta V1 indicator, if available", "gupta_v1_score"),
            ("P-gp active-efflux risk estimate", "pgp_probability"),
            ("pKa", "pka_pred"),
            ("Clint metabolic clearance risk", "clint_risk"),
            ("CATMoS toxicology-block indicator", "catmos_ld50"),
            ("Final educational CNS class", "final_cns_class"),
        ],
        "decision_text": {},
        "uncertainty_messages": {
            "low": "The main model signals are aligned; uncertainty is low within this teaching explanation.",
            "medium": "Interpret cautiously: some indicators are close to thresholds or not fully aligned.",
            "high": "Interpret very cautiously: calculation errors may be present or the molecule may not resemble structures for which the model is usually reliable.",
            "unknown": "Uncertainty level is unknown.",
        },
        "methodology_sections": [
            {"title": "1. Structure check", "text": "The SMILES string is converted into a molecular graph. If parsing fails, descriptors and the teaching explanation are not generated."},
            {"title": "2. Physicochemical descriptors", "text": "The teaching explanation uses molecular weight, LogP, TPSA, H-bond donors and acceptors, pKa, charge and the P-gp estimate."},
            {"title": "3. BBB passage", "text": "BBB means blood-brain barrier. This block estimates whether molecular properties are compatible with passive passage across that barrier."},
            {"title": "4. P-gp", "text": "P-gp is treated as a separate active efflux mechanism that can return a molecule to blood and reduce CNS exposure."},
            {"title": "5. Teaching explanation", "text": "The conclusion is an educational interpretation of descriptors and model estimates. It explains the reasoning but does not replace experimental validation."},
        ],
        "limitations_list": [
            "BioSynth-EDU provides a computational prediction, not medical advice.",
            "The prediction is not experimental proof of BBB passage, toxicity or efficacy.",
            "Model probabilities and indicators should be treated as computational estimates, not absolute biological truth.",
            "The What-if mode changes descriptors without changing the chemical structure, so it is a teaching simulation rather than a new-molecule prediction.",
            "For salts, mixtures, polyphenols, glycosides and very large molecules, check whether the molecule resembles structures for which the model is usually reliable.",
        ],
        "student_questions_list": [
            "Which descriptors most strongly support BBB passage?",
            "What most strongly limits CNS exposure?",
            "Is there a conflict between passive BBB passage and active P-gp efflux?",
            "How would the teaching conclusion change if TPSA or the P-gp estimate increased?",
        ],
        "disclaimer_text": {
            "in_silico": "This is a computational prediction, not experimental proof. Interpret the indicators together with descriptors and model limitations.",
            "what_if": "The What-if lab is a teaching simulation of descriptor changes, not a calculation of a new chemical structure.",
        },
    },
}


def report_labels(lang: str) -> dict[str, Any]:
    return deepcopy(REPORT_TEXT.get(lang) or REPORT_TEXT["ru"])


def score_rows(lang: str) -> list[tuple[str, str]]:
    return list(report_labels(lang)["scores_table"])


def report_methodology(lang: str) -> list[dict[str, str]]:
    return deepcopy(report_labels(lang).get("methodology_sections", []))


def report_limitations(lang: str) -> list[str]:
    return list(report_labels(lang).get("limitations_list", []))


def report_questions(lang: str) -> list[str]:
    return list(report_labels(lang).get("student_questions_list", []))


def report_disclaimer(kind: str, lang: str) -> str:
    return str(report_labels(lang).get("disclaimer_text", {}).get(kind, ""))


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
