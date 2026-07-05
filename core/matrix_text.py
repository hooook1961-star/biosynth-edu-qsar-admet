"""Single source for the BBB/GEB x P-gp teaching matrix."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

MATRIX_TEXT: dict[str, dict[str, Any]] = {
    "ru": {
        "title": "Матрица ГЭБ (BBB) × P-gp",
        "intro": (
            "Матрица соединяет два разных вопроса: может ли молекула пассивно пройти через "
            "гематоэнцефалический барьер (ГЭБ, BBB), и есть ли риск активного выведения через P-gp."
        ),
        "current": "Текущий сценарий",
        "columns": {
            "bbb": "Прохождение через ГЭБ",
            "pgp": "P-gp",
            "scenario": "Учебный сценарий",
            "interpretation": "Интерпретация",
            "current": "Текущий",
        },
        "current_labels": {
            "bbb_high_pgp_low": "ГЭБ: высокая оценка; P-gp: не выглядит субстратом",
            "bbb_high_pgp_high": "ГЭБ: высокая оценка; P-gp: вероятный субстрат",
            "bbb_low_pgp_low": "ГЭБ: низкая оценка; P-gp: не выглядит субстратом",
            "bbb_low_pgp_high": "ГЭБ: низкая оценка; P-gp: вероятный субстрат",
            "borderline": "Пограничный сценарий",
            "insufficient_data": "Недостаточно данных",
            "invalid_or_error": "Некорректная структура или ошибка расчёта",
        },
        "cells": {
            "bbb_high_pgp_low": {
                "bbb_label": "ГЭБ: высокая оценка",
                "pgp_label": "P-gp: не выглядит субстратом",
                "label": "Профиль поддерживает доступность для ЦНС",
                "interpretation": "Молекула выглядит благоприятной для прохождения через ГЭБ, и модель не видит выраженного риска активного выведения через P-gp.",
            },
            "bbb_high_pgp_high": {
                "bbb_label": "ГЭБ: высокая оценка",
                "pgp_label": "P-gp: вероятный субстрат",
                "label": "Конфликт: проходит через ГЭБ, но может выводиться P-gp",
                "interpretation": "Пассивное прохождение через ГЭБ выглядит благоприятным, но P-gp может снижать фактическую доступность для ЦНС.",
            },
            "bbb_low_pgp_low": {
                "bbb_label": "ГЭБ: низкая оценка",
                "pgp_label": "P-gp: не выглядит субстратом",
                "label": "Главное ограничение - свойства для прохождения через ГЭБ",
                "interpretation": "P-gp не выглядит главным ограничением; основная проблема связана с физико-химическим профилем молекулы.",
            },
            "bbb_low_pgp_high": {
                "bbb_label": "ГЭБ: низкая оценка",
                "pgp_label": "P-gp: вероятный субстрат",
                "label": "Двойное ограничение для ЦНС",
                "interpretation": "Молекуле мешают и свойства для пассивного прохождения через ГЭБ, и возможное активное выведение через P-gp.",
            },
            "borderline": {
                "bbb_label": "ГЭБ: пограничная оценка",
                "pgp_label": "P-gp: пограничная оценка",
                "label": "Пограничный сценарий",
                "interpretation": "Один или оба показателя находятся рядом с порогами; вывод нужно трактовать осторожно.",
            },
            "insufficient_data": {
                "bbb_label": "ГЭБ: нет оценки",
                "pgp_label": "P-gp: нет оценки",
                "label": "Недостаточно данных",
                "interpretation": "Для матричной интерпретации не хватает оценки прохождения через ГЭБ или оценки P-gp.",
            },
        },
        "expander": "Как P-gp меняет интерпретацию ГЭБ?",
        "expander_text": (
            "Высокая оценка прохождения через ГЭБ не всегда означает высокую доступность для ЦНС: "
            "если молекула является субстратом P-gp, транспортер может активно выводить ее обратно в кровь."
        ),
    },
    "kk": {
        "title": "Қан-ми тосқауылы × P-gp матрицасы",
        "intro": "Матрица екі сұрақты біріктіреді: молекула қан-ми тосқауылынан пассивті өте ала ма және P-gp арқылы белсенді шығарылу қаупі бар ма.",
        "current": "Ағымдағы сценарий",
        "columns": {"bbb": "Қан-ми тосқауылынан өту", "pgp": "P-gp", "scenario": "Оқу сценарийі", "interpretation": "Түсіндірме", "current": "Ағымдағы"},
        "current_labels": {
            "bbb_high_pgp_low": "BBB: жоғары баға; P-gp: субстратқа ұқсамайды",
            "bbb_high_pgp_high": "BBB: жоғары баға; P-gp: ықтимал субстрат",
            "bbb_low_pgp_low": "BBB: төмен баға; P-gp: субстратқа ұқсамайды",
            "bbb_low_pgp_high": "BBB: төмен баға; P-gp: ықтимал субстрат",
            "borderline": "Шекаралық сценарий",
            "insufficient_data": "Дерек жеткіліксіз",
            "invalid_or_error": "Қате құрылым немесе есептеу қатесі",
        },
        "cells": {
            "bbb_high_pgp_low": {"bbb_label": "Қан-ми тосқауылы: жоғары баға", "pgp_label": "P-gp: белсенді шығарылу қаупі төмен", "label": "Профиль ОЖЖ қолжетімділігін қолдайды", "interpretation": "Молекула қан-ми тосқауылынан өтуге қолайлы көрінеді және P-gp арқылы белсенді шығарылу қаупі жоғары емес."},
            "bbb_high_pgp_high": {"bbb_label": "Қан-ми тосқауылы: жоғары баға", "pgp_label": "P-gp: белсенді шығарылу қаупі жоғары", "label": "Қайшылық: тосқауылдан өтеді, бірақ P-gp шығаруы мүмкін", "interpretation": "Қан-ми тосқауылынан пассивті өту қолайлы, бірақ P-gp ОЖЖ қолжетімділігін төмендетуі мүмкін."},
            "bbb_low_pgp_low": {"bbb_label": "Қан-ми тосқауылы: төмен баға", "pgp_label": "P-gp: белсенді шығарылу қаупі төмен", "label": "Негізгі шектеу - қан-ми тосқауылынан өту қасиеттері", "interpretation": "P-gp басты шектеу емес; негізгі мәселе молекуланың физика-химиялық профилімен байланысты."},
            "bbb_low_pgp_high": {"bbb_label": "Қан-ми тосқауылы: төмен баға", "pgp_label": "P-gp: белсенді шығарылу қаупі жоғары", "label": "ОЖЖ үшін қос шектеу", "interpretation": "Молекулаға қан-ми тосқауылынан пассивті өту қасиеттері де, P-gp арқылы белсенді шығарылу да кедергі болуы мүмкін."},
            "borderline": {"bbb_label": "BBB: шекаралық баға", "pgp_label": "P-gp: шекаралық баға", "label": "Шекаралық сценарий", "interpretation": "Бір немесе екі көрсеткіш шектерге жақын; қорытындыны сақ түсіндіру керек."},
            "insufficient_data": {"bbb_label": "Қан-ми тосқауылы: баға жоқ", "pgp_label": "P-gp: баға жоқ", "label": "Дерек жеткіліксіз", "interpretation": "Матрицалық түсіндіру үшін қан-ми тосқауылы немесе P-gp бағасы жеткіліксіз."},
        },
        "expander": "P-gp BBB түсіндірмесін қалай өзгертеді?",
        "expander_text": "Қан-ми тосқауылынан өту бағасы жоғары болса да, P-gp молекуланы қанға қайта шығарып, ОЖЖ қолжетімділігін төмендетуі мүмкін.",
    },
    "en": {
        "title": "BBB × P-gp matrix",
        "intro": "The matrix combines two questions: whether the molecule may pass the blood-brain barrier passively, and whether P-gp may actively remove it.",
        "current": "Current scenario",
        "columns": {"bbb": "BBB passage", "pgp": "P-gp", "scenario": "Teaching scenario", "interpretation": "Interpretation", "current": "Current"},
        "current_labels": {
            "bbb_high_pgp_low": "BBB: high estimate; P-gp: not substrate-like",
            "bbb_high_pgp_high": "BBB: high estimate; P-gp: likely substrate",
            "bbb_low_pgp_low": "BBB: low estimate; P-gp: not substrate-like",
            "bbb_low_pgp_high": "BBB: low estimate; P-gp: likely substrate",
            "borderline": "Borderline scenario",
            "insufficient_data": "Insufficient data",
            "invalid_or_error": "Invalid structure or calculation error",
        },
        "cells": {
            "bbb_high_pgp_low": {"bbb_label": "BBB: high estimate", "pgp_label": "P-gp: not substrate-like", "label": "Profile supports CNS exposure", "interpretation": "The molecule looks favourable for BBB passage, and the model does not see a strong P-gp efflux risk."},
            "bbb_high_pgp_high": {"bbb_label": "BBB: high estimate", "pgp_label": "P-gp: likely substrate", "label": "Conflict: BBB passage but possible P-gp efflux", "interpretation": "Passive BBB passage looks favourable, but P-gp may reduce actual CNS exposure."},
            "bbb_low_pgp_low": {"bbb_label": "BBB: low estimate", "pgp_label": "P-gp: not substrate-like", "label": "Main limitation is BBB-passage properties", "interpretation": "P-gp does not look like the main limitation; the main issue is the physicochemical profile."},
            "bbb_low_pgp_high": {"bbb_label": "BBB: low estimate", "pgp_label": "P-gp: likely substrate", "label": "Double limitation for CNS exposure", "interpretation": "Both passive BBB-passage properties and possible P-gp efflux may limit CNS exposure."},
            "borderline": {"bbb_label": "BBB: borderline estimate", "pgp_label": "P-gp: borderline estimate", "label": "Borderline scenario", "interpretation": "One or both indicators are close to thresholds; interpret cautiously."},
            "insufficient_data": {"bbb_label": "BBB: no estimate", "pgp_label": "P-gp: no estimate", "label": "Insufficient data", "interpretation": "BBB or P-gp estimates are missing for matrix interpretation."},
        },
        "expander": "How does P-gp change BBB interpretation?",
        "expander_text": "A high BBB-passage estimate does not always mean high CNS exposure: if the molecule is a P-gp substrate, the transporter may actively remove it back to blood.",
    },
}


def matrix_labels(lang: str) -> dict[str, Any]:
    return deepcopy(MATRIX_TEXT.get(lang) or MATRIX_TEXT["ru"])


def matrix_intro(lang: str) -> str:
    return str(matrix_labels(lang)["intro"])


def matrix_cells(lang: str) -> dict[str, dict[str, str]]:
    return deepcopy(matrix_labels(lang)["cells"])


def matrix_current_label(current_cell: str, lang: str) -> str:
    labels = matrix_labels(lang).get("current_labels", {})
    return str(labels.get(current_cell, current_cell))
