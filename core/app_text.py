"""Application-level copy for BioSynth-EDU Streamlit screens.

This module intentionally contains only high-level app UI text that has not yet
been migrated into the main i18n catalog. Keep calculation logic out of here.
"""

from __future__ import annotations

from typing import Any

APP_TEXT = {
    "ru": {
        "sidebar.stage": "**РЈС‡РµР±РЅР°СЏ Р»Р°Р±РѕСЂР°С‚РѕСЂРёСЏ Explainable ADMET / QSAR**",
        "sidebar.developer_mode": "РџРѕРєР°Р·Р°С‚СЊ С‚РµС…РЅРёС‡РµСЃРєРёРµ РґРµС‚Р°Р»Рё",
        "sidebar.developer_help": "РџРѕРєР°Р·С‹РІР°РµС‚ runtime model selection, JSON-СЃС‚Р°С‚СѓСЃС‹ РјРѕРґРµР»РµР№ Рё РґСЂСѓРіРёРµ РґРµС‚Р°Р»Рё РґР»СЏ СЂР°Р·СЂР°Р±РѕС‚С‡РёРєР°.",
        "sidebar.model_selection_hint": "РСЃС‚РѕС‡РЅРёРє runtime-РјРѕРґРµР»РµР№:",
        "single.learning_note_title": "Р¦РµР»СЊ Р°РЅР°Р»РёР·Р°",
        "single.learning_note": (
            "Р­С‚Р° СЃС‚СЂР°РЅРёС†Р° СѓСЃС‚СЂРѕРµРЅР° РєР°Рє СѓС‡РµР±РЅР°СЏ Р»Р°Р±РѕСЂР°С‚РѕСЂРёСЏ. РЎРЅР°С‡Р°Р»Р° РїРѕСЃРјРѕС‚СЂРёС‚Рµ РїСЂРѕРіРЅРѕР·, Р·Р°С‚РµРј СЂР°Р·Р±РµСЂРёС‚Рµ, "
            "РєР°РєРёРµ СЃРІРѕР№СЃС‚РІР° РјРѕР»РµРєСѓР»С‹ РїРѕРјРѕРіР°СЋС‚ РёР»Рё РјРµС€Р°СЋС‚ BBB-РїСЂРѕРЅРёС†Р°РµРјРѕСЃС‚Рё. РџРѕСЃР»Рµ СЌС‚РѕРіРѕ РјРѕР¶РЅРѕ РѕС‚РєСЂС‹С‚СЊ "
            "What-if Р»Р°Р±РѕСЂР°С‚РѕСЂРёСЋ Рё РїСЂРѕРІРµСЂРёС‚СЊ, РєР°Рє РјРµРЅСЏРµС‚СЃСЏ СѓС‡РµР±РЅС‹Р№ CNS-score РїСЂРё РёР·РјРµРЅРµРЅРёРё РґРµСЃРєСЂРёРїС‚РѕСЂРѕРІ. "
            "ML-СЂР°Р·Р±РѕСЂ РЅСѓР¶РµРЅ РєР°Рє РґРѕРїРѕР»РЅРёС‚РµР»СЊРЅС‹Р№ СЃР»РѕР№: РѕРЅ РїРѕРєР°Р·С‹РІР°РµС‚, РєР°РєРёРµ РіСЂСѓРїРїС‹ РїСЂРёР·РЅР°РєРѕРІ РёСЃРїРѕР»СЊР·РѕРІР°Р»Р° "
            "RandomForest-РјРѕРґРµР»СЊ, РЅРѕ РѕСЃРЅРѕРІРЅРѕР№ СѓС‡РµР±РЅС‹Р№ РІС‹РІРѕРґ СЃС‚СЂРѕРёС‚СЃСЏ С‡РµСЂРµР· РґРµСЃРєСЂРёРїС‚РѕСЂС‹ Рё РјР°С‚СЂРёС†Сѓ BBB Г— P-gp."
        ),
        "nav.main_mode": "Р РµР¶РёРј СЂР°Р±РѕС‚С‹",
        "nav.single_section": "Р Р°Р·РґРµР» РёРЅРґРёРІРёРґСѓР°Р»СЊРЅРѕРіРѕ Р°РЅР°Р»РёР·Р°",
        "nav.batch_section": "Р Р°Р·РґРµР» РјР°СЃСЃРѕРІРѕРіРѕ Р°РЅР°Р»РёР·Р°",
        "forecast.qsar_bridge_title": "QSAR-СЃРјС‹СЃР» РїСЂРѕРіРЅРѕР·Р°",
        "forecast.qsar_bridge_text": (
            "РљР°Р¶РґР°СЏ РјРµС‚СЂРёРєР° Р·РґРµСЃСЊ вЂ” СЌС‚Рѕ РјРѕРґРµР»СЊРЅС‹Р№ СЃРёРіРЅР°Р», Р° РЅРµ СЌРєСЃРїРµСЂРёРјРµРЅС‚Р°Р»СЊРЅРѕРµ РґРѕРєР°Р·Р°С‚РµР»СЊСЃС‚РІРѕ. "
            "Р’ СѓС‡РµР±РЅРѕРј СЂРµР¶РёРјРµ РІР°Р¶РЅРѕ РїРѕРЅСЏС‚СЊ СЃРІСЏР·СЊ: СЃС‚СЂСѓРєС‚СѓСЂР° в†’ РґРµСЃРєСЂРёРїС‚РѕСЂС‹ в†’ РјРѕРґРµР»СЊРЅС‹Р№ score в†’ ADMET-РёРЅС‚РµСЂРїСЂРµС‚Р°С†РёСЏ."
        ),
        "forecast.model_status_title": "РўРµС…РЅРёС‡РµСЃРєРёР№ СЃС‚Р°С‚СѓСЃ РјРѕРґРµР»РµР№",
        "forecast.model_status_caption": "Р­С‚РѕС‚ Р±Р»РѕРє РЅСѓР¶РµРЅ РґР»СЏ РѕС‚Р»Р°РґРєРё Р»РѕРєР°Р»СЊРЅРѕР№ СѓСЃС‚Р°РЅРѕРІРєРё Рё РЅРµ РѕР±СЏР·Р°С‚РµР»РµРЅ РґР»СЏ СЃС‚СѓРґРµРЅС‚Р°.",
        "metric.catmos_score": "CATMoS score",
        "help.catmos_score": (
            "РџРѕРєР°Р·С‹РІР°РµС‚СЃСЏ РєР°Рє РјРѕРґРµР»СЊРЅС‹Р№ score. РЁРєР°Р»Р° consensus_LD50 РІС‹РіР»СЏРґРёС‚ log/transformed, "
            "РїРѕСЌС‚РѕРјСѓ Р·РЅР°С‡РµРЅРёРµ РЅРµ СЃР»РµРґСѓРµС‚ Р°РІС‚РѕРјР°С‚РёС‡РµСЃРєРё РёРЅС‚РµСЂРїСЂРµС‚РёСЂРѕРІР°С‚СЊ РєР°Рє mg/kg."
        ),
        "metric.bbb_rf": "BBB RF probability",
        "help.bbb_rf": "Р”РѕРїРѕР»РЅРёС‚РµР»СЊРЅС‹Р№ ML-СЃРёРіРЅР°Р» BBB. РћСЃРЅРѕРІРЅРѕР№ BBB-Р±Р»РѕРє вЂ” РёСЃРїСЂР°РІР»РµРЅРЅР°СЏ С„РѕСЂРјСѓР»Р° Gupta.",
        "metric.bbb_formula_version": "Gupta formula",
        "batch.learning_intro": (
            "РњР°СЃСЃРѕРІС‹Р№ СЂРµР¶РёРј РґРѕР±Р°РІР»СЏРµС‚ Рє РєР°Р¶РґРѕР№ РјРѕР»РµРєСѓР»Рµ РєСЂР°С‚РєРѕРµ СѓС‡РµР±РЅРѕРµ РѕР±СЉСЏСЃРЅРµРЅРёРµ: РёС‚РѕРіРѕРІС‹Р№ CNS-РєР»Р°СЃСЃ, "
            "СЃС†РµРЅР°СЂРёР№ BBB Г— P-gp, С„Р°РєС‚РѕСЂС‹ Р·Р°/РїСЂРѕС‚РёРІ, warnings Рё СѓСЂРѕРІРµРЅСЊ РЅРµРѕРїСЂРµРґРµР»С‘РЅРЅРѕСЃС‚Рё."
        ),
        "batch.result_stored": "Р РµР·СѓР»СЊС‚Р°С‚С‹ СЃРѕС…СЂР°РЅРµРЅС‹. РњРѕР¶РЅРѕ РїРµСЂРµРєР»СЋС‡Р°С‚СЊСЃСЏ РјРµР¶РґСѓ СЂР°Р·РґРµР»Р°РјРё Р±РµР· РїРѕРІС‚РѕСЂРЅРѕРіРѕ СЂР°СЃС‡С‘С‚Р°.",
        "common.na": "N/A",
        "section.forecast": "рџ“€ РџСЂРѕРіРЅРѕР·",
        "section.explain": "рџ§© Р Р°Р·Р±РѕСЂ СЂРµС€РµРЅРёСЏ РјРѕРґРµР»Рё",
        "section.ml": "рџ§  ML-СЂР°Р·Р±РѕСЂ",
        "section.what_if": "рџ§Є What-if Р»Р°Р±РѕСЂР°С‚РѕСЂРёСЏ",
        "section.report": "рџ“„ РЈС‡РµР±РЅС‹Р№ РѕС‚С‡С‘С‚",
        "section.matrix": "рџ§¬ РњР°С‚СЂРёС†Р° BBB Г— P-gp",
        "section.methodology": "рџ”¬ РњРµС‚РѕРґРѕР»РѕРіРёСЏ",
        "section.limitations": "вљ пёЏ РћРіСЂР°РЅРёС‡РµРЅРёСЏ РјРѕРґРµР»Рё",
        "batch.section.summary": "рџ“Љ РЈС‡РµР±РЅР°СЏ СЃРІРѕРґРєР°",
        "batch.section.table": "рџ§Є РџРѕР»РЅР°СЏ С‚Р°Р±Р»РёС†Р° ADMET + XAI",
        "batch.section.export": "рџ“Ґ Р­РєСЃРїРѕСЂС‚",
    },
    "kk": {
        "sidebar.stage": "**Explainable ADMET / QSAR РѕТ›Сѓ Р·РµСЂС‚С…Р°РЅР°СЃС‹**",
        "sidebar.developer_mode": "РўРµС…РЅРёРєР°Р»С‹Т› РјУ™Р»С–РјРµС‚С‚РµСЂРґС– РєУ©СЂСЃРµС‚Сѓ",
        "sidebar.developer_help": "Runtime model selection, РјРѕРґРµР»СЊРґРµСЂРґС–ТЈ JSON-СЃС‚Р°С‚СѓСЃС‚Р°СЂС‹ Р¶У™РЅРµ У™Р·С–СЂР»РµСѓС€С–РіРµ Р°СЂРЅР°Р»Т“Р°РЅ РјУ™Р»С–РјРµС‚С‚РµСЂРґС– РєУ©СЂСЃРµС‚РµРґС–.",
        "sidebar.model_selection_hint": "Runtime-РјРѕРґРµР»СЊРґРµСЂ РєУ©Р·С–:",
        "single.learning_note_title": "РўР°Р»РґР°СѓРґС‹ТЈ РјР°Т›СЃР°С‚С‹",
        "single.learning_note": (
            "Р‘Т±Р» Р±РµС‚ РѕТ›Сѓ Р·РµСЂС‚С…Р°РЅР°СЃС‹ СЂРµС‚С–РЅРґРµ Т›Т±СЂС‹Р»Т“Р°РЅ. РђР»РґС‹РјРµРЅ Р±РѕР»Р¶Р°РјРґС‹ Т›Р°СЂР°ТЈС‹Р·, СЃРѕРґР°РЅ РєРµР№С–РЅ РјРѕР»РµРєСѓР»Р°РЅС‹ТЈ "
            "Т›Р°Р№ Т›Р°СЃРёРµС‚С‚РµСЂС– BBB-У©С‚С–РјРґС–Р»С–РіС–РЅ Т›РѕР»РґР°Р№С‚С‹РЅС‹РЅ РЅРµРјРµСЃРµ С‚РµР¶РµР№С‚С–РЅС–РЅ С‚Р°Р»РґР°ТЈС‹Р·. РћРґР°РЅ РєРµР№С–РЅ What-if "
            "Р·РµСЂС‚С…Р°РЅР°СЃС‹РЅРґР° РґРµСЃРєСЂРёРїС‚РѕСЂР»Р°СЂРґС‹ У©Р·РіРµСЂС‚РєРµРЅРґРµ РѕТ›Сѓ CNS-score Т›Р°Р»Р°Р№ У©Р·РіРµСЂРµС‚С–РЅС–РЅ С‚РµРєСЃРµСЂСѓРіРµ Р±РѕР»Р°РґС‹. "
            "ML-С‚Р°Р»РґР°Сѓ Т›РѕСЃС‹РјС€Р° Т›Р°Р±Р°С‚ СЂРµС‚С–РЅРґРµ Т›Р°Р¶РµС‚: РѕР» RandomForest РјРѕРґРµР»С– Т›Р°РЅРґР°Р№ Р±РµР»РіС– С‚РѕРїС‚Р°СЂС‹РЅ Т›РѕР»РґР°РЅТ“Р°РЅС‹РЅ "
            "РєУ©СЂСЃРµС‚РµРґС–, Р±С–СЂР°Т› РЅРµРіС–Р·РіС– РѕТ›Сѓ Т›РѕСЂС‹С‚С‹РЅРґС‹СЃС‹ РґРµСЃРєСЂРёРїС‚РѕСЂР»Р°СЂ РјРµРЅ BBB Г— P-gp РјР°С‚СЂРёС†Р°СЃС‹ Р°СЂТ›С‹Р»С‹ Р±РµСЂС–Р»РµРґС–."
        ),
        "nav.main_mode": "Р–Т±РјС‹СЃ СЂРµР¶РёРјС–",
        "nav.single_section": "Р–РµРєРµ С‚Р°Р»РґР°Сѓ Р±У©Р»С–РјС–",
        "nav.batch_section": "РњР°СЃСЃР°Р»С‹Т› С‚Р°Р»РґР°Сѓ Р±У©Р»С–РјС–",
        "forecast.qsar_bridge_title": "Р‘РѕР»Р¶Р°РјРЅС‹ТЈ QSAR РјР°Т“С‹РЅР°СЃС‹",
        "forecast.qsar_bridge_text": (
            "РњТ±РЅРґР°Т“С‹ У™СЂ РјРµС‚СЂРёРєР° вЂ” РјРѕРґРµР»СЊРґС–Рє СЃРёРіРЅР°Р», СЌРєСЃРїРµСЂРёРјРµРЅС‚С‚С–Рє РґУ™Р»РµР» РµРјРµСЃ. РћТ›Сѓ СЂРµР¶РёРјС–РЅРґРµ Р±Р°Р№Р»Р°РЅС‹СЃ РјР°ТЈС‹Р·РґС‹: "
            "Т›Т±СЂС‹Р»С‹Рј в†’ РґРµСЃРєСЂРёРїС‚РѕСЂР»Р°СЂ в†’ РјРѕРґРµР»СЊРґС–Рє score в†’ ADMET С‚ТЇСЃС–РЅРґС–СЂРјРµСЃС–."
        ),
        "forecast.model_status_title": "РњРѕРґРµР»СЊРґРµСЂРґС–ТЈ С‚РµС…РЅРёРєР°Р»С‹Т› СЃС‚Р°С‚СѓСЃС‹",
        "forecast.model_status_caption": "Р‘Т±Р» Р±Р»РѕРє Р»РѕРєР°Р»РґС‹ РѕСЂРЅР°С‚СѓРґС‹ С‚РµРєСЃРµСЂСѓРіРµ Р°СЂРЅР°Р»Т“Р°РЅ, СЃС‚СѓРґРµРЅС‚ ТЇС€С–РЅ РјС–РЅРґРµС‚С‚С– РµРјРµСЃ.",
        "metric.catmos_score": "CATMoS score",
        "help.catmos_score": (
            "РњРѕРґРµР»СЊРґС–Рє score СЂРµС‚С–РЅРґРµ РєУ©СЂСЃРµС‚С–Р»РµРґС–. consensus_LD50 С€РєР°Р»Р°СЃС‹ log/transformed Р±РѕР»СѓС‹ РјТЇРјРєС–РЅ, "
            "СЃРѕРЅРґС‹Т›С‚Р°РЅ РјУ™РЅРґС– Р°РІС‚РѕРјР°С‚С‚С‹ С‚ТЇСЂРґРµ mg/kg РґРµРї С‚ТЇСЃС–РЅРґС–СЂСѓРіРµ Р±РѕР»РјР°Р№РґС‹."
        ),
        "metric.bbb_rf": "BBB RF С‹Т›С‚РёРјР°Р»РґС‹Т“С‹",
        "help.bbb_rf": "ТљРѕСЃС‹РјС€Р° BBB ML-СЃРёРіРЅР°Р»С‹. РќРµРіС–Р·РіС– BBB-Р±Р»РѕРє вЂ” С‚ТЇР·РµС‚С–Р»РіРµРЅ Gupta С„РѕСЂРјСѓР»Р°СЃС‹.",
        "metric.bbb_formula_version": "Gupta С„РѕСЂРјСѓР»Р°СЃС‹",
        "batch.learning_intro": (
            "РњР°СЃСЃР°Р»С‹Т› СЂРµР¶РёРј У™СЂ РјРѕР»РµРєСѓР»Р°Т“Р° Т›С‹СЃТ›Р° РѕТ›Сѓ С‚ТЇСЃС–РЅРґС–СЂРјРµСЃС–РЅ Т›РѕСЃР°РґС‹: Т›РѕСЂС‹С‚С‹РЅРґС‹ CNS-РєР»Р°СЃСЃ, "
            "BBB Г— P-gp СЃС†РµРЅР°СЂРёР№С–, Т›РѕР»РґР°Р№С‚С‹РЅ/Т›Р°СЂСЃС‹ С„Р°РєС‚РѕСЂР»Р°СЂ, warnings Р¶У™РЅРµ Р±РµР»РіС–СЃС–Р·РґС–Рє РґРµТЈРіРµР№С–."
        ),
        "batch.result_stored": "РќУ™С‚РёР¶РµР»РµСЂ СЃР°Т›С‚Р°Р»РґС‹. ТљР°Р№С‚Р° РµСЃРµРїС‚РµРјРµР№ Р±У©Р»С–РјРґРµСЂ Р°СЂР°СЃС‹РЅРґР° Р°СѓС‹СЃСѓТ“Р° Р±РѕР»Р°РґС‹.",
        "common.na": "N/A",
        "section.forecast": "рџ“€ Р‘РѕР»Р¶Р°Рј",
        "section.explain": "рџ§© РњРѕРґРµР»СЊ С€РµС€С–РјС–РЅ С‚Р°Р»РґР°Сѓ",
        "section.ml": "рџ§  ML-С‚Р°Р»РґР°Сѓ",
        "section.what_if": "рџ§Є What-if Р·РµСЂС‚С…Р°РЅР°СЃС‹",
        "section.report": "рџ“„ РћТ›Сѓ РµСЃРµР±С–",
        "section.matrix": "рџ§¬ BBB Г— P-gp РјР°С‚СЂРёС†Р°СЃС‹",
        "section.methodology": "рџ”¬ УРґС–СЃС‚РµРјРµ",
        "section.limitations": "вљ пёЏ РњРѕРґРµР»СЊ С€РµРєС‚РµСѓР»РµСЂС–",
        "batch.section.summary": "рџ“Љ РћТ›Сѓ Т›РѕСЂС‹С‚С‹РЅРґС‹СЃС‹",
        "batch.section.table": "рџ§Є РўРѕР»С‹Т› ADMET + XAI РєРµСЃС‚РµСЃС–",
        "batch.section.export": "рџ“Ґ Р­РєСЃРїРѕСЂС‚",
    },
    "en": {
        "sidebar.stage": "**Explainable ADMET / QSAR teaching lab**",
        "sidebar.developer_mode": "Show technical details",
        "sidebar.developer_help": "Shows runtime model selection, model JSON statuses and developer diagnostics.",
        "sidebar.model_selection_hint": "Runtime model source:",
        "single.learning_note_title": "Purpose of the analysis",
        "single.learning_note": (
            "This page is designed as a teaching lab. Start with the prediction, then inspect which molecular "
            "properties support or oppose BBB penetration. Then use the What-if lab to see how the educational "
            "CNS score changes when descriptors are modified. The ML breakdown is an additional layer: it shows "
            "which groups of features the RandomForest model used, while the main teaching conclusion remains "
            "descriptor-based and uses the BBB Г— P-gp matrix."
        ),
        "nav.main_mode": "Mode",
        "nav.single_section": "Single-molecule analysis section",
        "nav.batch_section": "Batch analysis section",
        "forecast.qsar_bridge_title": "QSAR meaning of the prediction",
        "forecast.qsar_bridge_text": (
            "Each metric here is a model signal, not experimental evidence. In teaching mode, the goal is to see "
            "the connection: structure в†’ descriptors в†’ model score в†’ ADMET interpretation."
        ),
        "forecast.model_status_title": "Technical model status",
        "forecast.model_status_caption": "This block is for local debugging and is not required for students.",
        "metric.catmos_score": "CATMoS score",
        "help.catmos_score": (
            "Displayed as a model score. The consensus_LD50 scale appears log/transformed, "
            "so the value should not automatically be interpreted as mg/kg."
        ),
        "metric.bbb_rf": "BBB RF probability",
        "help.bbb_rf": "Supplementary BBB ML signal. The primary BBB block is the corrected Gupta formula.",
        "metric.bbb_formula_version": "Gupta formula",
        "batch.learning_intro": (
            "Batch mode adds a compact teaching explanation for each molecule: final CNS class, "
            "BBB Г— P-gp scenario, supporting/opposing factors, warnings and uncertainty level."
        ),
        "batch.result_stored": "Results are stored. You can switch sections without recalculating.",
        "common.na": "N/A",
        "section.forecast": "рџ“€ Prediction",
        "section.explain": "рџ§© Model decision breakdown",
        "section.ml": "рџ§  ML breakdown",
        "section.what_if": "рџ§Є What-if lab",
        "section.report": "рџ“„ Student report",
        "section.matrix": "рџ§¬ BBB Г— P-gp matrix",
        "section.methodology": "рџ”¬ Methodology",
        "section.limitations": "вљ пёЏ Model limitations",
        "batch.section.summary": "рџ“Љ Teaching summary",
        "batch.section.table": "рџ§Є Full ADMET + XAI table",
        "batch.section.export": "рџ“Ґ Export",
    },
}



def tx(key: str, lang: str, **kwargs: Any) -> str:
    template = APP_TEXT.get(lang, APP_TEXT["ru"]).get(key, APP_TEXT["ru"].get(key, key))
    try:
        return template.format(**kwargs)
    except Exception:
        return template

