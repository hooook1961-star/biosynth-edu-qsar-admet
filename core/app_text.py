"""Application-level copy for BioSynth-EDU Streamlit screens."""

from __future__ import annotations

from typing import Any


APP_TEXT: dict[str, dict[str, str]] = {
    "ru": {
        "sidebar.stage": "**Учебная лаборатория ADMET / QSAR**",
        "sidebar.developer_mode": "Показать технические детали",
        "sidebar.developer_help": "Показывает статусы моделей и сведения для разработчика.",
        "sidebar.model_selection_hint": "Источник моделей:",
        "single.learning_note_title": "Цель анализа",
        "single.learning_note": (
            "Это учебная лаборатория по ADMET. Сначала посмотрите прогноз, затем разберите, какие свойства молекулы "
            "помогают или мешают прохождению через гематоэнцефалический барьер (ГЭБ, BBB). После этого можно открыть "
            "What-if лабораторию и проверить, как меняется учебная оценка доступности для ЦНС при изменении дескрипторов. "
            "ML-разбор является дополнительным техническим блоком: он показывает группы признаков, использованные "
            "RandomForest-моделью, а основной учебный вывод строится по дескрипторам и матрице ГЭБ x P-gp."
        ),
        "nav.main_mode": "Режим работы",
        "nav.single_section": "Раздел индивидуального анализа",
        "nav.batch_section": "Раздел массового анализа",
        "forecast.qsar_bridge_title": "QSAR-смысл прогноза",
        "forecast.qsar_bridge_text": (
            "Каждый показатель здесь является расчётным сигналом, а не экспериментальным доказательством. "
            "В учебном режиме важно увидеть связь: структура -> дескрипторы -> модельная оценка -> ADMET-интерпретация."
        ),
        "forecast.model_status_title": "Технический статус моделей",
        "forecast.model_status_caption": "Этот блок нужен для проверки локальной установки и не обязателен для студента.",
        "metric.catmos_score": "Показатель CATMoS",
        "help.catmos_score": "Показывается как расчётный показатель; его не следует автоматически читать как LD50 в mg/kg.",
        "metric.bbb_rf": "Оценка прохождения через ГЭБ по RF-модели",
        "help.bbb_rf": "Дополнительный ML-сигнал для ГЭБ. Основной учебный разбор строится по дескрипторам и формуле Gupta.",
        "metric.bbb_formula_version": "Формула Gupta",
        "batch.learning_intro": (
            "Массовый режим добавляет к каждой молекуле краткое учебное объяснение: итоговый класс для ЦНС, "
            "сценарий ГЭБ x P-gp, факторы за/против, предупреждения и уровень неопределённости."
        ),
        "batch.result_stored": "Результаты сохранены. Можно переключаться между разделами без повторного расчёта.",
        "common.na": "нет данных",
        "section.forecast": "Прогноз",
        "section.explain": "Разбор решения модели",
        "section.ml": "ML-разбор",
        "section.what_if": "What-if лаборатория",
        "section.report": "Учебный отчёт",
        "section.matrix": "Матрица ГЭБ x P-gp",
        "section.methodology": "Методология",
        "section.limitations": "Ограничения модели",
        "batch.section.summary": "Учебная сводка",
        "batch.section.table": "Компактная таблица screening_results",
        "batch.section.export": "Экспорт",
    },
    "kk": {
        "sidebar.stage": "**ADMET / QSAR оқу зертханасы**",
        "sidebar.developer_mode": "Техникалық мәліметтерді көрсету",
        "sidebar.developer_help": "Модельдердің статустары мен әзірлеушіге арналған мәліметтерді көрсетеді.",
        "sidebar.model_selection_hint": "Модельдер көзі:",
        "single.learning_note_title": "Талдаудың мақсаты",
        "single.learning_note": (
            "Бұл ADMET бойынша оқу зертханасы. Алдымен болжамды қараңыз, содан кейін молекуланың қай қасиеттері "
            "қан-ми тосқауылынан өтуге көмектесетінін немесе кедергі жасайтынын талдаңыз. Одан кейін What-if "
            "зертханасында дескрипторлар өзгергенде ОЖЖ қолжетімділігінің оқу бағасы қалай өзгеретінін тексеруге болады. "
            "ML-талдау қосымша техникалық блок: ол RandomForest моделі қолданған белгі топтарын көрсетеді, ал негізгі "
            "оқу қорытындысы дескрипторлар мен қан-ми тосқауылы x P-gp матрицасына сүйенеді."
        ),
        "nav.main_mode": "Жұмыс режимі",
        "nav.single_section": "Жеке талдау бөлімі",
        "nav.batch_section": "Массалық талдау бөлімі",
        "forecast.qsar_bridge_title": "Болжамның QSAR мағынасы",
        "forecast.qsar_bridge_text": (
            "Мұндағы әр көрсеткіш есептік сигнал болып табылады, эксперименттік дәлел емес. "
            "Оқу режимінде байланыс маңызды: құрылым -> дескрипторлар -> модельдік баға -> ADMET түсіндірмесі."
        ),
        "forecast.model_status_title": "Модельдердің техникалық статусы",
        "forecast.model_status_caption": "Бұл блок локалды орнатуды тексеруге арналған, студент үшін міндетті емес.",
        "metric.catmos_score": "CATMoS көрсеткіші",
        "help.catmos_score": "Есептік көрсеткіш ретінде беріледі; мәнді автоматты түрде LD50 mg/kg деп түсіндіруге болмайды.",
        "metric.bbb_rf": "RF-модель бойынша қан-ми тосқауылынан өту бағасы",
        "help.bbb_rf": "Қан-ми тосқауылы үшін қосымша модельдік белгі. Негізгі оқу талдауы дескрипторлар мен Gupta формуласына сүйенеді.",
        "metric.bbb_formula_version": "Gupta формуласы",
        "batch.learning_intro": (
            "Массалық режим әр молекулаға қысқа оқу түсіндірмесін қосады: ОЖЖ үшін қорытынды класс, "
            "қан-ми тосқауылы x P-gp сценарийі, қолдайтын/қарсы факторлар, ескертулер және белгісіздік деңгейі."
        ),
        "batch.result_stored": "Нәтижелер сақталды. Қайта есептемей бөлімдер арасында ауысуға болады.",
        "common.na": "деректер жоқ",
        "section.forecast": "Болжам",
        "section.explain": "Модель шешімін талдау",
        "section.ml": "ML-талдау",
        "section.what_if": "What-if зертханасы",
        "section.report": "Оқу есебі",
        "section.matrix": "Қан-ми тосқауылы x P-gp матрицасы",
        "section.methodology": "Әдістеме",
        "section.limitations": "Модель шектеулері",
        "batch.section.summary": "Оқу қорытындысы",
        "batch.section.table": "Ықшам screening_results кестесі",
        "batch.section.export": "Экспорт",
    },
    "en": {
        "sidebar.stage": "**Explainable ADMET / QSAR teaching lab**",
        "sidebar.developer_mode": "Show technical details",
        "sidebar.developer_help": "Shows model statuses and developer diagnostics.",
        "sidebar.model_selection_hint": "Model source:",
        "single.learning_note_title": "Purpose of the analysis",
        "single.learning_note": (
            "This is an ADMET teaching lab. Start with the prediction, then inspect which molecular properties support "
            "or oppose passage through the blood-brain barrier (BBB). Then use the What-if lab to see how the educational "
            "CNS-exposure estimate changes when descriptors are modified. The ML breakdown is an additional technical "
            "block; the main teaching conclusion is descriptor-based and uses the BBB x P-gp matrix."
        ),
        "nav.main_mode": "Mode",
        "nav.single_section": "Single-molecule analysis section",
        "nav.batch_section": "Batch analysis section",
        "forecast.qsar_bridge_title": "QSAR meaning of the prediction",
        "forecast.qsar_bridge_text": (
            "Each indicator here is a computational signal, not experimental evidence. In teaching mode, the goal is "
            "to see the connection: structure -> descriptors -> model estimate -> ADMET interpretation."
        ),
        "forecast.model_status_title": "Technical model status",
        "forecast.model_status_caption": "This block is for local diagnostics and is not required for students.",
        "metric.catmos_score": "CATMoS indicator",
        "help.catmos_score": "Displayed as a computational indicator; do not automatically read it as LD50 in mg/kg.",
        "metric.bbb_rf": "RF estimate of BBB passage",
        "help.bbb_rf": "Supplementary ML signal for BBB. The main teaching analysis uses descriptors and the Gupta formula.",
        "metric.bbb_formula_version": "Gupta formula",
        "batch.learning_intro": (
            "Batch mode adds a compact teaching explanation for each molecule: final CNS class, BBB x P-gp scenario, "
            "supporting/opposing factors, warnings and uncertainty level."
        ),
        "batch.result_stored": "Results are stored. You can switch sections without recalculating.",
        "common.na": "N/A",
        "section.forecast": "Prediction",
        "section.explain": "Model decision breakdown",
        "section.ml": "ML breakdown",
        "section.what_if": "What-if lab",
        "section.report": "Student report",
        "section.matrix": "BBB x P-gp matrix",
        "section.methodology": "Methodology",
        "section.limitations": "Model limitations",
        "batch.section.summary": "Teaching summary",
        "batch.section.table": "Compact screening_results table",
        "batch.section.export": "Export",
    },
}


def tx(key: str, lang: str, **kwargs: Any) -> str:
    template = APP_TEXT.get(lang, APP_TEXT["ru"]).get(key, APP_TEXT["ru"].get(key, key))
    try:
        return template.format(**kwargs)
    except Exception:
        return template
