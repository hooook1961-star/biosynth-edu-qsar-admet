"""Student-facing text for the optional ML explanation tab.

Keep this file free of Streamlit code and model calculations. It is copy only:
labels, short explanations, and localized wording for the ML tab.
"""

from __future__ import annotations

from typing import Any

from core.i18n import normalize_language


ML_UI_TEXT = {
    "ru": {
        "title": "### ML-разбор: дополнительная проверка модели",
        "intro": (
            "Этот раздел не заменяет химическое объяснение. Он показывает, какие типы данных "
            "машинная модель использовала при расчете вероятности. Для учебного вывода сначала "
            "смотрите раздел с дескрипторами: MW, LogP, TPSA, HBD/HBA, pKa и P-gp."
        ),
        "how_to": (
            "Если показана только важность групп, это значит: модель заметила такие признаки, "
            "но этот режим не говорит, повышают они вероятность или снижают. Поэтому используйте "
            "таблицу как подсказку, а не как самостоятельный вывод."
        ),
        "student_label": "Как это читать",
        "pgp_positive_class": "Модель оценивает вероятность, что молекула является субстратом P-gp.",
        "bbb_positive_class": "Модель оценивает вероятность класса BBB+.",
        "pgp_use": (
            "P-gp может выносить молекулу обратно из клеток барьера. Поэтому высокий P-gp-сигнал "
            "может ухудшать доступность для ЦНС даже при хороших свойствах пассивного проникновения."
        ),
        "bbb_use": (
            "Это дополнительная статистическая модель для BBB. Основной учебный разбор лучше читать "
            "через дескрипторы и формулу Gupta, потому что там видно химическую причину вывода. "
            "Формула Gupta должна быть описана и процитирована в разделе методологии."
        ),
        "model_result": "Результат модели",
        "probability": "Вероятность класса",
        "class": "Вывод модели",
        "threshold": "Порог",
        "method": "Тип объяснения",
        "method_shap": "локальное объяснение признаков",
        "method_fallback": "только важность групп",
        "group_table": "Какие группы данных заметила модель",
        "group": "Группа данных",
        "value": "Что найдено",
        "effect": "Как понимать",
        "contribution": "Сила",
        "developer_details": "Технические детали ML-разбора",
        "download_json": "Скачать технический JSON",
        "top_features": "Отдельные технические признаки",
        "unavailable": "Для этой молекулы или модели ML-разбор недоступен.",
        "structural_fragments": "структурные фрагменты молекулы",
        "structural_keys": "структурные шаблоны MACCS",
        "fragments_seen": "найдено активных фрагментов: {value}",
        "keys_seen": "найдено активных шаблонов: {value}",
        "scalar_seen": "значение: {value}",
        "effect_positive": "скорее поддерживает этот класс",
        "effect_negative": "скорее снижает вероятность этого класса",
        "effect_neutral": "влияние небольшое",
        "effect_importance": "модель использовала эту группу; направление здесь не показано",
        "class_pgp_1": "вероятный субстрат P-gp",
        "class_pgp_0": "скорее не субстрат P-gp",
        "class_bbb_1": "BBB+ по RF-модели",
        "class_bbb_0": "BBB- по RF-модели",
        "note_fallback": (
            "Сейчас доступна только важность групп. Это показывает, какие типы данных модель учитывала "
            "сильнее всего, но не объясняет направление влияния."
        ),
    },
    "kk": {
        "title": "### ML-талдау: модельдің қосымша тексеруі",
        "intro": (
            "Бұл бөлім химиялық түсіндіруді алмастырмайды. Ол машиналық модель ықтималдықты "
            "есептегенде қандай дерек түрлерін қолданғанын көрсетеді. Оқу қорытындысы үшін алдымен "
            "дескрипторлар бөлімін қараңыз: MW, LogP, TPSA, HBD/HBA, pKa және P-gp."
        ),
        "how_to": (
            "Егер тек топтардың маңыздылығы көрсетілсе, бұл модель осы белгілерді байқағанын білдіреді, "
            "бірақ олардың ықтималдықты арттыратынын не төмендететінін айтпайды. Сондықтан бұл кестені "
            "негізгі қорытынды емес, қосымша белгі ретінде қолданыңыз."
        ),
        "student_label": "Қалай оқу керек",
        "pgp_positive_class": "Модель молекуланың P-gp субстраты болу ықтималдығын бағалайды.",
        "bbb_positive_class": "Модель BBB+ класының ықтималдығын бағалайды.",
        "pgp_use": (
            "P-gp молекуланы барьер жасушаларынан қайта шығара алады. Сондықтан жоғары P-gp сигналы "
            "пассивті өту жақсы болса да, ЦНС үшін қолжетімділікті төмендетуі мүмкін."
        ),
        "bbb_use": (
            "Бұл BBB үшін қосымша статистикалық модель. Негізгі оқу талдауын дескрипторлар мен Gupta "
            "формуласы арқылы оқыған дұрыс, себебі онда қорытындының химиялық себебі көрінеді. "
            "Gupta формуласы әдістеме бөлімінде сипатталып, дереккөзі көрсетілуі керек."
        ),
        "model_result": "Модель нәтижесі",
        "probability": "Класс ықтималдығы",
        "class": "Модель қорытындысы",
        "threshold": "Шек",
        "method": "Түсіндіру түрі",
        "method_shap": "белгілердің жергілікті түсіндірмесі",
        "method_fallback": "тек топтардың маңыздылығы",
        "group_table": "Модель байқаған дерек топтары",
        "group": "Дерек тобы",
        "value": "Не табылды",
        "effect": "Қалай түсіну керек",
        "contribution": "Күші",
        "developer_details": "ML-талдаудың техникалық мәліметтері",
        "download_json": "Техникалық JSON жүктеу",
        "top_features": "Жеке техникалық белгілер",
        "unavailable": "Бұл молекула немесе модель үшін ML-талдау қолжетімсіз.",
        "structural_fragments": "молекуланың құрылымдық фрагменттері",
        "structural_keys": "MACCS құрылымдық үлгілері",
        "fragments_seen": "табылған белсенді фрагменттер: {value}",
        "keys_seen": "табылған белсенді үлгілер: {value}",
        "scalar_seen": "мәні: {value}",
        "effect_positive": "осы класты қолдауы мүмкін",
        "effect_negative": "осы кластың ықтималдығын төмендетуі мүмкін",
        "effect_neutral": "әсері аз",
        "effect_importance": "модель бұл топты қолданды; бағыты бұл жерде көрсетілмейді",
        "class_pgp_1": "ықтимал P-gp субстраты",
        "class_pgp_0": "P-gp субстраты емес болуы мүмкін",
        "class_bbb_1": "RF-модель бойынша BBB+",
        "class_bbb_0": "RF-модель бойынша BBB-",
        "note_fallback": (
            "Қазір тек топтардың маңыздылығы қолжетімді. Бұл модель қандай дерек түрлерін көбірек "
            "ескергенін көрсетеді, бірақ әсер бағытын түсіндірмейді."
        ),
    },
    "en": {
        "title": "### ML breakdown: supplementary model check",
        "intro": (
            "This section does not replace the chemistry explanation. It shows which data types the machine-learning "
            "model used when estimating probability. For the teaching conclusion, start with the descriptor section: "
            "MW, LogP, TPSA, HBD/HBA, pKa and P-gp."
        ),
        "how_to": (
            "If only group importance is shown, the model noticed these feature groups, but this mode does not say "
            "whether they raise or lower the probability. Use the table as a clue, not as a standalone conclusion."
        ),
        "student_label": "How to read this",
        "pgp_positive_class": "The model estimates the probability that the molecule is a P-gp substrate.",
        "bbb_positive_class": "The model estimates the probability of the BBB+ class.",
        "pgp_use": (
            "P-gp can pump a molecule back out of barrier cells. A high P-gp signal may reduce CNS availability "
            "even when passive BBB properties look good."
        ),
        "bbb_use": (
            "This is a supplementary statistical BBB model. The main teaching interpretation is better read through "
            "descriptors and the Gupta formula, because those show the chemical reason for the conclusion. The Gupta "
            "formula should be described and cited in the methodology section."
        ),
        "model_result": "Model result",
        "probability": "Class probability",
        "class": "Model conclusion",
        "threshold": "Threshold",
        "method": "Explanation type",
        "method_shap": "local feature explanation",
        "method_fallback": "group importance only",
        "group_table": "Data groups noticed by the model",
        "group": "Data group",
        "value": "What was found",
        "effect": "How to read it",
        "contribution": "Strength",
        "developer_details": "Technical ML details",
        "download_json": "Download technical JSON",
        "top_features": "Individual technical features",
        "unavailable": "ML breakdown is unavailable for this molecule or model.",
        "structural_fragments": "molecular structural fragments",
        "structural_keys": "MACCS structural patterns",
        "fragments_seen": "active fragments found: {value}",
        "keys_seen": "active patterns found: {value}",
        "scalar_seen": "value: {value}",
        "effect_positive": "tends to support this class",
        "effect_negative": "tends to lower this class probability",
        "effect_neutral": "small effect",
        "effect_importance": "the model used this group; direction is not shown here",
        "class_pgp_1": "likely P-gp substrate",
        "class_pgp_0": "likely not a P-gp substrate",
        "class_bbb_1": "BBB+ by RF model",
        "class_bbb_0": "BBB- by RF model",
        "note_fallback": (
            "Only group importance is available now. It shows which data types the model used most, "
            "but it does not explain the direction of the effect."
        ),
    },
}


def ml_ui_t(key: str, lang: str = "ru", **kwargs: Any) -> str:
    lang = normalize_language(lang)
    template = ML_UI_TEXT.get(lang, ML_UI_TEXT["ru"]).get(key, ML_UI_TEXT["ru"].get(key, key))
    try:
        return str(template).format(**kwargs)
    except Exception:
        return str(template)
