"""Student-facing text for the optional ML explanation tab.

Keep this file free of Streamlit code and model calculations. It is copy only:
labels, short explanations, and localized wording for the ML tab.
"""

from __future__ import annotations

from typing import Any

from core.i18n import normalize_language


ML_UI_TEXT = {
    "ru": {
        "title": "### ML-разбор: дополнительная проверка",
        "intro": (
            "Этот раздел не заменяет химическое объяснение. Он показывает, какие группы данных "
            "статистическая модель использовала при расчёте вероятности. Основной учебный вывод "
            "лучше читать по дескрипторам: размеру, липофильности, полярности, водородным связям, "
            "заряду и риску P-gp."
        ),
        "how_to": (
            "Таблица ниже отвечает на простой вопрос: какие признаки молекулы модель заметила сильнее всего. "
            "Если показана только важность группы, это подсказка о внимании модели, а не отдельное химическое "
            "доказательство."
        ),
        "student_label": "Как это читать",
        "pgp_positive_class": "Модель оценивает риск, что молекула будет активно выводиться через P-gp.",
        "bbb_positive_class": "Модель оценивает, похожа ли молекула на соединения, проходящие через ГЭБ.",
        "pgp_use": (
            "P-gp может возвращать молекулу из клеток барьера обратно в кровь. Поэтому высокий сигнал P-gp "
            "может снижать доступность для ЦНС даже при благоприятных свойствах пассивного прохождения через ГЭБ."
        ),
        "bbb_use": (
            "Это дополнительная статистическая проверка для гематоэнцефалического барьера (ГЭБ, BBB). "
            "Для учебного объяснения важнее дескрипторы: по ним видно, какая химическая причина стоит за выводом."
        ),
        "model_result": "Результат модели",
        "probability": "Оценка вероятности",
        "class": "Вывод модели",
        "threshold": "Порог",
        "method": "Тип объяснения",
        "method_shap": "локальное объяснение признаков",
        "method_fallback": "важность групп признаков",
        "group_table": "Какие группы данных использовала модель",
        "group": "Группа данных",
        "value": "Что найдено",
        "effect": "Как понимать",
        "contribution": "Сила сигнала",
        "developer_details": "Технические детали",
        "download_json": "Скачать технический JSON",
        "top_features": "Отдельные признаки модели",
        "unavailable": "Для этой молекулы или модели ML-разбор недоступен.",
        "loading": "Готовим ML-разбор...",
        "structural_fragments": "структурные фрагменты молекулы",
        "structural_keys": "структурные шаблоны MACCS",
        "fragments_seen": "найдено активных фрагментов: {value}",
        "keys_seen": "найдено активных шаблонов: {value}",
        "scalar_seen": "значение: {value}",
        "effect_positive": "скорее поддерживает такой вывод",
        "effect_negative": "скорее ослабляет такой вывод",
        "effect_neutral": "влияние небольшое",
        "effect_importance": "модель заметила эту группу; знак влияния здесь не рассчитывался",
        "class_pgp_1": "возможен активный вывод через P-gp",
        "class_pgp_0": "выраженный риск P-gp не виден",
        "class_bbb_1": "профиль похож на соединения, проходящие через ГЭБ",
        "class_bbb_0": "профиль похож на соединения со слабым прохождением через ГЭБ",
        "note_fallback": (
            "Сейчас доступна важность групп признаков. Она показывает, какие типы данных модель учитывала "
            "сильнее всего, но не доказывает химическую причину вывода."
        ),
    },
    "kk": {
        "title": "### ML-талдау: қосымша тексеру",
        "intro": (
            "Бұл бөлім химиялық түсіндіруді алмастырмайды. Ол статистикалық модель ықтималдықты "
            "есептегенде қандай дерек топтарын қолданғанын көрсетеді. Негізгі оқу қорытындысын "
            "дескрипторлар арқылы оқыған дұрыс: өлшем, липофильділік, полярлық, сутектік байланыстар, "
            "заряд және P-gp қаупі."
        ),
        "how_to": (
            "Төмендегі кесте қарапайым сұраққа жауап береді: модель молекуланың қай белгілерін көбірек "
            "ескерді. Егер тек топтың маңыздылығы көрсетілсе, бұл модельдің назарын көрсететін қосымша белгі, "
            "жеке химиялық дәлел емес."
        ),
        "student_label": "Қалай оқу керек",
        "pgp_positive_class": "Модель молекуланың P-gp арқылы белсенді шығарылу қаупін бағалайды.",
        "bbb_positive_class": "Модель молекула қан-ми тосқауылынан (BBB) өте алатын қосылыстарға ұқсай ма, соны бағалайды.",
        "pgp_use": (
            "P-gp молекуланы тосқауыл жасушаларынан қайта қанға шығара алады. Сондықтан P-gp сигналы жоғары болса, "
            "қан-ми тосқауылынан пассивті өту қолайлы көрінсе де, ОЖЖ үшін қолжетімділік төмендеуі мүмкін."
        ),
        "bbb_use": (
            "Бұл қан-ми тосқауылы (BBB) үшін қосымша статистикалық тексеру. Оқу түсіндіруі үшін дескрипторлар "
            "маңыздырақ: олар қорытындының химиялық себебін көрсетеді."
        ),
        "model_result": "Модель нәтижесі",
        "probability": "Ықтималдық бағасы",
        "class": "Модель қорытындысы",
        "threshold": "Шек",
        "method": "Түсіндіру түрі",
        "method_shap": "белгілердің жергілікті түсіндірмесі",
        "method_fallback": "белгі топтарының маңыздылығы",
        "group_table": "Модель қолданған дерек топтары",
        "group": "Дерек тобы",
        "value": "Не табылды",
        "effect": "Қалай түсіну керек",
        "contribution": "Сигнал күші",
        "developer_details": "Техникалық мәліметтер",
        "download_json": "Техникалық JSON жүктеу",
        "top_features": "Модельдің жеке белгілері",
        "unavailable": "Бұл молекула немесе модель үшін ML-талдау қолжетімсіз.",
        "loading": "ML-талдау дайындалып жатыр...",
        "structural_fragments": "молекуланың құрылымдық фрагменттері",
        "structural_keys": "MACCS құрылымдық үлгілері",
        "fragments_seen": "табылған белсенді фрагменттер: {value}",
        "keys_seen": "табылған белсенді үлгілер: {value}",
        "scalar_seen": "мәні: {value}",
        "effect_positive": "осы қорытындыны қолдауы мүмкін",
        "effect_negative": "осы қорытындыны әлсіретуі мүмкін",
        "effect_neutral": "әсері аз",
        "effect_importance": "модель бұл топты байқады; әсер белгісі бұл жерде есептелмеді",
        "class_pgp_1": "P-gp арқылы белсенді шығарылу мүмкін",
        "class_pgp_0": "айқын P-gp қаупі байқалмайды",
        "class_bbb_1": "профиль қан-ми тосқауылынан өтетін қосылыстарға ұқсайды",
        "class_bbb_0": "профиль қан-ми тосқауылынан нашар өтетін қосылыстарға ұқсайды",
        "note_fallback": (
            "Қазір белгі топтарының маңыздылығы қолжетімді. Ол модель қандай дерек түрлерін көбірек "
            "ескергенін көрсетеді, бірақ қорытындының химиялық себебін дәлелдемейді."
        ),
    },
    "en": {
        "title": "### ML breakdown: supplementary check",
        "intro": (
            "This section does not replace the chemistry explanation. It shows which groups of data the statistical "
            "model used when estimating probability. For the teaching conclusion, start with descriptors: size, "
            "lipophilicity, polarity, hydrogen bonding, charge, and P-gp risk."
        ),
        "how_to": (
            "The table answers a simple question: which molecular signals did the model notice most. If only group "
            "importance is shown, treat it as a clue about model attention, not as a separate chemistry proof."
        ),
        "student_label": "How to read this",
        "pgp_positive_class": "The model estimates the risk that the molecule is actively removed by P-gp.",
        "bbb_positive_class": "The model estimates whether the molecule resembles compounds that pass the BBB.",
        "pgp_use": (
            "P-gp can return a molecule from barrier cells back to the blood. A high P-gp signal can reduce CNS "
            "exposure even when passive BBB passage looks favorable."
        ),
        "bbb_use": (
            "This is a supplementary statistical check for the blood-brain barrier (BBB). For teaching, descriptors "
            "are more important because they show the chemical reason behind the conclusion."
        ),
        "model_result": "Model result",
        "probability": "Probability estimate",
        "class": "Model conclusion",
        "threshold": "Threshold",
        "method": "Explanation type",
        "method_shap": "local feature explanation",
        "method_fallback": "feature-group importance",
        "group_table": "Data groups used by the model",
        "group": "Data group",
        "value": "What was found",
        "effect": "How to read it",
        "contribution": "Signal strength",
        "developer_details": "Technical details",
        "download_json": "Download technical JSON",
        "top_features": "Individual model features",
        "unavailable": "ML breakdown is unavailable for this molecule or model.",
        "loading": "Preparing ML breakdown...",
        "structural_fragments": "molecular structural fragments",
        "structural_keys": "MACCS structural patterns",
        "fragments_seen": "active fragments found: {value}",
        "keys_seen": "active patterns found: {value}",
        "scalar_seen": "value: {value}",
        "effect_positive": "tends to support this conclusion",
        "effect_negative": "tends to weaken this conclusion",
        "effect_neutral": "small effect",
        "effect_importance": "the model noticed this group; effect direction was not calculated here",
        "class_pgp_1": "active removal by P-gp is possible",
        "class_pgp_0": "no clear P-gp risk is visible",
        "class_bbb_1": "profile resembles compounds that pass the BBB",
        "class_bbb_0": "profile resembles compounds with weak BBB passage",
        "note_fallback": (
            "Feature-group importance is available now. It shows which data types the model used most, "
            "but it does not prove the chemical reason for the conclusion."
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
