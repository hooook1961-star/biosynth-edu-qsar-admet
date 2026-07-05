"""Main multilingual catalog for BioSynth-EDU explainability.

Internal data keys stay stable; user-facing labels, messages, explanations,
warnings, reports and UI captions are localized here or in neighboring
content modules.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from core.text_catalog import (
    DESCRIPTOR_DISPLAY_NAMES,
    DESCRIPTOR_SHORT_LABELS,
    DESCRIPTOR_THRESHOLD_NOTES,
    ZONE_COMMENTS,
)

try:  # RU fallback comes from the accepted Stage 1 templates.
    from core.teaching_templates import (
        BASE_DESCRIPTOR_EXPLANATIONS_RU,
        FINAL_DECISION_TEXTS,
        IN_SILICO_DISCLAIMER_RU,
        WHAT_IF_DISCLAIMER_RU,
        DESCRIPTOR_META,
    )
except Exception:  # pragma: no cover - defensive for partial installs
    BASE_DESCRIPTOR_EXPLANATIONS_RU = {}
    FINAL_DECISION_TEXTS = {}
    IN_SILICO_DISCLAIMER_RU = ""
    WHAT_IF_DISCLAIMER_RU = ""
    DESCRIPTOR_META = {}

DEFAULT_LANGUAGE = "ru"
SUPPORTED_LANGUAGES = {
    "ru": "Русский",
    "kk": "Қазақша",
    "en": "English",
}
LANGUAGE_LABEL_TO_CODE = {value: key for key, value in SUPPORTED_LANGUAGES.items()}

ZONE_LABELS = {
    "ru": {"green": "помогает", "yellow": "погранично", "red": "мешает", "gray": "неопределённо"},
    "kk": {"green": "көмектеседі", "yellow": "шекаралық", "red": "кедергі", "gray": "анық емес"},
    "en": {"green": "supports", "yellow": "borderline", "red": "opposes", "gray": "uncertain"},
}

ZONE_BADGE_LABELS = {
    "ru": {"green": "🟢 помогает", "yellow": "🟡 погранично", "red": "🔴 мешает", "gray": "⚪ неопределённо"},
    "kk": {"green": "🟢 көмектеседі", "yellow": "🟡 шекаралық", "red": "🔴 кедергі", "gray": "⚪ анық емес"},
    "en": {"green": "🟢 supports", "yellow": "🟡 borderline", "red": "🔴 opposes", "gray": "⚪ uncertain"},
}

EFFECT_LABELS = {
    "ru": {
        "supports_bbb": "поддерживает прохождение через ГЭБ",
        "opposes_bbb": "снижает прохождение через ГЭБ",
        "borderline": "пограничное влияние",
        "supports_cns_exposure": "поддерживает доступность для ЦНС",
        "opposes_cns_exposure": "снижает доступность для ЦНС",
        "context_dependent": "зависит от контекста",
        "unknown": "недоступно",
        "uncertain": "неопределённо",
    },
    "kk": {
        "supports_bbb": "қан-ми тосқауылынан өтуді қолдайды",
        "opposes_bbb": "қан-ми тосқауылынан өтуді төмендетеді",
        "borderline": "шекаралық әсер",
        "supports_cns_exposure": "ОЖЖ қолжетімділігін қолдайды",
        "opposes_cns_exposure": "ОЖЖ қолжетімділігін төмендетеді",
        "context_dependent": "контекстке тәуелді",
        "unknown": "қолжетімсіз",
        "uncertain": "анық емес",
    },
    "en": {
        "supports_bbb": "supports BBB permeability",
        "opposes_bbb": "reduces BBB permeability",
        "borderline": "borderline effect",
        "supports_cns_exposure": "supports CNS exposure",
        "opposes_cns_exposure": "reduces CNS exposure",
        "context_dependent": "context-dependent",
        "unknown": "not available",
        "uncertain": "uncertain",
    },
}

UI = {'ru': {'language.select': 'Тіл / Язык / Language',
        'app.title': '🔬 Исследовательский модуль Биодоступность & ЦНС',
        'tabs.single': '🎯 Индивидуальный анализ',
        'tabs.batch': '📂 Массовый скрининг',
        'tabs.forecast': '📈 Прогноз',
        'tabs.explain': '🧩 Разбор решения модели',
        'tabs.what_if': '🧪 What-if лаборатория',
        'tabs.report': '📄 Учебный отчёт',
        'tabs.matrix': 'Матрица ГЭБ x P-gp',
        'tabs.methodology': '🔬 Методология',
        'tabs.limitations': '⚠️ Ограничения модели',
        'section.decision': '### 🧠 Учебный вывод модели',
        'section.molecule': '### 🧬 Исходная молекула',
        'section.descriptors': '### 📊 Ключевые дескрипторы',
        'section.factors': '### Факторы за и против прохождения через ГЭБ',
        'section.stepwise': '### 🧭 Пошаговое решение модели',
        'section.matrix': '### Матрица ГЭБ x P-gp',
        'section.what_if': '### 🧪 What-if лаборатория: измени свойство молекулы',
        'section.report': '### 📄 Учебный отчёт по молекуле',
        'section.methodology': '### 🔬 Методология',
        'section.limitations': '### ⚠️ Ограничения',
        'metric.bbb': 'Оценка прохождения через ГЭБ',
        'metric.pgp': 'Оценка P-gp',
        'metric.bbb_gupta': 'Показатель Gupta для ГЭБ',
        'metric.pgp_class': 'Статус P-gp',
        'metric.pgp_probability': 'Оценка риска P-gp',
        'metric.clint': 'Clint (метаболизм)',
        'metric.pka': 'pKa',
        'what_if.base_passive_bbb': 'Базовая оценка прохождения через ГЭБ',
        'what_if.base_cns': 'Базовая оценка доступности для ЦНС',
        'what_if.passive_bbb': 'Прохождение через ГЭБ',
        'what_if.educational_cns': 'Учебная оценка ЦНС-доступности',
        'what_if.slider.mw': 'Молекулярная масса (MW), Да',
        'what_if.slider.logp': 'Липофильность (LogP)',
        'what_if.slider.tpsa': 'Полярная поверхность (TPSA), A^2',
        'what_if.slider.pka': 'Кислотность/основность (pKa)',
        'what_if.slider.hbd': 'Доноры H-связей (HBD)',
        'what_if.slider.hba': 'Акцепторы H-связей (HBA)',
        'what_if.slider.pgp': 'Оценка риска P-gp',
        'valid.yes': '✅ SMILES распознан',
        'valid.no': '❌ SMILES не распознан',
        'warnings.none': 'Предупреждений нет.',
        'warnings.reasons': 'Причины предупреждений',
        'applicability.title': '#### Где модель работает надёжно',
        'descriptor.column.name': 'Дескриптор',
        'descriptor.column.value': 'Значение',
        'descriptor.column.unit': 'Ед.',
        'descriptor.column.zone': 'Зона',
        'descriptor.column.effect': 'Влияние',
        'descriptor.column.meaning': 'Смысл',
        'descriptor.details': 'Подробно: объяснение каждого дескриптора',
        'factors.positive': 'Поддерживают прохождение через ГЭБ',
        'factors.negative': 'Мешают прохождению через ГЭБ или доступности для ЦНС',
        'factors.borderline': '🟡 Пограничные',
        'factors.empty': 'Нет явных факторов в этой группе.',
        'matrix.current': 'Текущий сценарий',
        'matrix.pgp_expander': 'Почему P-gp может снижать доступность для ЦНС?',
        'what_if.slider_title': '#### Ползунки дескрипторов',
        'what_if.result': '#### Результат учебной симуляции',
        'what_if.changed_factors': '#### Какие факторы изменились',
        'what_if.commentary': '#### Учебный комментарий',
        'what_if.better': '##### Что стало лучше',
        'what_if.worse': '##### Что стало хуже',
        'what_if.no_improve': 'Нет выраженных улучшений по учебной эвристике.',
        'what_if.no_worse': 'Нет выраженных ухудшений по учебной эвристике.',
        'what_if.not_enough': 'Недостаточно дескрипторов для What-if симуляции.',
        'report.info': 'Этот отчёт собирает исходную молекулу, ключевые дескрипторы, факторы за и против, матрицу BBB × P-gp, пошаговое '
                       'решение модели, методологию и ограничения.',
        'report.preview': 'Предпросмотр Markdown-отчёта',
        'report.download_html': '📥 Скачать HTML-отчёт',
        'report.download_md': '📥 Скачать Markdown',
        'report.download_json': '📥 Скачать JSON',
        'batch.title': '### 📂 Сводка batch explainability',
        'batch.info': 'Этот блок добавляет к массовому скринингу краткое учебное объяснение для каждой молекулы.',
        'batch.total': 'Всего молекул',
        'batch.valid': 'Валидных',
        'batch.invalid': 'Ошибки / invalid',
        'batch.candidates': 'Кандидаты для ЦНС',
        'batch.summary': '#### Краткое резюме батча',
        'batch.table': 'Таблица батча, отсортированная по учебному приоритету',
        'sidebar.language': 'Тіл / Язык / Language',
        'sidebar.subtitle': '**Explainable ADMET/BBB + P-gp**',
        'sidebar.info': 'pKa + P-gp + формула Gupta + Clint + XAI',
        'single.input_header': '1. Входные данные',
        'single.smiles_label': 'SMILES:',
        'forecast.structure_title': '#### 📐 2D-граф молекулы',
        'forecast.descriptors_title': '#### 🧪 Полные дескрипторы',
        'forecast.integral_title': '#### 🧠 Интегральный прогноз (ADMET)',
        'forecast.visualization_title': '#### 📊 Визуализация',
        'forecast.descriptor_col': 'Дескриптор',
        'forecast.value_col': 'Значение',
        'metric.caco2': 'Caco-2 (проницаемость)',
        'metric.catmos': 'CATMoS (LD50)',
        'help.caco2': 'Прогноз логарифма коэффициента проницаемости LogPapp. LogPapp > -5.0: хорошая всасываемость в ЖКТ. LogPapp < -6.0: '
                      'плохая всасываемость.',
        'help.bbb_gupta': 'Расчётный показатель по формуле Gupta для оценки прохождения через гематоэнцефалический барьер.',
        'help.pgp_class': 'Класс 1 — высокий риск активного выведения через P-gp.',
        'help.pgp_probability': 'Оценка риска активного выведения через P-gp. Чем выше значение, тем сильнее этот риск.',
        'help.pka': 'Динамический pKa используется как часть гибридного BBB-индекса.',
        'help.clint': 'Прогноз печёночного клиренса / метаболической стабильности.',
        'help.catmos': 'In silico прогноз острой токсичности.',
        'status.yes': 'ДА',
        'status.no': 'НЕТ',
        'status.high': 'Высокая',
        'status.medium': 'Средняя',
        'status.low': 'Низкая',
        'status.high_risk': 'Высокий риск',
        'status.stable': 'Стабильное',
        'status.error': 'Ошибка',
        'status.na': 'N/A',
        'status.not_substrate': 'Выраженный риск P-gp не виден',
        'status.substrate_efflux': 'Возможен активный вывод через P-gp',
        'batch.page_title': '📂 Массовый скрининг + Explainable ADMET',
        'batch.intro_stage5': 'Массовая таблица дополняется кратким учебным объяснением для каждой молекулы: итоговый класс для ЦНС, сценарий '
                              'BBB × P-gp, факторы за/против, warnings и уровень неопределённости.',
        'batch.source_label': 'Источник:',
        'batch.source_file': 'Файл (CSV/Excel)',
        'batch.source_text': 'Текстовое поле (SMILES)',
        'batch.file_label': 'Файл:',
        'batch.smiles_col_label': 'Колонка SMILES:',
        'batch.text_area_label': 'SMILES (один на строку):',
        'batch.include_long_text': 'Добавить длинные объяснения в batch export',
        'batch.include_long_text_help': 'Для больших списков лучше оставить выключенным: Excel будет компактнее.',
        'batch.loaded_rows': 'Загружено строк: {n}',
        'batch.run_button': '🚀 Запустить массовый Explainable ADMET',
        'batch.progress': 'Расчёт: {progress}',
        'batch.done': 'Массовый Explainable ADMET расчёт завершён.',
        'batch.invalid_smiles_error': 'Некорректная структура SMILES',
        'batch.calculation_error': 'Ошибка расчёта: {error_type}: {error}',
        'batch.tab_summary': '📊 Explainability summary',
        'batch.tab_full_table': '🧪 Полная таблица ADMET + XAI',
        'batch.tab_export': '📥 Экспорт',
        'batch.show_all_columns': 'Показать все batch-колонки',
        'batch.download_excel': '📥 Скачать Excel: ADMET + Explainability',
        'batch.download_csv': '📥 Скачать CSV: XAI table',
        'batch.empty': 'Batch-таблица пуста.',
        'batch.result_empty': 'Batch-результат пуст.',
        'what_if.base': 'База',
        'what_if.col.factor': 'Фактор',
        'what_if.col.before': 'До',
        'what_if.col.after': 'После',
        'what_if.col.contribution': 'Вклад',
        'what_if.col.descriptor': 'Дескриптор',
        'what_if.col.from': 'Из зоны',
        'what_if.col.to': 'В зону',
        'matrix.col.scenario': 'Сценарий',
        'matrix.col.interpretation': 'Интерпретация',
        'matrix.col.current': 'Текущий',
        'msg.structure_unavailable': '2D-структура недоступна.',
        'msg.stepwise_unavailable': 'Пошаговое решение недоступно.',
        'label.input_smiles': 'Input SMILES',
        'label.canonical_smiles': 'Canonical SMILES',
        'label.validity': 'Валидность',
        'label.step': 'Шаг',
        'effect.unknown': 'недоступно',
        'label.after': 'После',
        'label.applicability_domain': 'Надёжность модели для этой молекулы',
        'label.before': 'До',
        'label.change': 'Изменение',
        'label.current_scenario': 'Текущий сценарий',
        'label.descriptor': 'Дескриптор',
        'label.download_html': '📥 Скачать HTML отчёт',
        'label.download_json': '📥 Скачать JSON',
        'label.download_md': '📥 Скачать Markdown',
        'label.effect': 'Влияние',
        'label.factor': 'Фактор',
        'label.interpretation': 'Интерпретация',
        'label.invalid_smiles': 'SMILES не распознан',
        'label.meaning': 'Значение',
        'label.no_warnings': 'Предупреждений нет.',
        'label.reasons': 'Причины',
        'label.unit': 'Ед.',
        'label.valid_smiles': 'SMILES распознан',
        'label.value': 'Значение',
        'label.zone': 'Зона',
        'msg.insufficient_what_if': 'Недостаточно дескрипторов для What-if симуляции.',
        'msg.pgp_vs_bbb': 'P-gp может снижать доступность для ЦНС даже при хорошей оценке прохождения через ГЭБ.',
        'msg.what_if_intro': 'Изменение ползунков является учебной симуляцией, а не прогнозом новой структуры.',
        'section.steps': '### 🧭 Пошаговое решение модели',
        'tabs.ml_explain': '🧠 ML-разбор',
        'section.ml_explain': '### 🧠 ML-разбор выбранных моделей',
        'ml.disclaimer': 'ML-разбор показывает, какие группы признаков использовала модель. Для структурных отпечатков BioSynth-EDU показывает агрегированные группы, потому '
                         'что отдельные технические биты трудно напрямую объяснять студентам.',
        'ml.model': 'Модель',
        'ml.status': 'Статус',
        'ml.method': 'Метод',
        'ml.probability': 'Оценка вероятности',
        'ml.class': 'Вывод модели',
        'ml.threshold': 'Порог',
        'ml.group_contributions': 'Групповые вклады признаков',
        'ml.top_features': 'Топ отдельных признаков',
        'ml.commentary': 'Учебный комментарий',
        'ml.unavailable': 'ML-разбор недоступен для этой молекулы или модели.',
        'ml.download_json': '📥 Скачать технический JSON',
        'ml.column.group': 'Группа признаков',
        'ml.column.value': 'Активные биты / значение',
        'ml.column.contribution': 'Вклад',
        'ml.column.abs_contribution': '|Вклад|',
        'ml.column.direction': 'Направление',
        'ml.column.feature': 'Признак'},
 'kk': {'language.select': 'Тіл / Язык / Language',
        'app.title': '🔬 Биожетімділік және ОЖЖ зерттеу модулі',
        'tabs.single': '🎯 Жеке талдау',
        'tabs.batch': '📂 Жаппай скрининг',
        'tabs.forecast': '📈 Болжам',
        'tabs.explain': '🧩 Модель шешімін талдау',
        'tabs.what_if': '🧪 What-if зертханасы',
        'tabs.report': '📄 Оқу есебі',
        'tabs.matrix': '🧬 Қан-ми тосқауылы × P-gp матрицасы',
        'tabs.methodology': '🔬 Әдістеме',
        'tabs.limitations': '⚠️ Модель шектеулері',
        'section.decision': '### 🧠 Модельдің оқу қорытындысы',
        'section.molecule': '### 🧬 Бастапқы молекула',
        'section.descriptors': '### 📊 Негізгі дескрипторлар',
        'section.factors': '### 🚦 Қан-ми тосқауылынан өтуді қолдайтын және шектейтін факторлар',
        'section.stepwise': '### 🧭 Модель шешімінің қадамдары',
        'section.matrix': '### 🧫 Қан-ми тосқауылы × P-gp матрицасы',
        'section.what_if': '### 🧪 What-if зертханасы: молекула қасиетін өзгерту',
        'section.report': '### 📄 Молекула бойынша оқу есебі',
        'section.methodology': '### 🔬 Әдістеме',
        'section.limitations': '### ⚠️ Шектеулер',
        'metric.bbb': 'Қан-ми тосқауылынан өту бағасы',
        'metric.pgp': 'P-gp бағасы',
        'metric.bbb_gupta': 'Қан-ми тосқауылы үшін Gupta көрсеткіші',
        'metric.pgp_class': 'P-gp статусы',
        'metric.pgp_probability': 'P-gp қаупінің бағасы',
        'metric.clint': 'Clint (метаболизм)',
        'metric.pka': 'pKa',
        'what_if.base_passive_bbb': 'Қан-ми тосқауылынан өтудің базалық бағасы',
        'what_if.base_cns': 'ОЖЖ қолжетімділігінің базалық бағасы',
        'what_if.passive_bbb': 'Қан-ми тосқауылынан өту',
        'what_if.educational_cns': 'ОЖЖ қолжетімділігінің оқу бағасы',
        'what_if.slider.mw': 'Молекулалық масса (MW), Да',
        'what_if.slider.logp': 'Липофильділік (LogP)',
        'what_if.slider.tpsa': 'Полярлық бет ауданы (TPSA), A^2',
        'what_if.slider.pka': 'Қышқылдық/негіздік (pKa)',
        'what_if.slider.hbd': 'H-байланыс донорлары (HBD)',
        'what_if.slider.hba': 'H-байланыс акцепторлары (HBA)',
        'what_if.slider.pgp': 'P-gp қаупінің бағасы',
        'valid.yes': '✅ SMILES танылды',
        'valid.no': '❌ SMILES танылмады',
        'warnings.none': 'Ескерту жоқ.',
        'warnings.reasons': 'Ескерту себептері',
        'applicability.title': '#### Қолданылу аймағы',
        'descriptor.column.name': 'Дескриптор',
        'descriptor.column.value': 'Мәні',
        'descriptor.column.unit': 'Бірлік',
        'descriptor.column.zone': 'Аймақ',
        'descriptor.column.effect': 'Әсері',
        'descriptor.column.meaning': 'Мағынасы',
        'descriptor.details': 'Толығырақ: әр дескриптордың түсіндірмесі',
        'factors.positive': 'Қан-ми тосқауылынан өтуді қолдайды',
        'factors.negative': 'Қан-ми тосқауылынан өтуге немесе ОЖЖ қолжетімділігіне кедергі жасайды',
        'factors.borderline': '🟡 Шекаралық',
        'factors.empty': 'Бұл топта айқын фактор жоқ.',
        'matrix.current': 'Ағымдағы сценарий',
        'matrix.pgp_expander': 'Неге P-gp қан-ми тосқауылынан жақсы өтуді әлсіретуі мүмкін?',
        'what_if.slider_title': '#### Дескриптор жүгірткілері',
        'what_if.result': '#### Оқу симуляциясының нәтижесі',
        'what_if.changed_factors': '#### Өзгерген факторлар',
        'what_if.commentary': '#### Оқу түсіндірмесі',
        'what_if.better': '##### Жақсарғаны',
        'what_if.worse': '##### Нашарлағаны',
        'what_if.no_improve': 'Оқу эвристикасы бойынша айқын жақсару жоқ.',
        'what_if.no_worse': 'Оқу эвристикасы бойынша айқын нашарлау жоқ.',
        'what_if.not_enough': 'What-if симуляциясы үшін дескрипторлар жеткіліксіз.',
        'report.info': 'Бұл есеп бастапқы молекуланы, негізгі дескрипторларды, оң/теріс факторларды, қан-ми тосқауылы × P-gp матрицасын, қадамдық '
                       'шешімді, әдістемені және шектеулерді біріктіреді.',
        'report.preview': 'Markdown есебін алдын ала көру',
        'report.download_html': '📥 HTML есебін жүктеу',
        'report.download_md': '📥 Markdown жүктеу',
        'report.download_json': '📥 JSON жүктеу',
        'batch.title': '### 📂 Batch explainability түйіні',
        'batch.info': 'Бұл блок жаппай скринингке әр молекула үшін қысқа оқу түсіндірмесін қосады.',
        'batch.total': 'Барлық молекулалар',
        'batch.valid': 'Валидті',
        'batch.invalid': 'Қате / invalid',
        'batch.candidates': 'ОЖЖ кандидаттары',
        'batch.summary': '#### Батчтың қысқа түйіні',
        'batch.table': 'Оқу басымдығы бойынша сұрыпталған батч кестесі',
        'sidebar.language': 'Тіл / Язык / Language',
        'sidebar.subtitle': '**Түсіндірілетін ADMET/BBB + P-gp**',
        'sidebar.info': 'pKa + P-gp + Gupta формуласы + Clint + XAI',
        'single.input_header': '1. Кіріс деректері',
        'single.smiles_label': 'SMILES:',
        'forecast.structure_title': '#### 📐 Молекуланың 2D-графы',
        'forecast.descriptors_title': '#### 🧪 Толық дескрипторлар',
        'forecast.integral_title': '#### 🧠 Интегралды болжам (ADMET)',
        'forecast.visualization_title': '#### 📊 Визуализация',
        'forecast.descriptor_col': 'Дескриптор',
        'forecast.value_col': 'Мәні',
        'metric.caco2': 'Caco-2 (өткізгіштік)',
        'metric.catmos': 'CATMoS (LD50)',
        'help.caco2': 'LogPapp өткізгіштік коэффициентінің логарифмін болжау. LogPapp > -5.0: асқазан-ішек жолында жақсы сіңу. LogPapp < '
                      '-6.0: нашар сіңу.',
        'help.bbb_gupta': 'Қан-ми тосқауылынан (BBB) өтуін бағалауға арналған Gupta формуласы бойынша есептік көрсеткіш.',
        'help.pgp_class': '1-класс — P-gp арқылы белсенді шығарылу қаупі жоғары.',
        'help.pgp_probability': 'P-gp арқылы белсенді шығарылу қаупінің бағасы. Мән жоғары болған сайын бұл қауіп жоғарырақ.',
        'help.pka': 'Динамикалық pKa гибридті BBB индексінің бір бөлігі ретінде қолданылады.',
        'help.clint': 'Бауыр клиренсі / метаболикалық тұрақтылық болжамы.',
        'help.catmos': 'Жедел уыттылықтың in silico болжамы.',
        'status.yes': 'ИӘ',
        'status.no': 'ЖОҚ',
        'status.high': 'Жоғары',
        'status.medium': 'Орташа',
        'status.low': 'Төмен',
        'status.high_risk': 'Жоғары қауіп',
        'status.stable': 'Тұрақты',
        'status.error': 'Қате',
        'status.na': 'N/A',
        'status.not_substrate': 'Айқын P-gp қаупі байқалмайды',
        'status.substrate_efflux': 'P-gp арқылы белсенді шығарылу мүмкін',
        'batch.page_title': '📂 Жаппай скрининг + Explainable ADMET',
        'batch.intro_stage5': 'Жаппай кесте әр молекула үшін қысқа оқу түсіндірмесімен толықтырылады: ОЖЖ үшін қорытынды класс, қан-ми тосқауылы × P-gp '
                              'сценарийі, оң/теріс факторлар, ескертулер және белгісіздік деңгейі.',
        'batch.source_label': 'Дереккөзі:',
        'batch.source_file': 'Файл (CSV/Excel)',
        'batch.source_text': 'Мәтін өрісі (SMILES)',
        'batch.file_label': 'Файл:',
        'batch.smiles_col_label': 'SMILES бағаны:',
        'batch.text_area_label': 'SMILES (әр жолға біреуден):',
        'batch.include_long_text': 'Batch export-қа ұзын түсіндірмелерді қосу',
        'batch.include_long_text_help': 'Үлкен тізімдер үшін өшірулі қалдырған дұрыс: Excel ықшам болады.',
        'batch.loaded_rows': 'Жүктелген жолдар: {n}',
        'batch.run_button': '🚀 Жаппай Explainable ADMET есебін іске қосу',
        'batch.progress': 'Есептеу: {progress}',
        'batch.done': 'Жаппай Explainable ADMET есебі аяқталды.',
        'batch.invalid_smiles_error': 'SMILES құрылымы дұрыс емес',
        'batch.calculation_error': 'Есептеу қатесі: {error_type}: {error}',
        'batch.tab_summary': '📊 Explainability түйіні',
        'batch.tab_full_table': '🧪 Толық ADMET + XAI кестесі',
        'batch.tab_export': '📥 Экспорт',
        'batch.show_all_columns': 'Барлық batch-бағандарды көрсету',
        'batch.download_excel': '📥 Excel жүктеу: ADMET + Explainability',
        'batch.download_csv': '📥 CSV жүктеу: XAI кестесі',
        'batch.empty': 'Batch кестесі бос.',
        'batch.result_empty': 'Batch нәтижесі бос.',
        'what_if.base': 'База',
        'what_if.col.factor': 'Фактор',
        'what_if.col.before': 'Бұрын',
        'what_if.col.after': 'Кейін',
        'what_if.col.contribution': 'Үлес',
        'what_if.col.descriptor': 'Дескриптор',
        'what_if.col.from': 'Қай аймақтан',
        'what_if.col.to': 'Қай аймаққа',
        'matrix.col.scenario': 'Сценарий',
        'matrix.col.interpretation': 'Интерпретация',
        'matrix.col.current': 'Ағымдағы',
        'msg.structure_unavailable': '2D-құрылым қолжетімсіз.',
        'msg.stepwise_unavailable': 'Қадамдық шешім қолжетімсіз.',
        'label.input_smiles': 'Input SMILES',
        'label.canonical_smiles': 'Canonical SMILES',
        'label.validity': 'Валидтілік',
        'label.step': 'Қадам',
        'effect.unknown': 'қолжетімсіз',
        'label.after': 'Кейін',
        'label.applicability_domain': 'Қолданылу домені',
        'label.before': 'Бұрын',
        'label.change': 'Өзгеріс',
        'label.current_scenario': 'Ағымдағы сценарий',
        'label.descriptor': 'Дескриптор',
        'label.download_html': '📥 HTML есепті жүктеу',
        'label.download_json': '📥 JSON жүктеу',
        'label.download_md': '📥 Markdown жүктеу',
        'label.effect': 'Әсері',
        'label.factor': 'Фактор',
        'label.interpretation': 'Түсіндірме',
        'label.invalid_smiles': 'SMILES танылмады',
        'label.meaning': 'Мәні',
        'label.no_warnings': 'Ескерту жоқ.',
        'label.reasons': 'Себептер',
        'label.unit': 'Бірлік',
        'label.valid_smiles': 'SMILES танылды',
        'label.value': 'Мәні',
        'label.zone': 'Аймақ',
        'msg.insufficient_what_if': 'What-if симуляциясы үшін дескрипторлар жеткіліксіз.',
        'msg.pgp_vs_bbb': 'P-gp қан-ми тосқауылынан өту қолайлы болса да ОЖЖ қолжетімділігін төмендетуі мүмкін.',
        'msg.what_if_intro': 'Жүгірткілерді өзгерту - жаңа құрылымның болжамы емес, оқу симуляциясы.',
        'section.steps': '### 🧭 Модель шешімінің қадамдары',
        'tabs.ml_explain': '🧠 ML-талдау',
        'section.ml_explain': '### 🧠 Таңдалған модельдердің ML-талдауы',
        'ml.disclaimer': 'ML-талдау модель қандай белгі топтарын қолданғанын көрсетеді. Құрылымдық отпечатоктар үшін BioSynth-EDU агрегатталған топтарды көрсетеді, '
                         'себебі жеке техникалық биттерді студентке тікелей түсіндіру қиын.',
        'ml.model': 'Модель',
        'ml.status': 'Статус',
        'ml.method': 'Әдіс',
        'ml.probability': 'Ықтималдық бағасы',
        'ml.class': 'Модель қорытындысы',
        'ml.threshold': 'Шек',
        'ml.group_contributions': 'Белгі топтарының үлестері',
        'ml.top_features': 'Жеке белгілердің топ тізімі',
        'ml.commentary': 'Оқу түсіндірмесі',
        'ml.unavailable': 'Бұл молекула немесе модель үшін ML-талдау қолжетімсіз.',
        'ml.download_json': '📥 Техникалық JSON жүктеу',
        'ml.column.group': 'Белгі тобы',
        'ml.column.value': 'Белсенді биттер / мән',
        'ml.column.contribution': 'Үлес',
        'ml.column.abs_contribution': '|Үлес|',
        'ml.column.direction': 'Бағыт',
        'ml.column.feature': 'Белгі'},
 'en': {'language.select': 'Language',
        'app.title': '🔬 Bioavailability & CNS research module',
        'tabs.single': '🎯 Single analysis',
        'tabs.batch': '📂 Batch screening',
        'tabs.forecast': '📈 Prediction',
        'tabs.explain': '🧩 Model decision breakdown',
        'tabs.what_if': '🧪 What-if lab',
        'tabs.report': '📄 Student report',
        'tabs.matrix': '🧬 BBB × P-gp matrix',
        'tabs.methodology': '🔬 Methodology',
        'tabs.limitations': '⚠️ Model limitations',
        'section.decision': '### 🧠 Educational model conclusion',
        'section.molecule': '### 🧬 Input molecule',
        'section.descriptors': '### 📊 Key descriptors',
        'section.factors': '### 🚦 Factors for and against BBB permeability',
        'section.stepwise': '### 🧭 Stepwise model trace',
        'section.matrix': '### 🧫 BBB × P-gp matrix',
        'section.what_if': '### 🧪 What-if lab: change a molecular property',
        'section.report': '### 📄 Student report for the molecule',
        'section.methodology': '### 🔬 Methodology',
        'section.limitations': '### ⚠️ Limitations',
        'metric.bbb': 'BBB passage indicator',
        'metric.pgp': 'P-gp probability',
        'metric.bbb_gupta': 'Gupta indicator for BBB',
        'metric.pgp_class': 'P-gp status',
        'metric.pgp_probability': 'P-gp risk estimate',
        'metric.clint': 'Clint (metabolism)',
        'metric.pka': 'pKa',
        'what_if.base_passive_bbb': 'Base passive BBB score',
        'what_if.base_cns': 'Base CNS score',
        'what_if.passive_bbb': 'Passive BBB score',
        'what_if.educational_cns': 'Educational CNS score',
        'what_if.slider.mw': 'Molecular weight (MW), Da',
        'what_if.slider.logp': 'Lipophilicity (LogP)',
        'what_if.slider.tpsa': 'Polar surface area (TPSA), A^2',
        'what_if.slider.pka': 'Acidity/basicity (pKa)',
        'what_if.slider.hbd': 'H-bond donors (HBD)',
        'what_if.slider.hba': 'H-bond acceptors (HBA)',
        'what_if.slider.pgp': 'P-gp risk estimate',
        'valid.yes': '✅ SMILES parsed',
        'valid.no': '❌ SMILES not parsed',
        'warnings.none': 'No warnings.',
        'warnings.reasons': 'Warning reasons',
        'applicability.title': '#### Applicability domain',
        'descriptor.column.name': 'Descriptor',
        'descriptor.column.value': 'Value',
        'descriptor.column.unit': 'Unit',
        'descriptor.column.zone': 'Zone',
        'descriptor.column.effect': 'Effect',
        'descriptor.column.meaning': 'Meaning',
        'descriptor.details': 'Details: explanation of each descriptor',
        'factors.positive': '🟢 Supports BBB+',
        'factors.negative': '🔴 Opposes BBB/CNS',
        'factors.borderline': '🟡 Borderline',
        'factors.empty': 'No clear factors in this group.',
        'matrix.current': 'Current scenario',
        'matrix.pgp_expander': 'Why can P-gp offset good BBB permeability?',
        'what_if.slider_title': '#### Descriptor sliders',
        'what_if.result': '#### Educational simulation result',
        'what_if.changed_factors': '#### Changed factors',
        'what_if.commentary': '#### Educational commentary',
        'what_if.better': '##### Improved',
        'what_if.worse': '##### Worsened',
        'what_if.no_improve': 'No clear improvement under the educational heuristic.',
        'what_if.no_worse': 'No clear worsening under the educational heuristic.',
        'what_if.not_enough': 'Not enough descriptors for the What-if simulation.',
        'report.info': 'This report combines the input molecule, key descriptors, positive and negative factors, the BBB × P-gp matrix, '
                       'stepwise trace, methodology and limitations.',
        'report.preview': 'Markdown report preview',
        'report.download_html': '📥 Download HTML report',
        'report.download_md': '📥 Download Markdown',
        'report.download_json': '📥 Download JSON',
        'batch.title': '### 📂 Batch explainability summary',
        'batch.info': 'This block adds a compact educational explanation for each molecule in batch screening.',
        'batch.total': 'Total molecules',
        'batch.valid': 'Valid',
        'batch.invalid': 'Errors / invalid',
        'batch.candidates': 'CNS candidates',
        'batch.summary': '#### Batch summary',
        'batch.table': 'Batch table sorted by educational priority',
        'sidebar.language': 'Language',
        'sidebar.subtitle': '**Explainable ADMET/BBB + P-gp**',
        'sidebar.info': 'pKa + P-gp + Gupta formula + Clint + XAI',
        'single.input_header': '1. Input data',
        'single.smiles_label': 'SMILES:',
        'forecast.structure_title': '#### 📐 2D molecular graph',
        'forecast.descriptors_title': '#### 🧪 Full descriptors',
        'forecast.integral_title': '#### 🧠 Integrated prediction (ADMET)',
        'forecast.visualization_title': '#### 📊 Visualizations',
        'forecast.descriptor_col': 'Descriptor',
        'forecast.value_col': 'Value',
        'metric.caco2': 'Caco-2 (permeability)',
        'metric.catmos': 'CATMoS (LD50)',
        'help.caco2': 'Predicted log permeability coefficient LogPapp. LogPapp > -5.0: good intestinal absorption. LogPapp < -6.0: poor '
                      'absorption.',
        'help.bbb_gupta': 'Updated Gupta model: dynamic pKa replaces the fixed value. Threshold in the current app: >= 3.0.',
        'help.pgp_class': 'Class 1 means high efflux risk.',
        'help.pgp_probability': 'P-gp classifier score. A high value indicates substrate/efflux risk.',
        'help.pka': 'Dynamic pKa is used as part of the hybrid BBB index.',
        'help.clint': 'Predicted hepatic clearance / metabolic stability.',
        'help.catmos': 'In silico acute toxicity prediction.',
        'status.yes': 'YES',
        'status.no': 'NO',
        'status.high': 'High',
        'status.medium': 'Medium',
        'status.low': 'Low',
        'status.high_risk': 'High risk',
        'status.stable': 'Stable',
        'status.error': 'Error',
        'status.na': 'N/A',
        'status.not_substrate': 'Non-substrate',
        'status.substrate_efflux': 'Substrate (efflux)',
        'batch.page_title': '📂 Batch screening + Explainable ADMET',
        'batch.intro_stage5': 'The batch table is extended with a compact educational explanation for each molecule: final CNS class, BBB '
                              '× P-gp scenario, supporting/opposing factors, warnings and uncertainty level.',
        'batch.source_label': 'Source:',
        'batch.source_file': 'File (CSV/Excel)',
        'batch.source_text': 'Text field (SMILES)',
        'batch.file_label': 'File:',
        'batch.smiles_col_label': 'SMILES column:',
        'batch.text_area_label': 'SMILES (one per line):',
        'batch.include_long_text': 'Add long explanations to batch export',
        'batch.include_long_text_help': 'For large lists it is better to keep this off: the Excel file will be more compact.',
        'batch.loaded_rows': 'Loaded rows: {n}',
        'batch.run_button': '🚀 Run batch Explainable ADMET',
        'batch.progress': 'Calculation: {progress}',
        'batch.done': 'Batch Explainable ADMET calculation completed.',
        'batch.invalid_smiles_error': 'Invalid SMILES structure',
        'batch.calculation_error': 'Calculation error: {error_type}: {error}',
        'batch.tab_summary': '📊 Explainability summary',
        'batch.tab_full_table': '🧪 Full ADMET + XAI table',
        'batch.tab_export': '📥 Export',
        'batch.show_all_columns': 'Show all batch columns',
        'batch.download_excel': '📥 Download Excel: ADMET + Explainability',
        'batch.download_csv': '📥 Download CSV: XAI table',
        'batch.empty': 'Batch table is empty.',
        'batch.result_empty': 'Batch result is empty.',
        'what_if.base': 'Base',
        'what_if.col.factor': 'Factor',
        'what_if.col.before': 'Before',
        'what_if.col.after': 'After',
        'what_if.col.contribution': 'Contribution',
        'what_if.col.descriptor': 'Descriptor',
        'what_if.col.from': 'From',
        'what_if.col.to': 'To',
        'matrix.col.scenario': 'Scenario',
        'matrix.col.interpretation': 'Interpretation',
        'matrix.col.current': 'Current',
        'msg.structure_unavailable': '2D structure unavailable.',
        'msg.stepwise_unavailable': 'Stepwise trace unavailable.',
        'label.input_smiles': 'Input SMILES',
        'label.canonical_smiles': 'Canonical SMILES',
        'label.validity': 'Validity',
        'label.step': 'Step',
        'effect.unknown': 'not available',
        'label.after': 'After',
        'label.applicability_domain': 'Applicability domain',
        'label.before': 'Before',
        'label.change': 'Change',
        'label.current_scenario': 'Current scenario',
        'label.descriptor': 'Descriptor',
        'label.download_html': '📥 Download HTML report',
        'label.download_json': '📥 Download JSON',
        'label.download_md': '📥 Download Markdown',
        'label.effect': 'Effect',
        'label.factor': 'Factor',
        'label.interpretation': 'Interpretation',
        'label.invalid_smiles': 'SMILES not parsed',
        'label.meaning': 'Meaning',
        'label.no_warnings': 'No warnings.',
        'label.reasons': 'Reasons',
        'label.unit': 'Unit',
        'label.valid_smiles': 'SMILES parsed',
        'label.value': 'Value',
        'label.zone': 'Zone',
        'msg.insufficient_what_if': 'Not enough descriptors for the What-if simulation.',
        'msg.pgp_vs_bbb': 'P-gp may reduce CNS exposure even when passive BBB properties look favourable.',
        'msg.what_if_intro': 'Changing sliders is an educational simulation, not a prediction for a new structure.',
        'section.steps': '### 🧭 Stepwise model trace',
        'tabs.ml_explain': '🧠 ML / SHAP breakdown',
        'section.ml_explain': '### 🧠 ML / SHAP breakdown for selected models',
        'ml.disclaimer': 'SHAP is an advanced ML explanation layer. For fingerprint features, BioSynth-EDU shows aggregated groups because '
                         'individual Morgan/MACCS bits are difficult to explain directly to students.',
        'ml.model': 'Model',
        'ml.status': 'Status',
        'ml.method': 'Method',
        'ml.probability': 'Probability / score',
        'ml.class': 'Class',
        'ml.threshold': 'Threshold',
        'ml.group_contributions': 'Feature-group contributions',
        'ml.top_features': 'Top individual features',
        'ml.commentary': 'Educational commentary',
        'ml.unavailable': 'ML explanation is unavailable for this molecule or model.',
        'ml.download_json': '📥 Download ML explanation JSON',
        'ml.column.group': 'Feature group',
        'ml.column.value': 'Active bits / value',
        'ml.column.contribution': 'Contribution',
        'ml.column.abs_contribution': '|Contribution|',
        'ml.column.direction': 'Direction',
        'ml.column.feature': 'Feature'}}

DESCRIPTOR_BASE = {
    "ru": BASE_DESCRIPTOR_EXPLANATIONS_RU,
    "kk": {
        "MW": "Молекулалық масса молекуланың өлшемін көрсетеді. Үлкен молекулалар, әсіресе полярлығы және сутектік байланыс донорлары мен акцепторлары көп болса, биологиялық бөгеттерден қиынырақ өтеді.",
        "LogP": "LogP молекуланың липофильдігін көрсетеді. Қан-ми тосқауылынан өту үшін молекула мембранаға ене алатындай липофильді болуы керек, бірақ ерігіштікті жоғалтатындай тым липофильді болмауы тиіс.",
        "TPSA": "TPSA молекуланың полярлық беткейін сипаттайды. TPSA жоғары болған сайын молекула сумен күштірек әрекеттеседі және липидті қабаттан өтуі қиындайды.",
        "HBD": "HBD сутектік байланыс донорларының санын көрсетеді. Донорлар көп болса, молекула сумен күштірек байланысып, пассивті диффузия қиындауы мүмкін.",
        "HBA": "HBA сутектік байланыс акцепторларының санын көрсетеді. Акцепторлар көп болса, полярлық өсіп, қан-ми тосқауылынан пассивті өту төмендеуі мүмкін.",
        "RotatableBonds": "Айналмалы байланыстар молекуланың икемділігін көрсетеді. Өте икемді молекулалардың ADMET профилі жиі күрделірек болады.",
        "AromaticRings": "Ароматты сақиналар липофильділік пен ақуыздармен байланысуды арттыра алады. Орташа саны қолайлы, ал артық саны ерігіштікті нашарлатуы мүмкін.",
        "pKa_pred": "pKa физиологиялық pH-та молекуланың иондану күйін бағалауға көмектеседі. Қатты иондалған түрлер липидті мембранадан нашар өтеді.",
        "FormalCharge": "Формальды заряд молекуланың толық оң немесе теріс заряд алып тұрғанын көрсетеді. Бейтарап түр қан-ми тосқауылынан өту үшін әдетте қолайлырақ.",
        "GasteigerMin": "Гастейгер жартылай зарядтары электрондық тығыздықтың таралуын көрсетеді және полярлық аймақтарды бағалауға көмектеседі.",
        "GasteigerMax": "Гастейгер жартылай зарядтары электрондық тығыздықтың таралуын көрсетеді және полярлық аймақтарды бағалауға көмектеседі.",
        "GasteigerAbsMax": "Максималды абсолютті Гастейгер заряды жергілікті полярлықтың айқындылығын көрсетеді.",
        "BBB_probability": "Қан-ми тосқауылынан өту бағасы молекуланың осы тосқауылдан өте алатын қосылыстарға ұқсастығын көрсетеді. Бұл эксперименттік ықтималдық емес.",
        "Pgp_probability": "P-gp — молекуланы жасушадан қанға қайта шығара алатын тасымалдаушы. Ол ОЖЖ қолжетімділігін төмендетуі мүмкін.",
        "Clint_risk": "Clint ішкі клиренс пен метаболизм қаупін сипаттайды. Бұл қан-ми тосқауылынан өтуді тікелей анықтамайды, бірақ ADMET профилін толықтырады.",
        "CATMoS_LD50": "CATMoS / LD50 токсикологиялық бағалауға жатады және BBB түсіндірмесін толықтыратын бөлек ADMET көрсеткіші болып саналады.",
    },
    "en": {
        "MW": "Molecular weight describes molecular size. Larger molecules usually cross biological barriers less easily, especially when high size is combined with high polarity, many HBD/HBA atoms or high flexibility.",
        "LogP": "LogP describes molecular lipophilicity. A BBB-penetrant molecule should be lipophilic enough to enter membranes, but not so lipophilic that it loses solubility or binds excessively to proteins.",
        "TPSA": "TPSA reflects polar surface area. Higher TPSA means stronger interaction with water and usually weaker passive diffusion through the lipid barrier.",
        "HBD": "HBD is the number of hydrogen-bond donors. More donors strengthen water interactions and can reduce passive BBB diffusion.",
        "HBA": "HBA is the number of hydrogen-bond acceptors. Too many acceptors increase polarity and can reduce passive BBB permeability.",
        "RotatableBonds": "Rotatable bonds describe molecular flexibility. Highly flexible molecules often have a less predictable ADMET profile.",
        "AromaticRings": "Aromatic rings can increase lipophilicity and protein binding. A moderate number is often acceptable, while excessive aromaticity may reduce solubility.",
        "pKa_pred": "pKa helps estimate ionisation at physiological pH. Strongly ionised forms usually cross lipid membranes less efficiently.",
        "FormalCharge": "Formal charge indicates whether the molecule carries a full positive or negative charge. Neutral forms are usually more favourable for passive BBB diffusion.",
        "GasteigerMin": "Gasteiger partial charges describe the distribution of electronic density and help identify polar regions.",
        "GasteigerMax": "Gasteiger partial charges describe the distribution of electronic density and help identify polar regions.",
        "GasteigerAbsMax": "Maximum absolute Gasteiger charge helps identify pronounced local polarity.",
        "BBB_probability": "BBB probability is a model score describing similarity to BBB+ compounds in the model space. It is not an experimental probability.",
        "Pgp_probability": "P-gp is an efflux transporter. It may actively remove the molecule back to blood and reduce CNS exposure even when passive diffusion looks favourable.",
        "Clint_risk": "Clint describes intrinsic clearance or metabolic stability risk. It complements BBB analysis but does not replace it.",
        "CATMoS_LD50": "CATMoS / LD50 is a toxicology estimate. It complements the ADMET profile but is not a direct explanation of BBB permeability.",
    },
}

FINAL_DECISIONS = {
    "ru": FINAL_DECISION_TEXTS,
    "kk": {
        "likely_cns_active": {"title": "ОЖЖ үшін қолайлы профиль болуы мүмкін", "final_label_ru": "ОЖЖ үшін қолайлы профиль болуы мүмкін", "summary": "Модель қан-ми тосқауылынан пассивті өтуді қолайлы, ал P-gp арқылы белсенді шығарылу қаупін төмен деп бағалайды.", "student_interpretation": "Бұл профиль ОЖЖ қолжетімділігі туралы гипотезаны қолдайды, бірақ эксперименттік дәлел емес."},
        "peripheral_action_risk": {"title": "ОЖЖ қолжетімділігі төмендеуі мүмкін", "final_label_ru": "ОЖЖ қолжетімділігі төмендеуі мүмкін", "summary": "Қан-ми тосқауылынан пассивті өту қолайлы көрінеді, бірақ P-gp арқылы белсенді шығарылу қаупі жоғары.", "student_interpretation": "Молекула тосқауыл аймағына өтуі мүмкін, бірақ P-gp арқылы қайта шығарылуы ықтимал."},
        "likely_not_bbb_penetrant": {"title": "Қан-ми тосқауылынан өтуі әлсіз болуы мүмкін", "final_label_ru": "Қан-ми тосқауылынан өтуі әлсіз болуы мүмкін", "summary": "P-gp негізгі шектеу емес, бірақ физика-химиялық профиль қан-ми тосқауылынан пассивті өтуді қолдамайды.", "student_interpretation": "Негізгі шектеу — пассивті диффузия үшін қолайсыз қасиеттер."},
        "full_barrier": {"title": "Екі шектеу бар", "final_label_ru": "Екі шектеу бар: қан-ми тосқауылынан өту әлсіз және P-gp қаупі жоғары", "summary": "Молекулада қан-ми тосқауылынан пассивті өту профилі қолайсыз және P-gp арқылы белсенді шығарылу қаупі жоғары.", "student_interpretation": "Бұл ОЖЖ қолжетімділігі үшін екі жақты шектеу."},
        "uncertain_or_borderline": {"title": "Анық емес / шекаралық", "final_label_ru": "Анық емес / шекаралық", "summary": "Бағалар шектерге жақын немесе модель блоктары ішінара қарама-қайшы сигнал береді.", "student_interpretation": "Мұндай нәтиже нақты қорытынды емес, әрі қарай талдау үшін себеп ретінде қарастырылуы керек."},
        "insufficient_data": {"title": "Қорытындыға дерек жеткіліксіз", "final_label_ru": "Дерек жеткіліксіз", "summary": "Қан-ми тосқауылы немесе P-gp бағасы жоқ.", "student_interpretation": "Қан-ми тосқауылы және P-gp модельдері сандық баға қайтарғанын тексеріңіз."},
    },
    "en": {
        "likely_cns_active": {"title": "Likely CNS-active profile", "final_label_ru": "Likely CNS-active profile", "summary": "The model sees favourable passive BBB permeability and low P-gp efflux risk.", "student_interpretation": "This profile supports a hypothesis of CNS exposure, but it is not experimental proof."},
        "peripheral_action_risk": {"title": "Likely peripheral action / reduced CNS exposure risk", "final_label_ru": "Efflux risk despite favourable BBB estimate", "summary": "Passive BBB permeability looks favourable, but P-gp efflux risk is high.", "student_interpretation": "The molecule may enter the barrier region but then be actively removed by P-gp."},
        "likely_not_bbb_penetrant": {"title": "Likely not BBB-penetrant", "final_label_ru": "Likely not BBB-penetrant", "summary": "P-gp does not look like the main limitation, but the physicochemical profile does not support passive BBB passage.", "student_interpretation": "The main limitation is an unfavourable physicochemical profile for passive diffusion."},
        "full_barrier": {"title": "Full barrier", "final_label_ru": "Full barrier: poor passive BBB plus P-gp efflux", "summary": "The molecule combines an unfavourable passive BBB profile with high P-gp efflux risk.", "student_interpretation": "This is a double limitation for CNS exposure."},
        "uncertain_or_borderline": {"title": "Uncertain / borderline", "final_label_ru": "Uncertain / borderline", "summary": "Scores are close to thresholds or model blocks provide partly conflicting signals.", "student_interpretation": "Use this result as a prompt for further analysis rather than a firm conclusion."},
        "insufficient_data": {"title": "Insufficient data for final interpretation", "final_label_ru": "Insufficient data", "summary": "BBB probability or P-gp probability is missing.", "student_interpretation": "Check that BBB and P-gp models returned numerical scores."},
    },
}

DISCLAIMERS = {
    "ru": {"in_silico": IN_SILICO_DISCLAIMER_RU, "what_if": WHAT_IF_DISCLAIMER_RU},
    "kk": {
        "in_silico": "BioSynth-EDU in silico болжам береді. Бұл медициналық ұсыныс емес және қан-ми тосқауылынан өту, уыттылық немесе тиімділік бойынша эксперименттік дәлел емес. Модель бағаларын абсолютті биологиялық шындық емес, есептік баға ретінде түсіндіру керек.",
        "what_if": "Бұл блок педагогикалық симуляция болып табылады. Сіз химиялық құрылымды емес, жеке дескрипторларды өзгертесіз; нәтиже жаңа молекулаға нақты болжам емес.",
    },
    "en": {
        "in_silico": "BioSynth-EDU provides an in silico prediction. It is not medical advice and not experimental evidence of BBB permeability, toxicity or efficacy. Model probabilities should be interpreted as model scores, not absolute biological truth.",
        "what_if": "This block is an educational simulation. You change descriptors, not the chemical structure; the result is not a prediction for a new molecule.",
    },
}

APPLICABILITY_MESSAGES = {
    "ru": {
        "inside": "Молекула не нарушает базовые учебные правила применимости.",
        "caution": "Молекула имеет предупреждения по домену применимости; прогноз полезен как учебная гипотеза, но не как твёрдый вывод.",
        "outside": "Молекула плохо похожа на структуры, на которых модель обычно работает надёжно; результат следует трактовать очень осторожно.",
        "invalid": "SMILES некорректен; модельный прогноз и объяснение недоступны или ненадёжны.",
    },
    "kk": {
        "inside": "Молекула негізгі оқу қолданылу ережелерін бұзбайды.",
        "caution": "Молекулада қолданылу аймағы бойынша ескертулер бар; болжам оқу гипотезасы ретінде пайдалы, бірақ нақты қорытынды емес.",
        "outside": "Молекулада модельдің қолданылу аймағынан шығу белгілері бар; нәтижені өте сақ түсіндіру керек.",
        "invalid": "SMILES қате; модельдік болжам мен түсіндірме қолжетімсіз немесе сенімсіз.",
    },
    "en": {
        "inside": "The molecule does not violate the basic educational applicability rules.",
        "caution": "The molecule has applicability-domain warnings; the prediction is useful as an educational hypothesis, not as a firm conclusion.",
        "outside": "The molecule shows signs of being outside the applicability domain; interpret the result very cautiously.",
        "invalid": "The SMILES is invalid; model prediction and explanation are unavailable or unreliable.",
    },
}

WARNING_MESSAGES = {
    "ru": {
        "invalid_smiles": "SMILES не удалось корректно распознать.",
        "multiple_fragments": "Молекула содержит несколько фрагментов; это может указывать на соль или смесь.",
        "salt_or_mixture": "Обнаружены признаки соли или смеси; прогноз может быть менее надёжен.",
        "no_carbon_or_inorganic": "Молекула не содержит углерод или похожа на неорганическую структуру; ADMET-модель может быть ненадёжной.",
        "very_large_molecule": "Очень большая молекулярная масса: молекула может выходить за область применимости модели.",
        "very_high_tpsa": "Очень высокая TPSA: молекула может выходить за типичную область применимости оценки ГЭБ, особенно для гликозидов и полифенолов.",
        "many_hbd": "Высокое число HBD: пассивное прохождение через ГЭБ может быть сильно ограничено.",
        "many_hba": "Высокое число HBA: профиль может быть слишком полярным для прохождения через ГЭБ.",
        "formal_charge_nonzero": "Формальный заряд отличен от 0; пассивное прохождение через ГЭБ нужно трактовать осторожно.",
        "high_abs_formal_charge": "Выраженный формальный заряд: прохождение через ГЭБ может быть менее надёжно оценено.",
        "extreme_logp_low": "Очень низкий LogP: молекула может быть слишком гидрофильной для пассивного прохождения BBB.",
        "extreme_logp_high": "Очень высокий LogP: возможны проблемы растворимости и неспецифического связывания.",
        "polyphenol_like": "Структура похожа на полифенольное природное соединение; модели BBB/P-gp могут быть менее надёжны.",
        "glycoside_like": "Обнаружены признаки гликозидного/сахарного фрагмента; молекула может плохо соответствовать структурам, на которых модель обычно работает надёжно.",
        "bbb_pgp_conflict": "Высокая оценка прохождения через ГЭБ (BBB) сочетается с высоким P-gp: пассивное прохождение может конфликтовать с активным выведением.",
    },
    "kk": {
        "invalid_smiles": "SMILES дұрыс танылмады.",
        "multiple_fragments": "Молекулада бірнеше фрагмент бар; бұл тұз немесе қоспа белгісі болуы мүмкін.",
        "salt_or_mixture": "Тұз немесе қоспа белгілері анықталды; болжам сенімділігі төмендеуі мүмкін.",
        "no_carbon_or_inorganic": "Молекулада көміртек жоқ немесе бейорганикалық құрылымға ұқсайды; ADMET моделі сенімсіз болуы мүмкін.",
        "very_large_molecule": "Молекулалық масса өте үлкен: молекула модельдің қолданылу аймағынан шығуы мүмкін.",
        "very_high_tpsa": "TPSA өте жоғары: гликозидтер мен полифенолдар үшін қан-ми тосқауылынан өтуді бағалаудың типтік қолданылу аймағынан шығуы мүмкін.",
        "many_hbd": "HBD саны жоғары: қан-ми тосқауылынан пассивті өту қатты шектелуі мүмкін.",
        "many_hba": "HBA саны жоғары: профиль қан-ми тосқауылынан өту үшін тым полярлы болуы мүмкін.",
        "formal_charge_nonzero": "Формальды заряд 0 емес; қан-ми тосқауылынан пассивті өтуді сақтықпен түсіндіру керек.",
        "high_abs_formal_charge": "Формальды заряд айқын; қан-ми тосқауылынан пассивті өту сенімсіз бағалануы мүмкін.",
        "extreme_logp_low": "LogP өте төмен: молекула қан-ми тосқауылынан пассивті өту үшін тым гидрофильді болуы мүмкін.",
        "extreme_logp_high": "LogP өте жоғары: ерігіштік және бейспецификалық байланысу мәселелері болуы мүмкін.",
        "polyphenol_like": "Құрылым полифенолдық табиғи қосылысқа ұқсайды; қан-ми тосқауылы/P-gp модельдері сенімсіз болуы мүмкін.",
        "glycoside_like": "Гликозидтік/қант фрагментінің белгілері бар; қолданылу аймағынан шығуы мүмкін.",
        "bbb_pgp_conflict": "Қан-ми тосқауылынан өту бағасы жоғары және P-gp қаупі жоғары: пассивті өту белсенді шығарылумен қайшы келуі мүмкін.",
    },
    "en": {
        "invalid_smiles": "The SMILES could not be parsed correctly.",
        "multiple_fragments": "The molecule contains multiple fragments; this may indicate a salt or mixture.",
        "salt_or_mixture": "Signs of a salt or mixture were detected; the prediction may be less reliable.",
        "no_carbon_or_inorganic": "The molecule has no carbon or looks inorganic; ADMET models may be unreliable.",
        "very_large_molecule": "Very high molecular weight: possible outside-domain molecule.",
        "very_high_tpsa": "Very high TPSA: possible outside the typical BBB domain, especially for glycosides and polyphenols.",
        "many_hbd": "High HBD count: passive BBB diffusion may be strongly limited.",
        "many_hba": "High HBA count: the profile may be too polar for BBB.",
        "formal_charge_nonzero": "Formal charge is not zero; passive BBB interpretation should be cautious.",
        "high_abs_formal_charge": "Pronounced formal charge: passive BBB permeability may be less reliably estimated.",
        "extreme_logp_low": "Very low LogP: the molecule may be too hydrophilic for passive BBB passage.",
        "extreme_logp_high": "Very high LogP: solubility and nonspecific binding may become problematic.",
        "polyphenol_like": "The structure looks polyphenol-like; BBB/P-gp models may be less reliable.",
        "glycoside_like": "A glycoside/sugar-like fragment is detected; the molecule may be outside the applicability domain.",
        "bbb_pgp_conflict": "Favourable BBB estimate combined with high P-gp: passive permeability may conflict with active efflux.",
    },
}

METHODOLOGY = {
    "ru": [
        {"title": "1. Проверка структуры", "text": "SMILES преобразуется в молекулярный граф. Если структуру не удаётся распознать, расчёт и объяснение не выполняются."},
        {"title": "2. Физико-химические дескрипторы", "text": "Для учебного объяснения используются молекулярная масса, LogP, TPSA, доноры и акцепторы водородных связей, pKa, заряд и оценка P-gp."},
        {"title": "3. Прохождение через ГЭБ", "text": "ГЭБ означает гематоэнцефалический барьер. Этот блок оценивает, насколько свойства молекулы совместимы с пассивным прохождением через такой барьер."},
        {"title": "4. P-gp", "text": "P-gp рассматривается как отдельный механизм активного выведения молекулы обратно в кровь. Он может снижать доступность для ЦНС даже при благоприятных свойствах для прохождения через ГЭБ."},
        {"title": "5. Учебное объяснение", "text": "Вывод строится как учебная интерпретация дескрипторов и модельных оценок. Он не заменяет экспериментальную проверку."},
    ],
    "kk": [
        {"title": "1. Құрылымды тексеру", "text": "SMILES молекулалық графқа түрлендіріледі. Құрылым танылмаса, есептеу және түсіндіру орындалмайды."},
        {"title": "2. Физика-химиялық дескрипторлар", "text": "Оқу түсіндірмесінде молекулалық масса, LogP, TPSA, сутектік байланыс донорлары мен акцепторлары, pKa, заряд және P-gp бағасы қолданылады."},
        {"title": "3. Қан-ми тосқауылынан өту", "text": "Бұл блок молекула қасиеттерінің қан-ми тосқауылынан пассивті өтуге қаншалықты сәйкес келетінін бағалайды."},
        {"title": "4. P-gp", "text": "P-gp молекуланы қайтадан қанға белсенді шығара алатын бөлек механизм ретінде қарастырылады. Сондықтан қан-ми тосқауылынан өту қолайлы болса да, ОЖЖ қолжетімділігі төмендеуі мүмкін."},
        {"title": "5. Оқу түсіндірмесі", "text": "Қорытынды дескрипторлар мен модельдік бағалардың оқу интерпретациясы ретінде беріледі. Ол эксперименттік тексеруді алмастырмайды."},
    ],
    "en": [
        {"title": "1. Structure validation", "text": "The SMILES is first converted into a molecular graph. If RDKit cannot parse the structure, calculation and explanation are not performed."},
        {"title": "2. Physicochemical descriptors", "text": "The teaching explanation uses molecular weight, LogP, TPSA, H-bond donors and acceptors, pKa, charge and the P-gp estimate."},
        {"title": "3. BBB passage", "text": "BBB means blood-brain barrier. This block estimates whether molecular properties are compatible with passive passage across that barrier."},
        {"title": "4. P-gp", "text": "P-gp is treated as a separate active efflux mechanism that can return a molecule to blood and reduce CNS exposure."},
        {"title": "5. Teaching explanation", "text": "The conclusion is an educational interpretation of descriptors and model estimates. It does not replace experimental validation."},
    ],
}

LIMITATIONS = {
    "ru": [
        "BioSynth-EDU даёт расчётный прогноз, а не медицинскую рекомендацию.",
        "Прогноз не является экспериментальным доказательством прохождения через ГЭБ, токсичности или эффективности.",
        "Вероятности и показатели моделей следует читать как расчётные оценки, а не как абсолютную биологическую истину.",
        "What-if режим меняет дескрипторы без изменения химической структуры, поэтому это учебная симуляция.",
        "Для солей, смесей, полифенолов, гликозидов и очень крупных молекул важно проверять, похожа ли молекула на те структуры, на которых модель обычно работает надёжно.",
    ],
    "kk": [
        "BioSynth-EDU есептік болжам береді, медициналық ұсыныс емес.",
        "Болжам қан-ми тосқауылынан өту, уыттылық немесе тиімділік бойынша эксперименттік дәлел емес.",
        "Модель ықтималдықтары мен көрсеткіштерін абсолютті биологиялық шындық емес, есептік бағалар ретінде түсіндіру керек.",
        "What-if режимі химиялық құрылымды өзгертпей дескрипторларды өзгертеді, сондықтан бұл оқу симуляциясы.",
        "Тұздар, қоспалар, полифенолдар, гликозидтер және өте ірі молекулалар үшін молекула модель сенімді жұмыс істейтін құрылымдарға ұқсай ма, соны бөлек тексеру маңызды.",
    ],
    "en": [
        "BioSynth-EDU provides an in silico prediction, not medical advice.",
        "The prediction is not experimental proof of BBB permeability, toxicity or efficacy.",
        "Model probabilities and scores should be interpreted as model estimates, not absolute biological truth.",
        "The What-if mode changes descriptors without changing the structure, so it is an educational simulation.",
        "Applicability-domain warnings are important for salts, mixtures, polyphenols, glycosides and very large molecules.",
    ],
}

STUDENT_QUESTIONS = {
    "ru": ["Какие дескрипторы сильнее всего поддерживают прохождение через ГЭБ?", "Какой фактор сильнее всего ограничивает доступность для ЦНС?", "Есть ли конфликт между пассивным прохождением через ГЭБ и активным выведением через P-gp?", "Как изменился бы учебный вывод, если TPSA или оценка P-gp стали выше?", "Какие предупреждения о надёжности модели нужно учесть?"],
    "kk": ["Қан-ми тосқауылынан өтуді ең көп қолдайтын дескрипторлар қайсы?", "ОЖЖ қолжетімділігін ең көп шектейтін фактор қандай?", "Қан-ми тосқауылынан пассивті өту мен P-gp арқылы белсенді шығарылу арасында қайшылық бар ма?", "TPSA немесе P-gp бағасы жоғарыласа, оқу қорытындысы қалай өзгерер еді?", "Модель сенімділігі туралы қандай ескертулерді ескеру керек?"],
    "en": ["Which two descriptors most strongly support BBB permeability?", "What is the main limitation for CNS exposure?", "Is there a conflict between passive BBB permeability and P-gp efflux?", "How would the educational conclusion change if TPSA or P-gp probability increased?", "Which applicability-domain warnings should be considered?"],
}

BATCH_PRIORITY_LABELS = {
    "ru": {
        "CNS candidate": "Кандидат для ЦНС",
        "CNS candidate / caution": "Кандидат для ЦНС / осторожно",
        "Efflux risk": "Риск активного выведения",
        "Review": "Проверить вручную",
        "Low passive BBB": "Слабое прохождение через ГЭБ",
        "Full barrier": "Два ограничения",
        "Outside domain": "Вне области применимости",
        "Invalid/error": "Ошибка / некорректные данные",
    },
    "kk": {
        "CNS candidate": "ОЖЖ кандидаты",
        "CNS candidate / caution": "ОЖЖ кандидаты / сақтық",
        "Efflux risk": "Белсенді шығарылу қаупі",
        "Review": "Қолмен тексеру",
        "Low passive BBB": "Қан-ми тосқауылынан өту әлсіз",
        "Full barrier": "Екі шектеу бар",
        "Outside domain": "Қолданылу аймағынан тыс",
        "Invalid/error": "Қате / жарамсыз дерек",
    },
    "en": {
        "CNS candidate": "CNS candidate",
        "CNS candidate / caution": "CNS candidate / caution",
        "Efflux risk": "Efflux risk",
        "Review": "Review",
        "Low passive BBB": "Low passive BBB",
        "Full barrier": "Full barrier",
        "Outside domain": "Outside domain",
        "Invalid/error": "Invalid/error",
    },
}

BATCH_SUMMARY_TEMPLATES = {
    "ru": "Обработано молекул: {total}. Валидных: {valid}. Ошибок: {invalid}. Кандидатов для ЦНС: {cns}. Конфликт ГЭБ/P-gp: {efflux}. Пограничных: {borderline}.",
    "kk": "Өңделген молекулалар: {total}. Валидті: {valid}. Қате: {invalid}. ОЖЖ кандидаттары: {cns}. Қан-ми тосқауылы/P-gp қайшылығы: {efflux}. Шекаралық: {borderline}.",
    "en": "Processed molecules: {total}. Valid: {valid}. Errors: {invalid}. CNS candidates: {cns}. BBB/P-gp conflict: {efflux}. Borderline: {borderline}.",
}

WHAT_IF_TEXTS = {
    "ru": {
        "passive_up": "Оценка прохождения через ГЭБ в симуляции улучшилась.",
        "passive_down": "Оценка прохождения через ГЭБ в симуляции снизилась.",
        "passive_same": "Оценка прохождения через ГЭБ изменилась незначительно.",
        "pgp_up": "Рост оценки P-gp снижает учебную оценку доступности для ЦНС, даже если прохождение через ГЭБ остаётся неплохим.",
        "pgp_down": "Снижение оценки P-gp уменьшает учебный риск активного выведения и может повысить оценку доступности для ЦНС.",
        "cns_up": "Итоговая оценка доступности для ЦНС выросла: профиль стал более благоприятным.",
        "cns_down": "Итоговая оценка доступности для ЦНС снизилась: профиль стал менее благоприятным.",
        "same": "Зоны и итоговые оценки почти не изменились; сдвиг остался внутри той же учебной зоны.",
    },
    "kk": {
        "passive_up": "Симуляцияда қан-ми тосқауылынан өту бағасы жақсарды.",
        "passive_down": "Симуляцияда қан-ми тосқауылынан өту бағасы төмендеді.",
        "passive_same": "Қан-ми тосқауылынан өту бағасы аз ғана өзгерді.",
        "pgp_up": "P-gp бағасының өсуі қан-ми тосқауылынан өту бағасы жақсы болса да ОЖЖ қолжетімділігінің оқу бағасын төмендетеді.",
        "pgp_down": "P-gp бағасының төмендеуі белсенді шығарылу қаупін азайтып, ОЖЖ қолжетімділігінің бағасын арттыруы мүмкін.",
        "cns_up": "ОЖЖ қолжетімділігінің қорытынды бағасы өсті: профиль қолайлырақ болды.",
        "cns_down": "ОЖЖ қолжетімділігінің қорытынды бағасы төмендеді: профиль қолайсыздау болды.",
        "same": "Аймақтар мен қорытынды бағалар айтарлықтай өзгермеді; өзгеріс сол оқу аймағында қалды.",
    },
    "en": {
        "passive_up": "The simulated passive BBB score improved; compare the CNS score because P-gp may still reduce exposure.",
        "passive_down": "The simulated passive BBB score decreased; compare the CNS score because P-gp may further reduce exposure.",
        "passive_same": "The passive BBB score changed only slightly.",
        "pgp_up": "Increasing P-gp probability lowers CNS score even if passive BBB score remains acceptable.",
        "pgp_down": "Decreasing P-gp probability reduces the educational efflux risk and may increase CNS score.",
        "cns_up": "The final CNS score increased: the profile became more favourable for CNS exposure.",
        "cns_down": "The final CNS score decreased: the profile became less favourable for CNS exposure.",
        "same": "Zones and final scores changed only slightly; the shift remained within the same educational zone.",
    },
}


def normalize_language(lang: str | None) -> str:
    if not lang:
        return DEFAULT_LANGUAGE
    code = str(lang).strip().lower()
    if code in SUPPORTED_LANGUAGES:
        return code
    if code in LANGUAGE_LABEL_TO_CODE:
        return LANGUAGE_LABEL_TO_CODE[code]
    for label, value in LANGUAGE_LABEL_TO_CODE.items():
        if str(label).strip().lower() == code:
            return value
    return DEFAULT_LANGUAGE


def language_options() -> list[str]:
    return list(SUPPORTED_LANGUAGES.values())


def language_label_to_code(label: str | None) -> str:
    if not label:
        return DEFAULT_LANGUAGE
    if label in LANGUAGE_LABEL_TO_CODE:
        return LANGUAGE_LABEL_TO_CODE[label]
    return normalize_language(label)


def t(key: str, lang: str = DEFAULT_LANGUAGE, **kwargs: Any) -> str:
    lang = normalize_language(lang)
    ui_template = UI.get(lang, {}).get(key, UI[DEFAULT_LANGUAGE].get(key))
    if ui_template is not None:
        try:
            return str(ui_template).format(**kwargs)
        except Exception:
            return str(ui_template)
    # Compatibility keys used by earlier Stage 1-5 modules.
    if key == "msg.in_silico":
        template = disclaimer("in_silico", lang)
    elif key == "msg.what_if_disclaimer":
        template = disclaimer("what_if", lang)
    elif key == "msg.descriptors_unavailable":
        template = {
            "ru": "Дескрипторы недоступны для этой молекулы.",
            "kk": "Бұл молекула үшін дескрипторлар қолжетімсіз.",
            "en": "Descriptors are unavailable for this molecule.",
        }.get(lang, "Дескрипторы недоступны для этой молекулы.")
    elif key.startswith("applicability."):
        template = applicability_message(key.split(".", 1)[1], lang)
    elif key.startswith("uncertainty."):
        level = key.split(".", 1)[1]
        template = {
            "ru": {
                "low": "Большинство факторов согласованы; неопределённость выглядит низкой.",
                "medium": "Есть пограничные значения или противоречивые сигналы; прогноз следует трактовать осторожно.",
                "high": "Неопределённость высокая; результат нельзя использовать как твёрдый вывод.",
            },
            "kk": {
                "low": "Факторлардың көбі келісімді; белгісіздік төмен көрінеді.",
                "medium": "Шекаралық мәндер немесе қайшы сигналдар бар; болжамды сақтықпен түсіндіру керек.",
                "high": "Белгісіздік жоғары; нәтижені нақты қорытынды ретінде қолдануға болмайды.",
            },
            "en": {
                "low": "Most factors are consistent; uncertainty appears low.",
                "medium": "There are borderline values or conflicting signals; interpret the prediction cautiously.",
                "high": "Uncertainty is high; do not use the result as a firm conclusion.",
            },
        }[lang].get(level, key)
    elif key == "batch.summary_text":
        template = batch_summary_text(
            total=kwargs.get("total", kwargs.get("n_total", 0)),
            valid=kwargs.get("valid", kwargs.get("n_valid", 0)),
            invalid=kwargs.get("invalid", kwargs.get("n_invalid", 0)),
            cns=kwargs.get("cns", kwargs.get("cns_count", 0)),
            efflux=kwargs.get("efflux", kwargs.get("efflux_count", 0)),
            borderline=kwargs.get("borderline", kwargs.get("borderline_count", 0)),
            lang=lang,
        )
        return str(template)
    else:
        template = UI.get(lang, {}).get(key, UI[DEFAULT_LANGUAGE].get(key, key))
    try:
        return str(template).format(**kwargs)
    except Exception:
        return str(template)


def zone_label(zone: str, lang: str = DEFAULT_LANGUAGE, *, badge: bool = False) -> str:
    lang = normalize_language(lang)
    table = ZONE_BADGE_LABELS if badge else ZONE_LABELS
    return table.get(lang, table[DEFAULT_LANGUAGE]).get(str(zone), str(zone))


def effect_label(effect: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    return EFFECT_LABELS.get(lang, EFFECT_LABELS[DEFAULT_LANGUAGE]).get(str(effect), str(effect))


def descriptor_base(name: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    return DESCRIPTOR_BASE.get(lang, DESCRIPTOR_BASE[DEFAULT_LANGUAGE]).get(str(name), DESCRIPTOR_BASE[DEFAULT_LANGUAGE].get(str(name), ""))


def descriptor_zone_comment(name: str, zone: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    lang_comments = ZONE_COMMENTS.get(lang, ZONE_COMMENTS[DEFAULT_LANGUAGE])
    ru_comments = ZONE_COMMENTS[DEFAULT_LANGUAGE]
    return lang_comments.get(str(name), ru_comments.get(str(name), {})).get(str(zone), ru_comments.get(str(name), {}).get(str(zone), ""))


def final_decision_text(final_class: str, lang: str = DEFAULT_LANGUAGE) -> dict[str, str]:
    lang = normalize_language(lang)
    table = FINAL_DECISIONS.get(lang, FINAL_DECISIONS[DEFAULT_LANGUAGE])
    fallback = FINAL_DECISIONS[DEFAULT_LANGUAGE].get("uncertain_or_borderline", {})
    result = deepcopy(table.get(str(final_class), fallback))
    # Stage 1-5 used final_label; Russian templates used final_label_ru.
    if "final_label" not in result and "final_label_ru" in result:
        result["final_label"] = result["final_label_ru"]
    if "final_label_ru" not in result and "final_label" in result:
        result["final_label_ru"] = result["final_label"]
    return result


def matrix_intro(lang: str = DEFAULT_LANGUAGE) -> str:
    from core.matrix_text import matrix_intro as _matrix_intro

    return _matrix_intro(normalize_language(lang))


def matrix_cells(lang: str = DEFAULT_LANGUAGE) -> dict[str, dict[str, str]]:
    from core.matrix_text import matrix_cells as _matrix_cells

    return _matrix_cells(normalize_language(lang))


def disclaimer(kind: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    return DISCLAIMERS.get(lang, DISCLAIMERS[DEFAULT_LANGUAGE]).get(kind, DISCLAIMERS[DEFAULT_LANGUAGE].get(kind, ""))


def applicability_message(level: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    return APPLICABILITY_MESSAGES.get(lang, APPLICABILITY_MESSAGES[DEFAULT_LANGUAGE]).get(str(level), APPLICABILITY_MESSAGES[DEFAULT_LANGUAGE].get(str(level), ""))


def warning_message(code: str, lang: str = DEFAULT_LANGUAGE, **kwargs: Any) -> str:
    lang = normalize_language(lang)
    template = WARNING_MESSAGES.get(lang, WARNING_MESSAGES[DEFAULT_LANGUAGE]).get(str(code), WARNING_MESSAGES[DEFAULT_LANGUAGE].get(str(code), str(code)))
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def methodology_sections(lang: str = DEFAULT_LANGUAGE) -> list[dict[str, str]]:
    lang = normalize_language(lang)
    return deepcopy(METHODOLOGY.get(lang, METHODOLOGY[DEFAULT_LANGUAGE]))


def limitations(lang: str = DEFAULT_LANGUAGE) -> list[str]:
    lang = normalize_language(lang)
    return list(LIMITATIONS.get(lang, LIMITATIONS[DEFAULT_LANGUAGE]))


def student_questions(lang: str = DEFAULT_LANGUAGE) -> list[str]:
    lang = normalize_language(lang)
    return list(STUDENT_QUESTIONS.get(lang, STUDENT_QUESTIONS[DEFAULT_LANGUAGE]))


def batch_priority_label(priority: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    return BATCH_PRIORITY_LABELS.get(lang, BATCH_PRIORITY_LABELS[DEFAULT_LANGUAGE]).get(str(priority), str(priority))


def batch_summary_text(*, total: int, valid: int, invalid: int, cns: int, efflux: int, borderline: int, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    template = BATCH_SUMMARY_TEMPLATES.get(lang, BATCH_SUMMARY_TEMPLATES[DEFAULT_LANGUAGE])
    return template.format(total=total, valid=valid, invalid=invalid, cns=cns, efflux=efflux, borderline=borderline)


def what_if_text(key: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    return WHAT_IF_TEXTS.get(lang, WHAT_IF_TEXTS[DEFAULT_LANGUAGE]).get(key, WHAT_IF_TEXTS[DEFAULT_LANGUAGE].get(key, key))


def descriptor_explanation(name: str, value_text: str, zone: str, lang: str = DEFAULT_LANGUAGE) -> str:
    base = descriptor_base(name, lang)
    comment = descriptor_zone_comment(name, zone, lang)
    prefix = f"{name} = {value_text}." if value_text else f"{name}: N/A."
    return " ".join(part for part in [prefix, base, comment] if part)


def localize_explanation_dict(explanation_dict: Mapping[str, Any], lang: str = DEFAULT_LANGUAGE) -> dict[str, Any]:
    """Return a copy of explanation_dict with user-facing fields localised."""
    lang = normalize_language(lang)
    data = deepcopy(dict(explanation_dict or {}))
    data["language"] = lang

    # Descriptors.
    descriptors = data.get("descriptors") or {}
    for name, item in descriptors.items():
        if not isinstance(item, dict):
            continue
        zone = str(item.get("zone", "gray"))
        effect = str(item.get("effect", "context_dependent"))
        item["zone_label"] = zone_label(zone, lang)
        item["effect_label"] = effect_label(effect, lang)
        value = item.get("value")
        unit = item.get("unit") or ""
        value_text = _format_value(value, unit)
        item["explanation"] = descriptor_explanation(str(name), value_text, zone, lang)
        # Keep display names technical, but localise threshold note if possible.
        if lang != "ru":
            item["threshold_note"] = "Educational heuristic; interpret together with the full molecular context." if lang == "en" else "Оқу эвристикасы; толық молекулалық контекстпен бірге түсіндіріңіз."

    # Factors reuse localised descriptor explanations.
    for group in (data.get("factor_summary") or {}).values():
        if isinstance(group, list):
            for factor in group:
                if isinstance(factor, dict):
                    name = str(factor.get("name") or "")
                    zone = str((descriptors.get(name) or {}).get("zone", factor.get("zone", "gray")))
                    factor["display_name"] = factor.get("display_name") or name
                    factor["reason"] = descriptor_zone_comment(name, zone, lang) or str(factor.get("reason", ""))

    # Decision.
    decision = data.get("decision_explanation") or {}
    final_class = str(decision.get("final_class") or (data.get("model_outputs") or {}).get("final_cns_class") or "uncertain_or_borderline")
    localized_decision = final_decision_text(final_class, lang)
    if isinstance(decision, dict):
        decision.update(localized_decision)
        decision["final_label"] = localized_decision.get("final_label_ru") or localized_decision.get("title")
        decision["final_label_ru"] = localized_decision.get("final_label_ru") or localized_decision.get("title")
        decision["caution_notes"] = _localize_notes(decision.get("caution_notes"), lang)
        data["decision_explanation"] = decision

    # Matrix.
    matrix = data.get("bbb_pgp_matrix") or {}
    if isinstance(matrix, dict):
        cells = matrix_cells(lang)
        current = str(matrix.get("current_cell") or "insufficient_data")
        matrix["intro_text"] = matrix_intro(lang)
        matrix["cells"] = cells
        matrix["current_interpretation"] = cells.get(current, cells.get("insufficient_data", {})).get("interpretation", "")
        data["bbb_pgp_matrix"] = matrix

    # Applicability/warnings.
    app = data.get("applicability_domain") or {}
    if isinstance(app, dict):
        level = str(app.get("level") or "unknown")
        app["student_message"] = applicability_message(level, lang) or str(app.get("student_message", ""))
        flags = list(app.get("flags") or [])
        warnings = app.get("warnings") or []
        localized_warnings = []
        if isinstance(warnings, list):
            for warning in warnings:
                if isinstance(warning, dict):
                    code = str(warning.get("code", "warning"))
                    w = deepcopy(warning)
                    w["message"] = warning_message(code, lang)
                    w["student_message"] = warning_message(code, lang)
                    w["recommendation"] = _recommendation_for_warning(code, lang)
                    localized_warnings.append(w)
        app["warnings"] = localized_warnings
        app["reasons"] = [warning_message(str(flag), lang) for flag in flags] if flags else _localize_notes(app.get("reasons"), lang)
        data["applicability_domain"] = app

    molecule = data.get("molecule") or {}
    if isinstance(molecule, dict):
        molecule_warnings = molecule.get("warnings") or []
        new_warnings = []
        if isinstance(molecule_warnings, list):
            for warning in molecule_warnings:
                if isinstance(warning, dict):
                    code = str(warning.get("code", "warning"))
                    w = deepcopy(warning)
                    w["message"] = warning_message(code, lang)
                    w["student_message"] = warning_message(code, lang)
                    new_warnings.append(w)
        molecule["warnings"] = new_warnings
        data["molecule"] = molecule

    # Stepwise trace.
    data["stepwise_trace"] = _localize_stepwise(data, lang)

    # Uncertainty.
    unc = data.get("uncertainty") or {}
    if isinstance(unc, dict):
        if str(unc.get("level")) == "high":
            unc["student_message"] = {
                "ru": "Прогноз следует трактовать очень осторожно из-за ошибок или потому, что молекула плохо похожа на структуры, для которых модель обычно надёжна.",
                "kk": "Қате немесе қолданылу доменінен шығу себебінен болжамды өте сақ түсіндіру керек.",
                "en": "Interpret the prediction very cautiously due to errors or outside-domain signals.",
            }[lang]
        elif str(unc.get("level")) == "medium":
            unc["student_message"] = {
                "ru": "Прогноз следует трактовать осторожно: отдельные факторы или модели дают пограничные либо противоречивые сигналы.",
                "kk": "Болжамды сақ түсіндіру керек: кейбір факторлар немесе модельдер шекаралық не қарама-қайшы сигнал береді.",
                "en": "Interpret cautiously: some factors or models provide borderline or conflicting signals.",
            }[lang]
        else:
            unc["student_message"] = {
                "ru": "Основные сигналы модели согласованы; неопределённость в рамках rule-based объяснения низкая.",
                "kk": "Модельдің негізгі сигналдары келісілген; rule-based түсіндірме шеңберінде белгісіздік төмен.",
                "en": "The main model signals are consistent; uncertainty is low within the rule-based explanation.",
            }[lang]
        unc["reasons"] = _localize_notes(unc.get("reasons"), lang)
        data["uncertainty"] = unc

    data["disclaimers"] = {"in_silico": disclaimer("in_silico", lang), "what_if": disclaimer("what_if", lang)}
    what_if_base = data.get("what_if_base") or {}
    if isinstance(what_if_base, dict):
        what_if_base["disclaimer"] = disclaimer("what_if", lang)
        data["what_if_base"] = what_if_base
    return data


def _localize_stepwise(data: Mapping[str, Any], lang: str) -> list[dict[str, Any]]:
    labels = {
        "ru": ["Проверка SMILES", "Расчёт дескрипторов", "Оценка прохождения через ГЭБ", "Оценка P-gp", "Финальная учебная интерпретация"],
        "kk": ["SMILES тексеру", "Дескрипторларды есептеу", "Қан-ми тосқауылынан пассивті өтуді бағалау", "P-gp бағалау", "Қорытынды оқу түсіндірмесі"],
        "en": ["SMILES validation", "Descriptor calculation", "Passive BBB permeability assessment", "P-gp assessment", "Final educational interpretation"],
    }[lang]
    steps = []
    model_outputs = data.get("model_outputs") or {}
    decision = data.get("decision_explanation") or {}
    molecule = data.get("molecule") or {}
    warnings = molecule.get("warnings") or []
    if molecule.get("valid") and not warnings:
        step1_status = "ok"
        step1_msg = {"ru": "SMILES успешно распознан; явных предупреждений нет.", "kk": "SMILES сәтті танылды; айқын ескерту жоқ.", "en": "SMILES was parsed successfully; no clear warnings."}[lang]
    elif molecule.get("valid"):
        step1_status = "warning"
        step1_msg = {"ru": "SMILES распознан, но есть предупреждения.", "kk": "SMILES танылды, бірақ ескертулер бар.", "en": "SMILES was parsed, but warnings are present."}[lang]
    else:
        step1_status = "error"
        step1_msg = {"ru": "SMILES не удалось корректно обработать.", "kk": "SMILES дұрыс өңделмеді.", "en": "SMILES could not be processed correctly."}[lang]
    desc_names = list((data.get("descriptors") or {}).keys())
    step2_msg = {
        "ru": "Рассчитаны или получены признаки: {names}.",
        "kk": "Есептелген немесе алынған белгілер: {names}.",
        "en": "Calculated or retrieved features: {names}.",
    }[lang].format(names=", ".join(desc_names) if desc_names else "N/A")
    bbb = model_outputs.get("bbb_classifier_probability")
    pgp = model_outputs.get("pgp_probability")
    step3_msg = {"ru": "Оценка прохождения через ГЭБ: {score}.", "kk": "Қан-ми тосқауылынан өту бағасы: {score}.", "en": "BBB score: {score}."}[lang].format(score=_format_value(bbb, ""))
    step4_msg = {"ru": "Оценка P-gp: {score}.", "kk": "P-gp бағасы: {score}.", "en": "P-gp score: {score}."}[lang].format(score=_format_value(pgp, ""))
    steps.append({"step": 1, "title": labels[0], "status": step1_status, "message": step1_msg})
    steps.append({"step": 2, "title": labels[1], "status": "ok" if desc_names else "warning", "message": step2_msg})
    steps.append({"step": 3, "title": labels[2], "status": "ok" if bbb is not None else "warning", "message": step3_msg})
    steps.append({"step": 4, "title": labels[3], "status": "ok" if pgp is not None else "warning", "message": step4_msg})
    steps.append({"step": 5, "title": labels[4], "status": "ok" if decision.get("final_class") == "likely_cns_active" else "warning", "message": str(decision.get("summary", ""))})
    return steps


def _localize_notes(notes: Any, lang: str) -> list[str]:
    if not notes:
        return []
    if isinstance(notes, str):
        notes = [notes]
    result = []
    for note in notes:
        text = str(note)
        # Translate common internal reason fragments.
        mapping = {
            "BBB and P-gp signals conflict.": {"ru": "ГЭБ и P-gp дают конфликтующие сигналы.", "kk": "Қан-ми тосқауылы және P-gp сигналдары қарама-қайшы.", "en": "BBB and P-gp signals conflict."},
            "invalid_smiles": {"ru": "Некорректный SMILES.", "kk": "Қате SMILES.", "en": "Invalid SMILES."},
        }
        result.append(mapping.get(text, {}).get(lang, text))
    return result


def _recommendation_for_warning(code: str, lang: str) -> str:
    recs = {
        "ru": "Интерпретируйте прогноз как учебную гипотезу и проверьте структуру/экспериментальные данные.",
        "kk": "Болжамды оқу гипотезасы ретінде түсіндіріп, құрылымды және эксперименттік деректерді тексеріңіз.",
        "en": "Treat the prediction as an educational hypothesis and verify the structure or experimental evidence.",
    }
    return recs[lang]


def _format_value(value: Any, unit: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        f = float(value)
        text = f"{f:.3g}" if abs(f) < 1000 else f"{f:.1f}"
    except (TypeError, ValueError):
        text = str(value)
    return f"{text} {unit}".strip()



# ---------------------------------------------------------------------------
# Compatibility aliases used by integration modules.
# ---------------------------------------------------------------------------

def language_selectbox_options() -> list[str]:
    return language_options()


def zone_badge(zone: str, lang: str = DEFAULT_LANGUAGE) -> str:
    return zone_label(zone, lang, badge=True)


def descriptor_base_text(name: str, lang: str = DEFAULT_LANGUAGE) -> str:
    return descriptor_base(name, lang)


def descriptor_display_name(name: str, fallback: str | None = None, lang: str = DEFAULT_LANGUAGE) -> str:
    meta = DESCRIPTOR_META.get(str(name), {}) if isinstance(DESCRIPTOR_META, dict) else {}
    return str(meta.get("display_name") or fallback or name)


def descriptor_short_label(name: str, fallback: str | None = None, lang: str = DEFAULT_LANGUAGE) -> str:
    meta = DESCRIPTOR_META.get(str(name), {}) if isinstance(DESCRIPTOR_META, dict) else {}
    return str(meta.get("short_label") or fallback or name)


def descriptor_threshold_note(name: str, lang: str = DEFAULT_LANGUAGE) -> str:
    meta = DESCRIPTOR_META.get(str(name), {}) if isinstance(DESCRIPTOR_META, dict) else {}
    return str(meta.get("threshold_note") or "")


def warning_text(code: str, lang: str = DEFAULT_LANGUAGE) -> dict[str, str]:
    msg = warning_message(code, lang)
    return {"message": msg, "student_message": msg, "recommendation": ""}


def localize_warning(warning: Mapping[str, Any] | str, lang: str = DEFAULT_LANGUAGE) -> dict[str, Any]:
    if isinstance(warning, Mapping):
        code = str(warning.get("code", "warning"))
        severity = str(warning.get("severity", "warning"))
        original = str(warning.get("message") or warning.get("text") or code)
    else:
        code = "warning"
        severity = "warning"
        original = str(warning)
    msg = warning_message(code, lang)
    if msg == code and original and original != code:
        msg = original
    return {"code": code, "severity": severity, "message": msg, "student_message": msg, "recommendation": ""}


def clone_translations_for_tests() -> dict[str, dict[str, str]]:
    # The current implementation is table-based rather than one giant TRANSLATIONS dict.
    return {"ru": dict(UI.get("ru", {})), "kk": dict(UI.get("kk", {})), "en": dict(UI.get("en", {}))}

def descriptor_zone_comment(name: str, zone: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    name = str(name)
    zone = str(zone)
    lang_comments = ZONE_COMMENTS.get(lang, {})
    comment = lang_comments.get(name, {}).get(zone)
    if comment:
        return comment
    if lang != DEFAULT_LANGUAGE:
        return {
            "kk": "Бұл белгі контекстпен бірге түсіндірілуі керек.",
            "en": "This feature should be interpreted in context.",
        }.get(lang, "")
    return ZONE_COMMENTS.get(DEFAULT_LANGUAGE, {}).get(name, {}).get(zone, "Влияние зависит от контекста.")


def descriptor_display_name(name: str, fallback: str | None = None, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    name = str(name)
    return DESCRIPTOR_DISPLAY_NAMES.get(lang, DESCRIPTOR_DISPLAY_NAMES[DEFAULT_LANGUAGE]).get(name, fallback or name)


def descriptor_short_label(name: str, fallback: str | None = None, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    name = str(name)
    return DESCRIPTOR_SHORT_LABELS.get(lang, DESCRIPTOR_SHORT_LABELS[DEFAULT_LANGUAGE]).get(name, fallback or descriptor_display_name(name, name, lang))


def descriptor_threshold_note(name: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    name = str(name)
    return DESCRIPTOR_THRESHOLD_NOTES.get(lang, DESCRIPTOR_THRESHOLD_NOTES[DEFAULT_LANGUAGE]).get(name, "")


def descriptor_explanation(name: str, value_text: str, zone: str, lang: str = DEFAULT_LANGUAGE) -> str:
    lang = normalize_language(lang)
    canonical = str(name)
    label = descriptor_short_label(canonical, canonical, lang)
    base = descriptor_base(canonical, lang)
    comment = descriptor_zone_comment(canonical, zone, lang)
    if value_text:
        prefix = {
            "ru": f"{label} = {value_text}.",
            "kk": f"{label} = {value_text}.",
            "en": f"{label} = {value_text}.",
        }[lang]
    else:
        prefix = {"ru": f"{label}: нет данных.", "kk": f"{label}: дерек жоқ.", "en": f"{label}: N/A."}[lang]
    return " ".join(part for part in [prefix, base, comment] if part)


_base_localize_explanation_dict = localize_explanation_dict


def localize_explanation_dict(explanation_dict: Mapping[str, Any], lang: str = DEFAULT_LANGUAGE) -> dict[str, Any]:
    """Localize descriptor labels, notes and factor names."""
    lang = normalize_language(lang)
    data = _base_localize_explanation_dict(explanation_dict, lang)
    descriptors = data.get("descriptors") or {}
    if isinstance(descriptors, dict):
        for name, item in descriptors.items():
            if not isinstance(item, dict):
                continue
            canonical = str(name)
            zone = str(item.get("zone", "gray"))
            item["display_name"] = descriptor_display_name(canonical, item.get("display_name"), lang)
            item["short_label"] = descriptor_short_label(canonical, item.get("short_label"), lang)
            item["zone_label"] = zone_label(zone, lang)
            item["effect_label"] = effect_label(str(item.get("effect", "context_dependent")), lang)
            item["threshold_note"] = descriptor_threshold_note(canonical, lang)
            item["explanation"] = descriptor_explanation(canonical, _format_value(item.get("value"), item.get("unit") or ""), zone, lang)
    factor_summary = data.get("factor_summary") or {}
    if isinstance(factor_summary, dict):
        for group in factor_summary.values():
            if not isinstance(group, list):
                continue
            for factor in group:
                if not isinstance(factor, dict):
                    continue
                name = str(factor.get("name") or "")
                zone = str((descriptors.get(name) or {}).get("zone", factor.get("zone", "gray")))
                factor["display_name"] = descriptor_short_label(name, factor.get("display_name"), lang)
                factor["reason"] = descriptor_zone_comment(name, zone, lang)
    return data

# Legacy uncertainty/reason note localization.
def _localize_notes(notes: Any, lang: str) -> list[str]:
    lang = normalize_language(lang)
    if not notes:
        return []
    if isinstance(notes, str):
        notes = [notes]
    mapping = {
        "BBB and P-gp signals conflict.": {"ru": "ГЭБ и P-gp дают конфликтующие сигналы.", "kk": "Қан-ми тосқауылы және P-gp сигналдары қарама-қайшы.", "en": "BBB and P-gp signals conflict."},
        "invalid_smiles": {"ru": "Некорректный SMILES.", "kk": "Қате SMILES.", "en": "Invalid SMILES."},
        "Молекула вне базового домена применимости.": {"ru": "Молекула плохо похожа на структуры, для которых модель обычно надёжна.", "kk": "Молекула модель әдетте сенімді жұмыс істейтін құрылымдарға онша ұқсамайды.", "en": "The molecule is outside the basic applicability domain."},
        "Есть предупреждения по домену применимости.": {"ru": "Есть предупреждения о надёжности модели для этой молекулы.", "kk": "Бұл молекула үшін модель сенімділігі туралы ескертулер бар.", "en": "Applicability-domain warnings are present."},
        "Недоступен BBB score или P-gp score.": {"ru": "Недоступна оценка ГЭБ или P-gp.", "kk": "Қан-ми тосқауылы немесе P-gp бағасы қолжетімсіз.", "en": "BBB score or P-gp score is unavailable."},
        "BBB score находится в пограничной зоне.": {"ru": "Оценка прохождения через ГЭБ находится в пограничной зоне.", "kk": "Қан-ми тосқауылынан өту бағасы шекаралық аймақта.", "en": "BBB score is in a borderline zone."},
        "P-gp score находится в зоне неопределённости.": {"ru": "Оценка P-gp находится в зоне неопределённости.", "kk": "P-gp бағасы анық емес аймақта.", "en": "P-gp score is in the uncertainty zone."},
        "Есть одновременно положительные и отрицательные факторы.": {"ru": "Есть одновременно положительные и отрицательные факторы.", "kk": "Оң және теріс факторлар қатар бар.", "en": "Positive and negative factors are both present."},
        "BBB score близок к верхнему порогу; интерпретация чувствительна к выбранному cut-off.": {"ru": "Оценка прохождения через ГЭБ близка к верхнему порогу; вывод чувствителен к выбранной границе.", "kk": "Қан-ми тосқауылынан өту бағасы жоғарғы шекке жақын; түсіндіру таңдалған шекке сезімтал.", "en": "BBB score is close to the upper threshold; interpretation is sensitive to the selected cut-off."},
        "BBB score близок к нижнему порогу; интерпретация чувствительна к выбранному cut-off.": {"ru": "Оценка прохождения через ГЭБ близка к нижнему порогу; вывод чувствителен к выбранной границе.", "kk": "Қан-ми тосқауылынан өту бағасы төменгі шекке жақын; түсіндіру таңдалған шекке сезімтал.", "en": "BBB score is close to the lower threshold; interpretation is sensitive to the selected cut-off."},
        "P-gp score близок к порогу substrate; вывод о P-gp следует считать осторожным.": {"ru": "Оценка P-gp близка к порогу активного выведения; вывод о P-gp следует считать осторожным.", "kk": "P-gp бағасы белсенді шығарылу шегіне жақын; P-gp қорытындысын сақтықпен түсіндіру керек.", "en": "P-gp score is close to the substrate threshold; interpret the P-gp conclusion cautiously."},
        "P-gp score близок к порогу non-substrate; вывод о P-gp следует считать осторожным.": {"ru": "Оценка P-gp близка к порогу низкого риска; вывод о P-gp следует считать осторожным.", "kk": "P-gp бағасы төмен қауіп шегіне жақын; P-gp қорытындысын сақтықпен түсіндіру керек.", "en": "P-gp score is close to the non-substrate threshold; interpret the P-gp conclusion cautiously."},
    }
    result: list[str] = []
    for note in notes:
        text = str(note)
        result.append(mapping.get(text, {}).get(lang, text))
    return result
