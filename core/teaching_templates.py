"""Russian teaching templates and rule configuration for BioSynth-EDU explainability.

This module intentionally contains no model inference code.  It stores the
educational vocabulary, descriptor metadata and threshold constants used by
``core.explainability``.
"""

from __future__ import annotations

BBB_LOW_THRESHOLD = 0.40
BBB_HIGH_THRESHOLD = 0.70
PGP_LOW_THRESHOLD = 0.35
PGP_HIGH_THRESHOLD = 0.65
BORDERLINE_DISTANCE = 0.05

ZONE_LABELS_RU = {
    "green": "помогает",
    "yellow": "погранично",
    "red": "мешает",
    "gray": "неопределённо",
}

EFFECT_LABELS_RU = {
    "supports_bbb": "поддерживает BBB-проницаемость",
    "opposes_bbb": "снижает BBB-проницаемость",
    "borderline": "пограничное влияние",
    "supports_cns_exposure": "поддерживает CNS-доступность",
    "opposes_cns_exposure": "снижает CNS-доступность",
    "context_dependent": "зависит от контекста",
    "unknown": "недоступно",
    "uncertain": "неопределённо",
}

DESCRIPTOR_ORDER = [
    "MW",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "AromaticRings",
    "pKa_pred",
    "FormalCharge",
    "GasteigerMin",
    "GasteigerMax",
    "GasteigerAbsMax",
    "BBB_probability",
    "Pgp_probability",
    "Clint_risk",
    "CATMoS_LD50",
]

DESCRIPTOR_META = {
    "MW": {
        "display_name": "Molecular Weight",
        "short_label": "MW",
        "unit": "Da",
        "importance": "high",
        "threshold_note": "Учебная эвристика: 150–450 Da обычно благоприятнее для CNS-профиля.",
    },
    "LogP": {
        "display_name": "LogP",
        "short_label": "LogP",
        "unit": "",
        "importance": "high",
        "threshold_note": "Учебная эвристика: умеренный LogP примерно 1.5–3.5 поддерживает пассивную диффузию.",
    },
    "TPSA": {
        "display_name": "Topological Polar Surface Area",
        "short_label": "TPSA",
        "unit": "Å²",
        "importance": "high",
        "threshold_note": "Учебная эвристика: TPSA ≤ 70 Å² обычно благоприятнее для BBB; > 90 Å² часто неблагоприятно.",
    },
    "HBD": {
        "display_name": "Hydrogen Bond Donors",
        "short_label": "HBD",
        "unit": "",
        "importance": "medium",
        "threshold_note": "Учебная эвристика: 0–1 HBD обычно благоприятно; ≥ 3 часто мешает пассивной BBB-диффузии.",
    },
    "HBA": {
        "display_name": "Hydrogen Bond Acceptors",
        "short_label": "HBA",
        "unit": "",
        "importance": "medium",
        "threshold_note": "Учебная эвристика: 0–5 HBA обычно допустимо; ≥ 8 часто неблагоприятно.",
    },
    "RotatableBonds": {
        "display_name": "Rotatable Bonds",
        "short_label": "RotB",
        "unit": "",
        "importance": "low",
        "threshold_note": "Учебная эвристика: 0–5 вращаемых связей обычно лучше; > 8 указывает на высокую гибкость.",
    },
    "AromaticRings": {
        "display_name": "Aromatic Rings",
        "short_label": "Aromatic rings",
        "unit": "",
        "importance": "low",
        "threshold_note": "Учебная эвристика: умеренное число ароматических колец допустимо; избыток может повышать липофильность и связывание.",
    },
    "pKa_pred": {
        "display_name": "Predicted pKa",
        "short_label": "pKa",
        "unit": "",
        "importance": "medium",
        "threshold_note": "Интерпретация pKa зависит от кислотно-основного центра; pKa около 7.4 требует осторожности.",
    },
    "FormalCharge": {
        "display_name": "Formal Charge",
        "short_label": "Charge",
        "unit": "",
        "importance": "medium",
        "threshold_note": "Учебная эвристика: нейтральная форма обычно благоприятнее для пассивной BBB-диффузии.",
    },
    "GasteigerMin": {
        "display_name": "Minimum Gasteiger Charge",
        "short_label": "Gasteiger min",
        "unit": "",
        "importance": "low",
        "threshold_note": "Частичные заряды следует интерпретировать вместе с TPSA, HBD/HBA и формальным зарядом.",
    },
    "GasteigerMax": {
        "display_name": "Maximum Gasteiger Charge",
        "short_label": "Gasteiger max",
        "unit": "",
        "importance": "low",
        "threshold_note": "Частичные заряды следует интерпретировать вместе с TPSA, HBD/HBA и формальным зарядом.",
    },
    "GasteigerAbsMax": {
        "display_name": "Maximum Absolute Gasteiger Charge",
        "short_label": "|Gasteiger|max",
        "unit": "",
        "importance": "low",
        "threshold_note": "Высокий локальный заряд может указывать на выраженную полярность.",
    },
    "BBB_probability": {
        "display_name": "BBB probability",
        "short_label": "BBB prob.",
        "unit": "probability",
        "importance": "high",
        "threshold_note": "Вероятность модели — это in silico score, а не экспериментальная вероятность.",
    },
    "Pgp_probability": {
        "display_name": "P-gp substrate probability",
        "short_label": "P-gp prob.",
        "unit": "probability",
        "importance": "high",
        "threshold_note": "Высокий P-gp score указывает на риск активного эффлюкса, а не на низкую пассивную проницаемость саму по себе.",
    },
    "Clint_risk": {
        "display_name": "Intrinsic Clearance Risk",
        "short_label": "Clint",
        "unit": "",
        "importance": "medium",
        "threshold_note": "Clint описывает метаболическую стабильность/клиренс и дополняет, но не заменяет BBB-анализ.",
    },
    "CATMoS_LD50": {
        "display_name": "CATMoS / LD50",
        "short_label": "LD50",
        "unit": "",
        "importance": "medium",
        "threshold_note": "Токсикологический прогноз следует трактовать отдельно от BBB-проницаемости.",
    },
}

# Mapping from many possible pipeline keys to a stable educational key.
DESCRIPTOR_ALIASES = {
    "mw": "MW",
    "molwt": "MW",
    "molecularweight": "MW",
    "molecular_weight": "MW",
    "molecular weight": "MW",
    "mol_wt": "MW",
    "mol wt": "MW",
    "logp": "LogP",
    "mol_logp": "LogP",
    "mollogp": "LogP",
    "clogp": "LogP",
    "crippenlogp": "LogP",
    "tpsa": "TPSA",
    "hbd": "HBD",
    "numhdonors": "HBD",
    "num_h_donors": "HBD",
    "hydrogen_bond_donors": "HBD",
    "hydrogen bond donors": "HBD",
    "hba": "HBA",
    "numhacceptors": "HBA",
    "num_h_acceptors": "HBA",
    "hydrogen_bond_acceptors": "HBA",
    "hydrogen bond acceptors": "HBA",
    "rotatablebonds": "RotatableBonds",
    "rotatable_bonds": "RotatableBonds",
    "rotatable bonds": "RotatableBonds",
    "numrotatablebonds": "RotatableBonds",
    "rotb": "RotatableBonds",
    "aromaticrings": "AromaticRings",
    "aromatic_rings": "AromaticRings",
    "aromatic rings": "AromaticRings",
    "numaromaticrings": "AromaticRings",
    "num_aromatic_rings": "AromaticRings",
    "pka": "pKa_pred",
    "pka_pred": "pKa_pred",
    "pka predicted": "pKa_pred",
    "predicted_pka": "pKa_pred",
    "predicted pka": "pKa_pred",
    "formalcharge": "FormalCharge",
    "formal_charge": "FormalCharge",
    "formal charge": "FormalCharge",
    "charge": "FormalCharge",
    "gasteigermin": "GasteigerMin",
    "gasteiger_min": "GasteigerMin",
    "gasteiger min": "GasteigerMin",
    "min_gasteiger_charge": "GasteigerMin",
    "gasteigermax": "GasteigerMax",
    "gasteiger_max": "GasteigerMax",
    "gasteiger max": "GasteigerMax",
    "max_gasteiger_charge": "GasteigerMax",
    "gasteigerabsmax": "GasteigerAbsMax",
    "gasteiger_abs_max": "GasteigerAbsMax",
    "gasteiger abs max": "GasteigerAbsMax",
    "max_abs_gasteiger_charge": "GasteigerAbsMax",
    "bbb_probability": "BBB_probability",
    "bbb probability": "BBB_probability",
    "bbb_prob": "BBB_probability",
    "bbb_classifier_probability": "BBB_probability",
    "bbb_classifier_prob": "BBB_probability",
    "pgp_probability": "Pgp_probability",
    "pgp probability": "Pgp_probability",
    "p-gp probability": "Pgp_probability",
    "p_gp_probability": "Pgp_probability",
    "pgp_prob": "Pgp_probability",
    "pgp_score": "Pgp_probability",
    "clint": "Clint_risk",
    "clint_risk": "Clint_risk",
    "intrinsic_clearance_risk": "Clint_risk",
    "catmos_ld50": "CATMoS_LD50",
    "ld50": "CATMoS_LD50",
    "catmos": "CATMoS_LD50",
}

BASE_DESCRIPTOR_EXPLANATIONS_RU = {
    "MW": (
        "Молекулярная масса показывает размер молекулы. Более крупные молекулы обычно хуже проходят "
        "через биологические барьеры, особенно если большая масса сопровождается высокой полярностью, "
        "большим числом HBD/HBA или высокой гибкостью."
    ),
    "LogP": (
        "LogP показывает липофильность молекулы. Для прохождения через ГЭБ молекула должна быть "
        "достаточно липофильной, чтобы растворяться в мембране, но не настолько липофильной, чтобы "
        "терять растворимость и чрезмерно связываться с белками. Умеренный LogP обычно поддерживает "
        "BBB-проницаемость."
    ),
    "TPSA": (
        "TPSA отражает полярную поверхность молекулы. Чем выше TPSA, тем сильнее молекула "
        "взаимодействует с водой и тем труднее ей проходить через липидный бислой. Высокая TPSA "
        "обычно снижает вероятность проникновения в ЦНС."
    ),
    "HBD": (
        "HBD показывает число доноров водородных связей. Доноры водородных связей усиливают "
        "взаимодействие с водой и могут мешать пассивной диффузии через липидный барьер. Низкое число "
        "HBD обычно благоприятно для BBB."
    ),
    "HBA": (
        "HBA показывает число акцепторов водородных связей. Большое число HBA повышает полярность "
        "и может снижать способность молекулы проходить через BBB пассивной диффузией."
    ),
    "RotatableBonds": (
        "Rotatable Bonds отражают гибкость молекулы. Очень гибкие молекулы могут хуже проходить через "
        "мембраны и чаще имеют менее предсказуемый ADMET-профиль. Этот фактор не является главным для "
        "BBB, но помогает оценивать общий drug-like профиль."
    ),
    "AromaticRings": (
        "Ароматические кольца могут повышать липофильность и связывание с белками. Умеренное число "
        "ароматических колец часто встречается у CNS-молекул, но избыток ароматичности может ухудшать "
        "растворимость и повышать неспецифическое связывание."
    ),
    "pKa_pred": (
        "pKa помогает оценить ионизационное состояние молекулы при физиологическом pH. Сильно "
        "ионизированные формы хуже проходят через липидные мембраны. Поэтому динамический pKa важен "
        "для более реалистичной оценки BBB-проницаемости."
    ),
    "FormalCharge": (
        "Формальный заряд показывает, несёт ли молекула полный положительный или отрицательный заряд. "
        "Нейтральные молекулы обычно легче проходят через липидные мембраны. Заряженные формы могут "
        "хуже проходить BBB, если для них нет специальных транспортных механизмов."
    ),
    "GasteigerMin": (
        "Частичные заряды Гастейгера показывают распределение электронной плотности в молекуле. "
        "Выраженные локальные заряды могут указывать на полярные участки, которые затрудняют пассивное "
        "прохождение через мембрану."
    ),
    "GasteigerMax": (
        "Частичные заряды Гастейгера показывают распределение электронной плотности в молекуле. "
        "Выраженные локальные заряды могут указывать на полярные участки, которые затрудняют пассивное "
        "прохождение через мембрану."
    ),
    "GasteigerAbsMax": (
        "Максимальный абсолютный частичный заряд Гастейгера помогает увидеть выраженные локальные "
        "полярные участки. Этот показатель нужно интерпретировать вместе с TPSA, HBD/HBA и формальным "
        "зарядом."
    ),
    "BBB_probability": (
        "BBB probability — это score модели, отражающий, насколько молекула похожа на BBB+ соединения "
        "в обучающем пространстве модели. Это не абсолютная биологическая вероятность и не "
        "экспериментальное доказательство."
    ),
    "Pgp_probability": (
        "P-gp — это эффлюкс-транспортер. Он может активно удалять молекулу из клеток обратно в кровь. "
        "Поэтому даже молекула с хорошими параметрами пассивной диффузии может иметь низкую фактическую "
        "доступность в ЦНС, если она является субстратом P-gp."
    ),
    "Clint_risk": (
        "Clint описывает риск быстрого внутреннего клиренса. Этот параметр не определяет саму BBB-"
        "проницаемость, но помогает понять, может ли молекула быстро удаляться или метаболизироваться."
    ),
    "CATMoS_LD50": (
        "CATMoS / LD50 относится к токсикологической оценке. Этот показатель дополняет ADMET-профиль, "
        "но не является прямым объяснением BBB-проницаемости."
    ),
}

ZONE_COMMENTS_RU = {
    "MW": {
        "green": "Значение находится в учебно благоприятном диапазоне для BBB-профиля.",
        "yellow": "Значение пограничное: само по себе не блокирует BBB, но требует контекстной оценки.",
        "red": "Значение может ухудшать пассивную BBB-проницаемость.",
        "gray": "Влияние размера на BBB не удалось оценить по доступным данным.",
    },
    "LogP": {
        "green": "Липофильность выглядит умеренной и поддерживает пассивную диффузию.",
        "yellow": "Липофильность пограничная: возможен компромисс между прохождением через мембрану и растворимостью.",
        "red": "Липофильность выглядит неблагоприятной: молекула может быть слишком гидрофильной или слишком липофильной.",
        "gray": "Влияние LogP не удалось оценить по доступным данным.",
    },
    "TPSA": {
        "green": "Полярная поверхность низкая или умеренная, что поддерживает пассивное прохождение через BBB.",
        "yellow": "TPSA находится в пограничной зоне: фактор может снижать уверенность в BBB+.",
        "red": "Высокая TPSA обычно препятствует пассивному прохождению через липидный барьер.",
        "gray": "Влияние TPSA не удалось оценить по доступным данным.",
    },
    "HBD": {
        "green": "Низкое число доноров водородных связей благоприятно для BBB.",
        "yellow": "Число доноров водородных связей пограничное.",
        "red": "Повышенное число HBD может мешать пассивной диффузии через BBB.",
        "gray": "Влияние HBD не удалось оценить по доступным данным.",
    },
    "HBA": {
        "green": "Число акцепторов водородных связей выглядит допустимым для BBB-профиля.",
        "yellow": "Число HBA пограничное и может повышать полярность.",
        "red": "Высокое число HBA может снижать пассивную BBB-проницаемость.",
        "gray": "Влияние HBA не удалось оценить по доступным данным.",
    },
    "RotatableBonds": {
        "green": "Гибкость не выглядит чрезмерной.",
        "yellow": "Гибкость пограничная: фактор не критический, но требует контекста.",
        "red": "Высокая гибкость может ухудшать ADMET-профиль и пассивную проницаемость.",
        "gray": "Влияние гибкости не удалось оценить по доступным данным.",
    },
    "AromaticRings": {
        "green": "Число ароматических колец выглядит умеренным.",
        "yellow": "Ароматичность повышена; возможен рост липофильности и связывания с белками.",
        "red": "Избыточная ароматичность может ухудшать растворимость и повышать неспецифическое связывание.",
        "gray": "Влияние ароматических колец зависит от контекста структуры.",
    },
    "pKa_pred": {
        "green": "pKa не указывает на выраженную ионизацию при pH 7.4 в рамках простой учебной эвристики.",
        "yellow": "pKa близок к физиологическому pH или требует знания кислотно-основного типа центра.",
        "red": "pKa может указывать на преимущественно ионизированную форму при физиологическом pH.",
        "gray": "Интерпретация pKa зависит от кислотно-основного центра; данных недостаточно.",
    },
    "FormalCharge": {
        "green": "Нейтральная формальная зарядность поддерживает пассивную диффузию.",
        "yellow": "Наличие формального заряда может снижать пассивную проницаемость.",
        "red": "Выраженный формальный заряд часто неблагоприятен для пассивной BBB-проницаемости.",
        "gray": "Влияние формального заряда не удалось оценить по доступным данным.",
    },
    "GasteigerAbsMax": {
        "green": "Выраженной локальной полярности по этому показателю не видно.",
        "yellow": "Есть локально полярные участки; интерпретируйте вместе с TPSA и HBD/HBA.",
        "red": "Выраженные локальные заряды могут мешать пассивной диффузии.",
        "gray": "Частичные заряды зависят от метода расчёта и требуют контекста.",
    },
    "BBB_probability": {
        "green": "Модель относит молекулу к зоне высокой оценки прохождения через ГЭБ (BBB).",
        "yellow": "Вероятность близка к переходной зоне; вывод лучше считать пограничным.",
        "red": "Модель относит молекулу к зоне низкой оценки прохождения через ГЭБ (BBB).",
        "gray": "BBB score недоступен.",
    },
    "Pgp_probability": {
        "green": "Низкий риск P-gp substrate поддерживает CNS-доступность.",
        "yellow": "P-gp score находится в зоне неопределённости.",
        "red": "Высокий риск P-gp substrate может снижать фактическую доступность в ЦНС.",
        "gray": "P-gp score недоступен.",
    },
}

FINAL_DECISION_TEXTS = {
    "likely_cns_active": {
        "title": "Вероятно ЦНС-активный профиль",
        "final_label_ru": "Вероятно ЦНС-активный профиль",
        "summary": (
            "Модель видит благоприятную пассивную BBB-проницаемость и низкий риск P-gp efflux."
        ),
        "student_interpretation": (
            "Такой профиль поддерживает гипотезу о потенциальной CNS-доступности, но не является "
            "экспериментальным доказательством проникновения в ЦНС."
        ),
    },
    "peripheral_action_risk": {
        "title": "Вероятно периферическое действие / риск низкой CNS-доступности",
        "final_label_ru": "Вероятно периферическое действие / сниженная CNS-доступность",
        "summary": (
            "Молекула имеет благоприятную пассивную BBB-проницаемость, но высокий риск P-gp efflux."
        ),
        "student_interpretation": (
            "Это пример ситуации, когда молекула может физически проходить через барьер, но затем "
            "активно вымываться транспортёром P-gp."
        ),
    },
    "likely_not_bbb_penetrant": {
        "title": "Вероятно не проникает через BBB",
        "final_label_ru": "Вероятно не проникает через BBB",
        "summary": (
            "P-gp риск не выглядит главным ограничением, но физико-химический профиль не поддерживает "
            "пассивное прохождение через BBB."
        ),
        "student_interpretation": (
            "Главное ограничение в этом сценарии — не активный эффлюкс, а недостаточно благоприятные "
            "физико-химические свойства для пассивной диффузии."
        ),
    },
    "full_barrier": {
        "title": "Полный барьер",
        "final_label_ru": "Полный барьер: плохая пассивная BBB-проницаемость плюс P-gp efflux",
        "summary": (
            "Молекула одновременно имеет неблагоприятный профиль пассивной BBB-проницаемости и высокий "
            "риск P-gp efflux."
        ),
        "student_interpretation": (
            "Это двойное ограничение для CNS-доступности: молекуле трудно пройти через барьер, а если она "
            "попадает в клетки барьера, P-gp может дополнительно вымывать её обратно в кровь."
        ),
    },
    "uncertain_or_borderline": {
        "title": "Неопределённо / погранично",
        "final_label_ru": "Неопределённо / погранично",
        "summary": (
            "Вероятности близки к порогам или разные блоки модели дают частично противоречивые сигналы."
        ),
        "student_interpretation": (
            "Такой результат лучше использовать как повод для дальнейшего анализа, а не как твёрдый вывод."
        ),
    },
    "insufficient_data": {
        "title": "Недостаточно данных для итоговой интерпретации",
        "final_label_ru": "Недостаточно данных",
        "summary": "Не хватает BBB probability или P-gp probability для матричной интерпретации.",
        "student_interpretation": (
            "Проверьте, что BBB- и P-gp-модели вернули численные score. Остальные дескрипторы можно "
            "интерпретировать отдельно."
        ),
    },
}

IN_SILICO_DISCLAIMER_RU = (
    "BioSynth-EDU предоставляет in silico-прогноз. Это не медицинская рекомендация и не "
    "экспериментальное доказательство BBB-проницаемости, токсичности или эффективности. "
    "Вероятности моделей следует трактовать как модельные score / уверенность классификатора, "
    "а не как абсолютную биологическую истину."
)

WHAT_IF_DISCLAIMER_RU = (
    "Этот блок является педагогической симуляцией. Вы изменяете отдельные дескрипторы, а не "
    "химическую структуру. Поэтому результат показывает направление влияния признаков, но не "
    "является прогнозом для новой молекулы."
)

DEFAULT_APPLICABILITY_MESSAGE_RU = "Молекула не нарушает базовые учебные правила применимости."
