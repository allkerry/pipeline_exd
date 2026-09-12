import os
from langdetect import detect_langs, DetectorFactory, LangDetectException
from exorde_data import Translation, Language, Translated, Item

DetectorFactory.seed = 0

LANG_CONFIDENCE_THRESHOLD = float(os.getenv("LANG_CONFIDENCE_THRESHOLD", "0.90"))


class NonEnglishError(ValueError):
    pass


def translate(item: Item, installed_languages, low_memory: bool = False) -> Translation:
    content = str(item.content)

    if not content or not any(c.isalpha() for c in content):
        return Translation(language=Language(""), translation=Translated(""))

    try:
        candidates = detect_langs(content)
    except LangDetectException:
        return Translation(language=Language(""), translation=Translated(""))

    if not candidates:
        return Translation(language=Language(""), translation=Translated(""))

    top = candidates[0]
    if top.lang != "en" and top.prob >= LANG_CONFIDENCE_THRESHOLD:
        raise NonEnglishError(f"не-английский текст обнаружен: {top.lang} (p={top.prob:.2f})")

    return Translation(
        language=Language("en"),
        translation=Translated(content),
    )
