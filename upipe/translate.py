import os
from exorde_data import Translation, Language, Translated, Item
from lang_detect import is_target_language

LANG_MIN_CONFIDENCE = float(os.getenv("LANG_MIN_CONFIDENCE", "0.9"))
LANG_DETECT_MIN_LEN = int(os.getenv("LANG_DETECT_MIN_LEN", "30"))


class NonEnglishError(ValueError):
    pass


def translate(item: Item, installed_languages, low_memory: bool = False) -> Translation:
    content = str(item.content)
    passed, detected_lang, confidence = is_target_language(
        content, "en", LANG_MIN_CONFIDENCE, LANG_DETECT_MIN_LEN
    )
    if not passed:
        raise NonEnglishError(f"не-английский/неопределённый текст: lang={detected_lang} confidence={confidence:.3f}")

    return Translation(
        language=Language("en"),
        translation=Translated(content),
    )
