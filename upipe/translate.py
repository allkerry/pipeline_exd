from langdetect import detect, DetectorFactory, LangDetectException
from exorde_data import Translation, Language, Translated, Item

DetectorFactory.seed = 0


class NonEnglishError(ValueError):
    pass


def translate(item: Item, installed_languages, low_memory: bool = False) -> Translation:
    content = str(item.content)
    try:
        detected = detect(content)
    except LangDetectException:
        raise NonEnglishError("не удалось определить язык")

    if detected != "en":
        raise NonEnglishError(f"не-английский текст обнаружен: {detected}")

    return Translation(
        language=Language("en"),
        translation=Translated(content),
    )
