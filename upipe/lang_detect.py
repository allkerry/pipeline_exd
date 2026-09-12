import re
from langdetect import detect_langs, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")


def clean_for_detection(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    return text.strip()


def is_target_language(
    text: str, target: str, min_confidence: float, min_len: int
) -> tuple[bool, str, float]:
    """Возвращает (passed, detected_lang, confidence). Fail-closed на коротких/неопределённых текстах."""
    cleaned = clean_for_detection(text)
    if len(cleaned) < min_len:
        return False, "too_short", 0.0

    try:
        results = detect_langs(cleaned)
    except LangDetectException:
        return False, "undetermined", 0.0

    if not results:
        return False, "undetermined", 0.0

    top = results[0]
    passed = (top.lang == target) and (top.prob >= min_confidence)
    return passed, top.lang, float(top.prob)
