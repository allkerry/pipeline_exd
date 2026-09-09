import logging
import os
import tiktoken

# Кэшируем энкодер на уровне модуля — tiktoken.get_encoding() парсит
# BPE-таблицу, вызывать это на каждый текст (как было раньше) — лишняя работа.
_ENCODING_NAME = "r50k_base"
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoding


def evaluate_token_count(item_content_string: str, encoding_name: str = None) -> int:
    """Возвращает приблизительное число токенов в строке.

    Используется как универсальная (не привязанная к конкретной модели)
    оценка длины — реальные модели в bpipe используют свои токенизаторы
    (BERT/DeBERTa/RoBERTa wordpiece/BPE), но r50k_base даёт достаточно
    точную оценку "на глаз", чтобы отсеять/обрезать неадекватно длинные
    тексты ещё до дорогого перевода и GPU-инференса.
    """
    try:
        if item_content_string is None or len(item_content_string) <= 1:
            logging.info("[evaluate_token_count] the content is empty")
            return 0
        encoding = _get_encoding() if encoding_name is None else tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(item_content_string))
    except Exception as e:
        logging.info(f"[evaluate_token_count] error: {e}")
        num_tokens = 0
    return num_tokens


# Максимум токенов, которые мы готовы прогонять через модели bpipe.
# Большинство используемых там transformer-моделей (DeBERTa/RoBERTa/DistilBERT)
# имеют лимит в 512 токенов собственного токенизатора; держим запас (r50k_base
# считает не так же, как их токенизаторы + добавляются служебные токены).
MAX_MODEL_TOKENS = int(os.getenv("MAX_MODEL_TOKENS", "400"))


def truncate_to_token_limit(text: str, max_tokens: int = None) -> tuple[str, int, bool]:
    """Обрезает текст так, чтобы влезть в max_tokens по оценке r50k_base.

    Возвращает (обрезанный_текст, число_токенов_после_обрезки, было_ли_обрезано).
    Если текст пустой/некорректный — возвращает ("", 0, False).
    """
    if not text:
        return "", 0, False

    limit = max_tokens if max_tokens is not None else MAX_MODEL_TOKENS

    try:
        encoding = _get_encoding()
        tokens = encoding.encode(text)
        if len(tokens) <= limit:
            return text, len(tokens), False
        truncated_tokens = tokens[:limit]
        truncated_text = encoding.decode(truncated_tokens)
        logging.info(
            f"[truncate_to_token_limit] обрезано {len(tokens)} -> {len(truncated_tokens)} токенов"
        )
        return truncated_text, len(truncated_tokens), True
    except Exception as e:
        # Если токенизатор упал по какой-то причине — fallback на грубую
        # обрезку по символам (примерно 3 символа/токен с запасом на
        # не-английские языки).
        logging.warning(f"[truncate_to_token_limit] fallback на обрезку по символам: {e}")
        char_limit = limit * 3
        if len(text) <= char_limit:
            return text, 0, False
        return text[:char_limit], 0, True
