import logging
import os
import tiktoken

_ENCODING_NAME = "r50k_base"
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoding


def evaluate_token_count(item_content_string: str, encoding_name: str = None) -> int:
    try:
        if item_content_string is None or len(item_content_string) <= 1:
            return 0
        encoding = _get_encoding() if encoding_name is None else tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(item_content_string))
    except Exception as e:
        logging.info(f"[evaluate_token_count] error: {e}")
        num_tokens = 0
    return num_tokens


MAX_MODEL_TOKENS = int(os.getenv("MAX_MODEL_TOKENS", "400"))
