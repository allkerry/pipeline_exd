import time, logging
from preprocess import preprocess
from translate import translate
from extract_keywords import extract_keywords
from zero_shot import zero_shot
from evaluate_token_count import truncate_to_token_limit, MAX_MODEL_TOKENS
from exorde_data import Translation, Classification, Keywords, Processed, Item, Translated

def process(item: Item, lab_configuration, max_depth_classification) -> Processed:
    t0 = time.perf_counter()
    try:
        item = preprocess(item, False)
        translation: Translation = translate(item, lab_configuration["installed_languages"])
        if translation.translation == "":
            raise ValueError("No content to work with")

        # ─── Фильтрация/обрезка по токенам ──────────────────────
        # Модели в bpipe (DeBERTa zero-shot, RoBERTa emotion/irony/text-type,
        # sentiment-модели) падают или ведут себя непредсказуемо на входах
        # длиннее их max_position_embeddings (обычно 512 токенов). Один
        # длинный текст в батче может уронить обработку всего батча.
        # Поэтому здесь принудительно обрезаем translation до безопасного
        # числа токенов — bpipe.tag() дополнительно подстрахован
        # truncation=True на случай, если что-то просочится.
        truncated_text, n_tokens, was_truncated = truncate_to_token_limit(
            translation.translation, MAX_MODEL_TOKENS
        )
        if was_truncated:
            translation = Translation(
                language=translation.language,
                translation=Translated(truncated_text),
            )
            logging.info(
                f"✂️ Текст обрезан по токенам (лимит={MAX_MODEL_TOKENS}, итог={n_tokens})"
            )
        if not truncated_text.strip():
            raise ValueError("No content to work with after token truncation")

        top_keywords: Keywords = extract_keywords(translation)
        classification: Classification = zero_shot(translation, lab_configuration, max_depth=max_depth_classification)
        logging.info(f"⏱ Обработка: {(time.perf_counter()-t0)*1000:.1f}ms")
        return Processed(item=item, translation=translation, top_keywords=top_keywords, classification=classification)
    except Exception as err:
        logging.info(f"⏱ Обработка (ошибка): {(time.perf_counter()-t0)*1000:.1f}ms")
        raise err
