import logging

# translate.py и zero_shot.py — стабы, ни один не использует ML-модели.
# extract_keywords.py — чистый YAKE, без transformers/torch.
# Поэтому здесь больше не грузятся zs_pipe/sentence_transformer/Emotion/Irony/
# TextType/fdb/gdb/bert_tokenizer/labeldict — все они были мёртвым весом
# (память, время старта, скачивание с HuggingFace) без единого вызова.
# Если стабы translate/zero_shot заменят на реальную логику — модели нужно
# будет вернуть сюда.


def lab_initialization():
    logging.info("[LAB INITIALIZATION] upipe: ML-модели не грузятся (translate/zero_shot — стабы)")
    return {
        "labeldict": {},
        "device": -1,
        "mappings": {},
        "max_depth": 2,
        "remove_stopwords": False,
        "installed_languages": [],
        "models": {},
    }
