import logging
import cupy as cp
import torch
import gc  # Для сборки мусора
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from finvader import finvader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from exorde_compat import (
    Classification, LanguageScore, Sentiment, Embedding, TextType,
    Emotion, Irony, Age, Gender, Analysis,
)
from exorde_compat import (
    Classification, LanguageScore, Sentiment, Embedding, TextType,
    Emotion, Irony, Age, Gender, Analysis,
)
from exorde_compat import (
    Classification, LanguageScore, Sentiment, Embedding, TextType,
    Emotion, Irony, Age, Gender, Analysis,
)
from exorde_compat import (
    Classification, LanguageScore, Sentiment, Embedding,
    TextType, Emotion, Irony, Age, Gender, Analysis,
)

logging.basicConfig(level=logging.INFO)

def clear_gpu_memory():
    """Очистка GPU памяти"""
    try:
        # Очистка PyTorch кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Очистка CuPy кэша
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        # Сборка мусора Python
        gc.collect()

        logging.debug("GPU memory cleared")
    except Exception as e:
        logging.warning(f"Error clearing GPU memory: {e}")

def get_gpu_memory_info():
    """Получение информации о GPU памяти"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return allocated, reserved
    return 0, 0

def tag(documents: list[str], lab_configuration):
    # Ensure documents are valid
    assert documents is not None and len(documents) > 0
    logging.info(f"Starting Tagging Batch pipeline for {len(documents)} documents...")

    # Получение информации о памяти в начале
    allocated_start, reserved_start = get_gpu_memory_info()
    logging.info(f"GPU Memory at start - Allocated: {allocated_start:.2f}GB, Reserved: {reserved_start:.2f}GB")

    # Loading models from lab configuration
    models = lab_configuration["models"]

    # Load models
    model = models['sentence_transformer']
    zs_pipe = models['zs_pipe']
    classification_labels = list(lab_configuration["labeldict"].keys())
    sentiment_analyzer = models['sentiment_analyzer']
    fdb_pipe = models['fdb_pipe']
    gdb_pipe = models['gdb_pipe']

    text_classification_models = {
        "Emotion": models['Emotion'],
        "Irony": models['Irony'],
        "TextType": models['TextType']
    }

    # Возвращаем оригинальный размер батча - больший размер работает лучше
    batch_size = 15  # Восстанавливаем оригинальный размер
    logging.info(f"Using batch_size: {batch_size} for {len(documents)} documents")

    # Защита от ошибок на слишком длинных текстах: даже если что-то
    # просочилось мимо обрезки по токенам в upipe (evaluate_token_count),
    # truncation=True здесь не даст HF-пайплайнам упасть с
    # "The expanded size of the tensor..." на входе длиннее лимита модели.
    # max_length=512 — стандартный потолок для всех используемых здесь
    # BERT/RoBERTa/DeBERTa моделей.
    HF_SAFETY_KWARGS = {"truncation": True, "max_length": 512}

    try:
        # Embeddings в batch с проверкой памяти
        logging.info("Processing embeddings...")
        # SentenceTransformer сам обрезает по model.max_seq_length, но
        # выставляем это явно на случай нестандартного чекпоинта.
        try:
            if getattr(model, "max_seq_length", None) and model.max_seq_length > 512:
                model.max_seq_length = 512
        except Exception:
            pass
        embedding_vectors = model.encode(
            documents,
            convert_to_tensor=True,
            device='cuda',
            batch_size=batch_size,
            show_progress_bar=False  # Отключаем progress bar для экономии памяти
        )
        embedding_vectors = embedding_vectors.cpu().numpy()

        # Принудительная очистка после embeddings
        clear_gpu_memory()
        allocated_after_emb, _ = get_gpu_memory_info()
        logging.info(f"GPU Memory after embeddings: {allocated_after_emb:.2f}GB")

        # Zero-shot classification в batch.
        # ВАЖНО: ZeroShotClassificationPipeline в разных версиях transformers
        # по-разному принимает truncation/max_length как kwargs при вызове
        # (в части версий это даёт TypeError) — поэтому здесь НЕ передаём
        # HF_SAFETY_KWARGS явно. Защита от переполнения обеспечивается
        # tokenizer.model_max_length=512, выставленным в lab_initialization.py,
        # плюс упреждающей обрезкой текста в upipe (evaluate_token_count).
        classification_results = zs_pipe(documents, candidate_labels=classification_labels, batch_size=batch_size)
        clear_gpu_memory()

        # Text classification models в batch
        logging.info("Processing text classification...")
        text_type_results = text_classification_models['TextType'](documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        clear_gpu_memory()

        emotion_results = text_classification_models['Emotion'](documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        clear_gpu_memory()

        irony_results = text_classification_models['Irony'](documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        clear_gpu_memory()

        # Sentiment analysis в batch
        logging.info("Processing sentiment analysis...")
        fdb_predictions = fdb_pipe(documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        clear_gpu_memory()

        gdb_predictions = gdb_pipe(documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        clear_gpu_memory()

        # VADER и FinVADER (CPU операции)
        logging.info("Processing VADER sentiment...")
        vader_scores = [sentiment_analyzer.polarity_scores(text)["compound"] for text in documents]
        finvader_scores = [finvader(text, use_sentibignomics=True, use_henry=True, indicator='compound') for text in documents]

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"CUDA OOM Error during model inference: {e}")
        clear_gpu_memory()
        # Возвращаем пустые результаты или fallback
        return []
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        clear_gpu_memory()
        return []

    _out = []
    logging.info("Building analysis results...")

    for idx, text in enumerate(documents):
        try:
            # Embedding
            embedding_vector = embedding_vectors[idx]
            embedding = Embedding(list(embedding_vector.astype(float)))

            # Classification
            classification_result = classification_results[idx]
            top_label = classification_result["labels"][0]
            top_score = round(classification_result["scores"][0], 4)
            classification = Classification(label=top_label, score=top_score)

            # Text Type
            text_type_result = [(y["label"], float(y["score"])) for y in text_type_results[idx]]
            types = {item[0]: item[1] for item in text_type_result}
            text_type = TextType(
                assumption=types.get("Assumption", 0.0),
                anecdote=types.get("Anecdote", 0.0),
                none=types.get("None", 0.0),
                definition=types.get("Definition", 0.0),
                testimony=types.get("Testimony", 0.0),
                other=types.get("Other", 0.0),
                study=types.get("Statistics/Study", 0.0),
            )

            # Emotion
            emotion_result = [(y["label"], float(y["score"])) for y in emotion_results[idx]]
            emotions = {item[0]: item[1] for item in emotion_result}
            emotions = {k: round(v, 4) for k, v in emotions.items()}
            emotion = Emotion(
                love=emotions.get("love", 0.0),
                admiration=emotions.get("admiration", 0.0),
                joy=emotions.get("joy", 0.0),
                approval=emotions.get("approval", 0.0),
                caring=emotions.get("caring", 0.0),
                excitement=emotions.get("excitement", 0.0),
                gratitude=emotions.get("gratitude", 0.0),
                desire=emotions.get("desire", 0.0),
                anger=emotions.get("anger", 0.0),
                optimism=emotions.get("optimism", 0.0),
                disapproval=emotions.get("disapproval", 0.0),
                grief=emotions.get("grief", 0.0),
                annoyance=emotions.get("annoyance", 0.0),
                pride=emotions.get("pride", 0.0),
                curiosity=emotions.get("curiosity", 0.0),
                neutral=emotions.get("neutral", 0.0),
                disgust=emotions.get("disgust", 0.0),
                disappointment=emotions.get("disappointment", 0.0),
                realization=emotions.get("realization", 0.0),
                fear=emotions.get("fear", 0.0),
                relief=emotions.get("relief", 0.0),
                confusion=emotions.get("confusion", 0.0),
                remorse=emotions.get("remorse", 0.0),
                embarrassment=emotions.get("embarrassment", 0.0),
                surprise=emotions.get("surprise", 0.0),
                sadness=emotions.get("sadness", 0.0),
                nervousness=emotions.get("nervousness", 0.0),
            )

            # Irony
            irony_result = [(y["label"], float(y["score"])) for y in irony_results[idx]]
            ironies = {item[0]: item[1] for item in irony_result}
            irony = Irony(
                irony=ironies.get("irony", 0.0),
                non_irony=ironies.get("non_irony", 0.0)
            )

            # Sentiments
            vader_sent_score = round(vader_scores[idx], 2)
            fin_vader_sent_score = round(finvader_scores[idx], 2)

            fdb_prediction = fdb_predictions[idx]
            fdb_sentiment_dict = {e["label"]: round(e["score"], 3) for e in fdb_prediction}
            fdb_sent_score = round(fdb_sentiment_dict.get("positive", 0.0) - fdb_sentiment_dict.get("negative", 0.0), 3)

            gdb_prediction = gdb_predictions[idx]
            gdb_sentiment_dict = {e["label"]: round(e["score"], 3) for e in gdb_prediction}
            gdb_sent_score = round(gdb_sentiment_dict.get("positive", 0.0) - gdb_sentiment_dict.get("negative", 0.0), 3)

            compounded_fin_sentiment = round((0.70 * fdb_sent_score + 0.30 * fin_vader_sent_score), 2)

            # Compute the compounded sentiment score based on thresholds
            if abs(compounded_fin_sentiment) >= 0.6:
                sentiment_score = round((0.30 * gdb_sent_score + 0.10 * vader_sent_score + 0.60 * compounded_fin_sentiment), 2)
            elif abs(compounded_fin_sentiment) >= 0.4:
                sentiment_score = round((0.40 * gdb_sent_score + 0.20 * vader_sent_score + 0.40 * compounded_fin_sentiment), 2)
            elif abs(compounded_fin_sentiment) >= 0.1:
                sentiment_score = round((0.60 * gdb_sent_score + 0.25 * vader_sent_score + 0.15 * compounded_fin_sentiment), 2)
            else:
                sentiment_score = round((0.60 * gdb_sent_score + 0.40 * vader_sent_score), 2)

            sentiment = Sentiment(sentiment_score)

            # Mock gender (since the model is untrained)
            gender = Gender(male=0.5, female=0.5)

            # Age (untrained model — uniform distribution, sums to 1.0)
            age = Age(
                below_twenty=0.25,
                twenty_thirty=0.25,
                thirty_forty=0.25,
                forty_more=0.25
            )

            # Language score (untrained model)
            language_score = LanguageScore(1.0)

            # Compile analysis
            analysis = Analysis(
                classification=classification,
                language_score=language_score,
                sentiment=sentiment,
                embedding=embedding,
                gender=gender,
                text_type=text_type,
                emotion=emotion,
                irony=irony,
                age=age,
            )

            _out.append(analysis)

        except Exception as e:
            logging.error(f"Error processing document {idx}: {e}")
            # Создаем fallback analysis
            _out.append(create_fallback_analysis())

    # Финальная очистка памяти
    clear_gpu_memory()
    allocated_end, _ = get_gpu_memory_info()
    logging.info(f"GPU Memory at end: {allocated_end:.2f}GB")
    logging.info(f"Completed processing {len(_out)} documents")

    return _out

def create_fallback_analysis():
    """Создает базовый анализ в случае ошибок"""
    return Analysis(
        classification=Classification(label="other", score=0.5),
        language_score=LanguageScore(1.0),
        sentiment=Sentiment(0.0),
        embedding=Embedding([0.0] * 384),  # Размер для all-MiniLM-L6-v2
        gender=Gender(male=0.5, female=0.5),
        text_type=TextType(assumption=0.0, anecdote=0.0, none=1.0, definition=0.0, testimony=0.0, other=0.0, study=0.0),
        emotion=Emotion(love=0.0, admiration=0.0, joy=0.0, approval=0.0, caring=0.0, excitement=0.0, gratitude=0.0, desire=0.0, anger=0.0, optimism=0.0, disapproval=0.0, grief=0.0, annoyance=0.0, pride=0.0, curiosity=0.0, neutral=1.0, disgust=0.0, disappointment=0.0, realization=0.0, fear=0.0, relief=0.0, confusion=0.0, remorse=0.0, embarrassment=0.0, surprise=0.0, sadness=0.0, nervousness=0.0),
        irony=Irony(irony=0.0, non_irony=1.0),
        age=Age(below_twenty=0.25, twenty_thirty=0.25, thirty_forty=0.25, forty_more=0.25),
    )
