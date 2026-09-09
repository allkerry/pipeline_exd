import json
import torch
import requests
import logging
import gc  # Для сборки мусора
import os  # Для переменных окружения
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from argostranslate import translate as _translate
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from finvader import finvader
from huggingface_hub import hf_hub_download, snapshot_download

def clear_gpu_memory():
    """Очистка GPU памяти"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logging.debug("GPU memory cleared during model loading")
    except Exception as e:
        logging.warning(f"Error clearing GPU memory: {e}")

def initialize_models(device):
    logging.info("[TAGGING] Initializing models to be pre-ready for batch processing:")
    models = {}

    # Настройка прокси для загрузки моделей из переменных окружения
    USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
    PROXY_URL = os.getenv("PROXY_URL", "")

    proxies = {}
    if USE_PROXY and PROXY_URL:
        proxies = {
            "http": PROXY_URL,
            "https": PROXY_URL
        }
        logging.info(f"[TAGGING] Используем прокси для загрузки моделей: {PROXY_URL}")
        # Устанавливаем прокси для requests (используется в HuggingFace)
        os.environ['HTTP_PROXY'] = PROXY_URL
        os.environ['HTTPS_PROXY'] = PROXY_URL
    else:
        logging.info("[TAGGING] Прокси не используется для загрузки моделей")

    # Настройки для экономии памяти (убираем low_cpu_mem_usage для совместимости)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "torch_dtype": torch_dtype
    }

    try:
        # Zero-shot classification model
        logging.info("[TAGGING] Loading model: MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33")
        models['zs_pipe'] = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
            device=device,
            model_kwargs=model_kwargs
        )
        clear_gpu_memory()

        # Sentence Transformer model for embeddings
        logging.info("[TAGGING] Loading model: sentence-transformers/all-MiniLM-L6-v2")
        models['sentence_transformer'] = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=f"cuda:{device}" if device >= 0 else "cpu"
        )
        # Включаем half precision для экономии памяти
        if device >= 0:
            models['sentence_transformer'].half()
        clear_gpu_memory()

        # Text classification models с меньшими batch размерами
        text_classification_models = [
            ("Emotion", "SamLowe/roberta-base-go_emotions"),
            ("Irony", "cardiffnlp/twitter-roberta-base-irony"),
            ("TextType", "marieke93/MiniLM-evidence-types"),
        ]

        for col_name, model_name in text_classification_models:
            logging.info(f"[TAGGING] Loading model: {model_name}")

            # Некоторые модели не имеют safetensors на main ветке,
            # только .bin — что заблокировано transformers после CVE-2025-32434.
            # Грузим с PR веток где safetensors есть.
            SNAPSHOT_REVISIONS = {
                "cardiffnlp/twitter-roberta-base-irony": "refs/pr/3",
                "marieke93/MiniLM-evidence-types":       "refs/pr/1",
            }
            if model_name in SNAPSHOT_REVISIONS:
                local_path = snapshot_download(
                    repo_id=model_name,
                    revision=SNAPSHOT_REVISIONS[model_name]
                )
                load_name = local_path
            else:
                load_name = model_name

            models[col_name] = pipeline(
                "text-classification",
                model=load_name,
                device=device,
                top_k=None,
                max_length=512,
                padding=True,
                model_kwargs=model_kwargs
            )
            clear_gpu_memory()

        # BERT tokenizer (CPU only)
        logging.info("[TAGGING] Loading tokenizer: bert-large-uncased")
        models['bert_tokenizer'] = AutoTokenizer.from_pretrained("bert-large-uncased")

        # VADER sentiment analyzer (CPU only)
        logging.info("[TAGGING] Loading model: vaderSentiment")
        models['sentiment_analyzer'] = SentimentIntensityAnalyzer()

        # Add finvader_analyzer to models (CPU only)
        logging.info("[TAGGING] Loading finvader analyzer")
        models['finvader_analyzer'] = finvader

        # Load additional lexicons for VADER sentiment analyzer
        try:
            emoji_lexicon = hf_hub_download(
                repo_id="ExordeLabs/SentimentDetection",
                filename="emoji_unic_lexicon.json",
            )
            loughran_dict = hf_hub_download(
                repo_id="ExordeLabs/SentimentDetection", filename="loughran_dict.json"
            )
            logging.info("[TAGGING] Loading Loughran_dict & unic_emoji_dict for sentiment_analyzer.")
            with open(emoji_lexicon) as f:
                unic_emoji_dict = json.load(f)
            with open(loughran_dict) as f:
                Loughran_dict = json.load(f)
            models['sentiment_analyzer'].lexicon.update(Loughran_dict)
            models['sentiment_analyzer'].lexicon.update(unic_emoji_dict)
        except Exception as e:
            logging.info("[TAGGING] Error loading Loughran_dict & unic_emoji_dict for sentiment_analyzer. Proceeding without them.")

        # Financial DistilBERT model for sentiment analysis
        logging.info("[TAGGING] Loading model: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
        models['fdb_tokenizer'] = AutoTokenizer.from_pretrained(
            "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        )
        models['fdb_model'] = AutoModelForSequenceClassification.from_pretrained(
            "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            **model_kwargs
        )
        models['fdb_pipe'] = pipeline(
            "text-classification",
            model=models['fdb_model'],
            tokenizer=models['fdb_tokenizer'],
            device=device,
            top_k=None,
            max_length=512,
            padding=True,
        )
        clear_gpu_memory()

        # General DistilBERT model for sentiment analysis
        logging.info("[TAGGING] Loading model: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
        models['gdb_tokenizer'] = AutoTokenizer.from_pretrained(
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        )
        models['gdb_model'] = AutoModelForSequenceClassification.from_pretrained(
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            **model_kwargs
        )
        models['gdb_pipe'] = pipeline(
            "text-classification",
            model=models['gdb_model'],
            tokenizer=models['gdb_tokenizer'],
            device=device,
            top_k=None,
            max_length=512,
            padding=True,
        )
        clear_gpu_memory()

        logging.info("[TAGGING] Models loaded successfully.")

        # Финальная информация о GPU памяти
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logging.info(f"[TAGGING] GPU Memory after model loading - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"[TAGGING] CUDA OOM during model loading: {e}")
        logging.error("[TAGGING] Try reducing number of models or using CPU fallback")
        raise
    except Exception as e:
        logging.error(f"[TAGGING] Error loading models: {e}")
        raise

    return models

def lab_initialization():
    # Determine device (GPU or CPU)
    device = 0 if torch.cuda.is_available() else -1
    logging.info(f"[LAB INITIALIZATION] Using device: {'GPU' if device != -1 else 'CPU'}")

    # Настройка PyTorch для экономии памяти
    if torch.cuda.is_available():
        # Включаем TensorFloat-32 для ускорения и экономии памяти
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Настройка аллокатора памяти
        torch.cuda.empty_cache()
        logging.info(f"[LAB INITIALIZATION] GPU device: {torch.cuda.get_device_name(0)}")
        logging.info(f"[LAB INITIALIZATION] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    mappings = {
        "Gender": {0: "Female", 1: "Male"},
        "Age": {0: "<20", 1: "20<30", 2: "30<40", 3: ">=40"},
    }

    # Initialize models
    models = initialize_models(device)

    # Load classification labels
    try:
        # Используем прокси если настроен
        USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
        PROXY_URL = os.getenv("PROXY_URL", "")

        proxies = {}
        if USE_PROXY and PROXY_URL:
            proxies = {
                "http": PROXY_URL,
                "https": PROXY_URL
            }

        labels = requests.get(
            "https://raw.githubusercontent.com/exorde-labs/TestnetProtocol/main/targets/class_names.json",
            proxies=proxies if proxies else None,
            timeout=30
        ).json()
        logging.info("[LAB INITIALIZATION] Classification labels loaded successfully")
    except Exception as e:
        logging.error(f"[LAB INITIALIZATION] Error fetching labels: {e}")
        labels = {}  # Fallback to empty labels or handle accordingly

    # Return the lab configuration
    lab_configuration = {
        "labeldict": labels,
        "device": device,
        "mappings": mappings,
        "max_depth": 2,
        "remove_stopwords": False,
        "installed_languages": [],
        "models": models
    }

    logging.info("[LAB INITIALIZATION] Lab configuration initialized successfully.")
    return lab_configuration
