import os
import logging
from typing import Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)

def get_proxy_config() -> Dict[str, str]:
    """
    Возвращает словарь с настройками прокси из переменных окружения.
    """
    use_proxy = os.getenv("USE_PROXY", "true").lower() in ("true", "1", "yes")
    proxy_url = os.getenv("PROXY_URL", "http://u1:p1@185.11.135.75:8888")

    if use_proxy and proxy_url:
        logging.info(f"[PROXY] Proxy enabled: {proxy_url}")
        return {
            "http": proxy_url,
            "https": proxy_url,
        }
    else:
        logging.info("[PROXY] Proxy disabled or not configured")
        return {}

def get_model_kwargs() -> Dict[str, Any]:
    """
    Возвращает kwargs для загрузки моделей (например, AutoModel, AutoTokenizer)
    с учетом настроек прокси.
    """
    proxies = get_proxy_config()
    if proxies:
        return {"proxies": proxies}
    return {}

def get_hf_hub_download_kwargs() -> Dict[str, Any]:
    """
    Возвращает kwargs для hf_hub_download с учетом настроек прокси.
    """
    proxies = get_proxy_config()
    if proxies:
        return {"proxies": proxies}
    return {}

# Устанавливаем прокси для requests и других библиотек, которые читают переменные окружения
_proxies = get_proxy_config()
if _proxies:
    os.environ['HTTP_PROXY'] = _proxies.get("http", "")
    os.environ['HTTPS_PROXY'] = _proxies.get("https", "")
    os.environ['http_proxy'] = _proxies.get("http", "") # Для некоторых библиотек
    os.environ['https_proxy'] = _proxies.get("https", "") # Для некоторых библиотек
else:
    # Убедимся, что переменные прокси очищены, если прокси отключен
    if 'HTTP_PROXY' in os.environ: del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ: del os.environ['HTTPS_PROXY']
    if 'http_proxy' in os.environ: del os.environ['http_proxy']
    if 'https_proxy' in os.environ: del os.environ['https_proxy']
