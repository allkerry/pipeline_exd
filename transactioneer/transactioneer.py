"""
Transactioneer (standalone) — получает обработанные батчи от bpipe
и загружает их на Exorde Upload API.

Слушает: POST /commit на TRANSACTIONEER_PORT (по умолчанию 8002)

ВАЖНО: установи MAIN_ADDRESS в .env (твой Exorde-кошелёк)
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

import aiohttp
from aiohttp import web, ClientSession, FormData, ClientTimeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [transactioneer] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Конфигурация ─────────────────────────────────────────────
TRANSACTIONEER_PORT = int(os.getenv("TRANSACTIONEER_PORT", "8002"))
UPLOAD_API_URL      = os.getenv("UPLOAD_API_URL", "http://upload.exorde.network/v1/upload")
MAIN_ADDRESS        = os.getenv("MAIN_ADDRESS", "")
UPLOAD_API_TIMEOUT  = int(os.getenv("UPLOAD_API_TIMEOUT", "120"))
USE_PROXY           = os.getenv("USE_PROXY", "false").lower() == "true"
PROXY_URL           = os.getenv("PROXY_URL", "")
MAX_RETRIES         = int(os.getenv("UPLOAD_MAX_RETRIES", "3"))

# ─── Статистика ───────────────────────────────────────────────
_stats = {
    "batches_received": 0,
    "items_received": 0,
    "uploads_success": 0,
    "uploads_failed": 0,
    "items_accepted": 0,
}
_active_tasks: list = []


# ─── Загрузка на Upload API ────────────────────────────────────

async def upload_to_api(items: list, main_address: str) -> Optional[dict]:
    """Загружает батч на Exorde Upload API. Возвращает ответ API или None."""

    batch_dict = {"items": items, "kind": "SPOTTING"}
    batch_json = json.dumps(batch_dict, default=str, ensure_ascii=False)

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                delay = 2 ** attempt
                log.info(f"🔄 Повторная попытка {attempt+1}/{MAX_RETRIES} через {delay}с...")
                await asyncio.sleep(delay)

            form = FormData()
            form.add_field(
                "file",
                batch_json,
                filename=f"batch_{int(time.time())}.json",
                content_type="application/json",
            )

            connector_kwargs = {}
            if USE_PROXY and PROXY_URL:
                try:
                    from aiohttp_socks import ProxyConnector
                    connector_kwargs["connector"] = ProxyConnector.from_url(PROXY_URL)
                except ImportError:
                    log.warning("aiohttp_socks не установлен, прокси игнорируется")

            async with ClientSession(**connector_kwargs) as session:
                async with session.post(
                    UPLOAD_API_URL,
                    data=form,
                    headers={"MAIN_ADDRESS": main_address},
                    timeout=ClientTimeout(total=UPLOAD_API_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        response_data = await resp.json()
                        file_id       = response_data.get("file_id", "?")
                        total_items   = response_data.get("total_items", 0)
                        filtered      = response_data.get("filtered_items", 0)
                        rejected      = response_data.get("rejected_items", 0)
                        duplicates    = response_data.get("duplicate_items", 0)
                        proc_ms       = response_data.get("processing_time_ms", 0)

                        log.info(
                            f"✅ Загружено | file_id={file_id} | "
                            f"отправлено={len(items)} принято={filtered} "
                            f"отклонено={rejected} дубликаты={duplicates} | "
                            f"API обработка={proc_ms}ms"
                        )

                        # Подробное логирование отклонённых items
                        if rejected > 0:
                            rejection_details = response_data.get("rejection_details", [])
                            rejected_urls     = response_data.get("rejected_urls", [])
                            rejected_reasons  = response_data.get("rejected_reasons", [])
                            errors            = response_data.get("errors", [])
                            log.warning(f"⚠️ Детали отклонения: {rejection_details or rejected_urls or rejected_reasons or errors or 'нет деталей в ответе'}")
                            log.warning(f"⚠️ Полный ответ API: {json.dumps(response_data, ensure_ascii=False)[:2000]}")
                            # Сохраняем первый отклонённый батч для анализа
                            try:
                                import os
                                debug_path = f"/tmp/rejected_batch_{file_id}.json"
                                with open(debug_path, "w") as dbf:
                                    json.dump({"items": items[:3], "kind": "SPOTTING"}, dbf, default=str, ensure_ascii=False, indent=2)
                                log.warning(f"⚠️ Первые 3 items сохранены в {debug_path}")
                            except Exception as de:
                                log.warning(f"Не удалось сохранить debug файл: {de}")

                        return response_data
                    else:
                        err = await resp.text()
                        log.error(f"❌ HTTP {resp.status}: {err[:300]}")

        except asyncio.TimeoutError:
            log.error(f"⏱️ Таймаут (попытка {attempt+1}/{MAX_RETRIES})")
        except Exception as e:
            log.exception(f"❌ Ошибка при загрузке (попытка {attempt+1}/{MAX_RETRIES}): {e}")

    log.error(f"❌ Все {MAX_RETRIES} попытки загрузки исчерпаны")
    return None


async def _upload_task(items: list, item_count: int):
    """Фоновая задача загрузки."""
    result = await upload_to_api(items, MAIN_ADDRESS)
    if result:
        _stats["uploads_success"] += 1
        _stats["items_accepted"] += result.get("filtered_items", 0)
    else:
        _stats["uploads_failed"] += 1


# ─── HTTP обработчики ──────────────────────────────────────────

async def handle_commit(request: web.Request) -> web.Response:
    global _active_tasks
    try:
        items = await request.json()
    except Exception as e:
        return web.Response(text=f"bad json: {e}", status=400)

    if not isinstance(items, list):
        return web.Response(text="expected array", status=400)

    item_count = len(items)
    _stats["batches_received"] += 1
    _stats["items_received"] += item_count

    log.info(f"📥 Батч получен: {item_count} элементов")

    # Очищаем завершённые задачи
    _active_tasks = [t for t in _active_tasks if not t.done()]

    task = asyncio.create_task(_upload_task(items, item_count))
    _active_tasks.append(task)

    return web.Response(text="received", status=200)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "main_address": MAIN_ADDRESS or "NOT SET",
        "upload_api": UPLOAD_API_URL,
        "stats": _stats,
        "active_tasks": len(_active_tasks),
    })


async def on_startup(app: web.Application):
    if not MAIN_ADDRESS:
        log.error("❌ MAIN_ADDRESS не задан! Установи его в .env файле")
        log.error("   MAIN_ADDRESS=0x... (твой Exorde кошелёк)")
    else:
        log.info(f"✅ Transactioneer запущен | address={MAIN_ADDRESS[:10]}...")

    log.info(f"   Upload API: {UPLOAD_API_URL}")
    log.info(f"   Порт: {TRANSACTIONEER_PORT}")


app = web.Application(client_max_size=500 * 1024 * 1024)
app.router.add_post("/commit", handle_commit)
app.router.add_get("/", handle_health)
app.router.add_get("/health", handle_health)
app.on_startup.append(on_startup)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=TRANSACTIONEER_PORT, print=None)
