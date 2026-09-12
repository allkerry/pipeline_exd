import asyncio
import itertools
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import orjson
from aiohttp import web

sys.path.insert(0, os.path.dirname(__file__))

from exorde_data import Item, CreatedAt, Content, Domain, Url, Title, ExternalId, Author, ExternalParentId
from process import process
from translate import NonEnglishError
from lab_initialization import lab_initialization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [upipe] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

UPIPE_PORT   = int(os.getenv("UPIPE_PORT", "5981"))
WORKERS      = int(os.getenv("UPIPE_WORKERS", "4"))
QUEUE_LIMIT  = int(os.getenv("UPIPE_QUEUE_LIMIT", "200"))
MAX_DEPTH_CLASSIFICATION = int(os.getenv("MAX_DEPTH_CLASSIFICATION", "2"))

def _parse_bpipe_urls() -> list[str]:
    urls_env = os.getenv("BPIPE_URLS", "")
    if urls_env:
        return [u.strip() for u in urls_env.split(",") if u.strip()]
    single = os.getenv("BPIPE_URL", "http://127.0.0.1:7995/")
    return [single]

BPIPE_URLS: list[str] = _parse_bpipe_urls()

_session: aiohttp.ClientSession | None = None
_lab_config: dict | None = None
_process_queue: asyncio.Queue | None = None
_thread_pool: ThreadPoolExecutor | None = None

_bpipe_cycle: itertools.cycle | None = None
_bpipe_lock = asyncio.Lock()

_stats = {"received": 0, "processed": 0, "forwarded": 0, "errors": 0, "dropped": 0, "filtered_lang": 0}


def _process_sync(item: Item, lab_config: dict) -> dict:
    processed = process(item, lab_config, MAX_DEPTH_CLASSIFICATION)
    return {
        "item": {
            "created_at":         str(processed.item.get("created_at", "")),
            "title":              str(processed.item.get("title", "")),
            "content":            str(processed.item.get("content", "")),
            "domain":             str(processed.item.get("domain", "")),
            "url":                str(processed.item.get("url", "")),
            "external_id":        str(processed.item.get("external_id", "")),
            "external_parent_id": str(processed.item.get("external_parent_id", "")),
            "author":             str(processed.item.get("author", "")),
            "username":           str(processed.item.get("username", "")) if processed.item.get("username") else "",
        },
        "translation": {
            "language":    str(processed.translation.language),
            "translation": str(processed.translation.translation),
        },
        "top_keywords": list(processed.top_keywords),
        "classification": {
            "label": str(processed.classification.label),
            "score": float(processed.classification.score),
        },
    }


async def forward_to_bpipe(payload: dict) -> bool:
    global _session, _bpipe_cycle
    if _session is None or _bpipe_cycle is None:
        return False

    async with _bpipe_lock:
        url = next(_bpipe_cycle)

    try:
        async with _session.post(
            url,
            data=orjson.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                log.warning(f"⚠️  bpipe {url} вернул {resp.status}")
            return ok
    except aiohttp.ClientConnectorError:
        log.error(f"❌ bpipe недоступен: {url}")
        return False
    except Exception as e:
        log.debug(f"Ошибка отправки в bpipe: {e}")
        return False


async def worker_loop(worker_id: int):
    global _stats
    log.info(f"👷 Upipe-воркер #{worker_id} запущен")

    loop = asyncio.get_event_loop()
    while True:
        try:
            item, raw_item = await _process_queue.get()

            try:
                payload = await loop.run_in_executor(
                    _thread_pool,
                    _process_sync,
                    item,
                    _lab_config,
                )
                _stats["processed"] += 1

                if raw_item.get("username"):
                    payload["item"]["username"] = raw_item["username"]

                ok = await forward_to_bpipe(payload)
                if ok:
                    _stats["forwarded"] += 1
                    log.debug(
                        f"✅ [{worker_id}] → bpipe | {raw_item.get('url', '')[:60]}"
                    )
                else:
                    _stats["errors"] += 1

            except NonEnglishError as e:
                # Раньше логировалось на DEBUG при работающем на INFO сервисе —
                # дропы происходили без единого следа в логах. Теперь INFO.
                _stats["filtered_lang"] += 1
                log.info(f"🌐 [{worker_id}] Не-английский текст отброшен: {e}")

            except Exception as e:
                _stats["errors"] += 1
                log.warning(f"⚠️ [{worker_id}] Ошибка обработки: {e}")

            finally:
                _process_queue.task_done()

            if _stats["forwarded"] % 25 == 0 and _stats["forwarded"] > 0:
                log.info(
                    f"📊 recv={_stats['received']} proc={_stats['processed']} "
                    f"fwd={_stats['forwarded']} err={_stats['errors']} "
                    f"lang={_stats['filtered_lang']} dropped={_stats['dropped']}"
                )

        except Exception as e:
            log.error(f"Ошибка воркера #{worker_id}: {e}")
            await asyncio.sleep(1)


async def handle_receive_item(request: web.Request) -> web.Response:
    global _stats
    try:
        raw_item = await request.json()
    except Exception as e:
        return web.Response(text=f"bad json: {e}", status=400)

    _stats["received"] += 1

    try:
        external_parent_id = raw_item.get("external_parent_id") or ""
        item = Item(
            created_at=CreatedAt(raw_item["created_at"]),
            title=Title(raw_item.get("title", "")),
            content=Content(raw_item["content"]),
            domain=Domain(raw_item["domain"]),
            url=Url(raw_item["url"]),
            external_id=ExternalId(raw_item.get("external_id", "")),
            external_parent_id=ExternalParentId(external_parent_id),
            author=Author(raw_item.get("author", "")),
        )
    except Exception as e:
        log.warning(f"Ошибка создания Item: {e} | данные: {raw_item}")
        return web.Response(text="invalid_item", status=400)

    if _process_queue.qsize() >= QUEUE_LIMIT:
        _stats["dropped"] += 1
        if _stats["dropped"] % 10 == 0:
            log.warning(
                f"🗑️  Очередь переполнена — дропнуто элементов: {_stats['dropped']} "
                f"(queue={_process_queue.qsize()}/{QUEUE_LIMIT})"
            )
        return web.Response(text="queue_full", status=503)

    await _process_queue.put((item, raw_item))
    return web.Response(text="received")


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "queue":  _process_queue.qsize() if _process_queue else 0,
        "bpipe_urls": BPIPE_URLS,
        "stats":  _stats,
    })


async def on_startup(app: web.Application):
    global _session, _lab_config, _process_queue, _thread_pool, _bpipe_cycle

    log.info("🔬 Инициализация upipe...")
    log.info(f"   bpipe инстансов: {len(BPIPE_URLS)} → {BPIPE_URLS}")

    loop = asyncio.get_event_loop()
    _thread_pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="upipe_worker")
    _lab_config = await loop.run_in_executor(None, lab_initialization)

    _process_queue = asyncio.Queue()
    _bpipe_cycle   = itertools.cycle(BPIPE_URLS)

    connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=60)
    _session = aiohttp.ClientSession(connector=connector)

    for i in range(WORKERS):
        asyncio.create_task(worker_loop(i))

    log.info(f"✅ Upipe запущен на порту {UPIPE_PORT} ({WORKERS} воркеров)")
    log.info(f"   Round-robin → {BPIPE_URLS}")


async def on_shutdown(app: web.Application):
    global _session, _thread_pool
    if _session:
        await _session.close()
    if _thread_pool:
        _thread_pool.shutdown(wait=False)
    log.info(f"📊 Итог upipe: {_stats}")


app = web.Application(client_max_size=50 * 1024 * 1024)
app.router.add_post("/", handle_receive_item)
app.router.add_get("/", handle_health)
app.router.add_get("/health", handle_health)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=UPIPE_PORT, print=None)
