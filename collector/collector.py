import asyncio
import hashlib
import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone

import aiohttp
import orjson
from aiohttp import web
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [collector] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

UPIPE_URL       = os.getenv("UPIPE_URL", "http://127.0.0.1:5981/")
COLLECTOR_PORT  = int(os.getenv("COLLECTOR_PORT", "9000"))
MIN_TEXT_LEN         = int(os.getenv("MIN_TEXT_LEN", "20"))
MAX_TEXT_LEN         = int(os.getenv("MAX_TEXT_LEN", "20000"))
MAX_OLDNESS_SECONDS = int(os.getenv("MAX_OLDNESS_SECONDS", "86400"))
LANG_FILTER          = os.getenv("LANG_FILTER", "en").strip().lower()

_seen_ids: OrderedDict = OrderedDict()
_DEDUP_MAX_SIZE = 100_000

_stats = {"received": 0, "forwarded": 0, "filtered_old": 0, "filtered_dup": 0, "filtered_lang": 0, "truncated": 0, "errors": 0, "dropped": 0}
_session: aiohttp.ClientSession | None = None


def _hash_author(author: str) -> str:
    if not author:
        return ""
    return hashlib.sha1(author.encode("utf-8")).hexdigest()


async def forward_to_upipe(item: dict) -> bool:
    global _session
    if _session is None:
        return False
    try:
        async with _session.post(UPIPE_URL, data=orjson.dumps(item),
                                  headers={"Content-Type": "application/json"}) as resp:
            if 200 <= resp.status < 300:
                return True
            log.warning(f"upipe ответил {resp.status}")
            return False
    except aiohttp.ClientConnectorError:
        log.error(f"❌ upipe недоступен: {UPIPE_URL}")
        return False
    except Exception as e:
        log.debug(f"Ошибка пересылки: {e}")
        return False


def _parse_created_at(created_at_str: str) -> datetime | None:
    if not created_at_str:
        return None
    try:
        s = created_at_str.rstrip("Z")
        if "." in s:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _passes_lang_filter(content: str) -> bool:
    if not LANG_FILTER:
        return True
    try:
        return detect(content) == LANG_FILTER
    except LangDetectException:
        return False


async def handle_store_item(request: web.Request) -> web.Response:
    global _stats
    try:
        item = await request.json()
    except Exception as e:
        return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

    _stats["received"] += 1
    content = item.get("content", "")

    item["author"] = _hash_author(item.get("author", ""))

    ext_id = item.get("external_id", "")
    if ext_id:
        if ext_id in _seen_ids:
            _stats["filtered_dup"] += 1
            log.debug(f"♻️ Дубликат: {ext_id}")
            return web.json_response({"message": "duplicate"}, status=200)
        _seen_ids[ext_id] = True
        _seen_ids.move_to_end(ext_id)
        if len(_seen_ids) > _DEDUP_MAX_SIZE:
            for _ in range(_DEDUP_MAX_SIZE // 2):
                _seen_ids.popitem(last=False)

    if len(content) < MIN_TEXT_LEN:
        return web.json_response({"message": "skipped_short"}, status=200)

    # Обрезаем ДО lang-detect: langdetect на очень длинных/повторяющихся
    # текстах (>MAX_TEXT_LEN) даёт ложные срабатывания не-en языка.
    if len(content) > MAX_TEXT_LEN:
        content = content[:MAX_TEXT_LEN]
        item["content"] = content
        _stats["truncated"] = _stats.get("truncated", 0) + 1

    if not _passes_lang_filter(content):
        _stats["filtered_lang"] += 1
        return web.json_response({"message": "filtered_lang"}, status=200)

    created_at_str = item.get("created_at", "")
    tweet_dt = _parse_created_at(created_at_str)
    if tweet_dt is not None:
        age_seconds = (datetime.now(timezone.utc) - tweet_dt).total_seconds()
        if age_seconds > MAX_OLDNESS_SECONDS:
            _stats["filtered_old"] += 1
            log.debug(f"🕒 Устарел ({age_seconds/3600:.1f}ч): {content[:60]}")
            return web.json_response({"message": "filtered_old"}, status=200)
    else:
        log.warning(f"⚠️ Не удалось распарсить created_at: {created_at_str!r}")

    ok = await forward_to_upipe(item)
    if ok:
        _stats["forwarded"] += 1
        if _stats["forwarded"] % 50 == 0:
            log.info(
                f"📊 recv={_stats['received']} fwd={_stats['forwarded']} "
                f"old={_stats['filtered_old']} dup={_stats['filtered_dup']} lang={_stats['filtered_lang']} dropped={_stats['dropped']}"
            )
    else:
        _stats["errors"] += 1
        _stats["dropped"] += 1
        if _stats["dropped"] % 10 == 0:
            log.warning(
                f"⚠️  Потеряно элементов (upipe недоступен/503): {_stats['dropped']}"
            )

    return web.json_response({"message": "OK"}, status=200)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "stats": _stats,
        "upipe": UPIPE_URL,
        "lang_filter": LANG_FILTER or None,
    })


async def on_startup(app: web.Application):
    global _session
    connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=60)
    _session = aiohttp.ClientSession(connector=connector)
    log.info(f"🚀 Collector запущен на порту {COLLECTOR_PORT}")
    log.info(f"   Пересылает в upipe: {UPIPE_URL}")
    log.info(f"   Языковой фильтр: {LANG_FILTER or 'выключен'}")
    log.info(f"   Макс. возраст твита: {MAX_OLDNESS_SECONDS}с ({MAX_OLDNESS_SECONDS/3600:.1f}ч)")


async def on_shutdown(app: web.Application):
    global _session
    if _session:
        await _session.close()
    log.info(f"📊 Итог: {_stats}")


app = web.Application(client_max_size=10 * 1024 * 1024)
app.router.add_post("/store_item", handle_store_item)
app.router.add_get("/health", handle_health)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=COLLECTOR_PORT, print=None)
