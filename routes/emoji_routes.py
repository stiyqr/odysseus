# routes/emoji_routes.py
# Same-origin emoji SVG proxy. The frontend rewrites emoji in chat to a
#   <span class="emoji" style="--em:url('/api/emoji/<codepoints>.svg')">
# which uses the returned SVG as a CSS mask tinted to the text color, so emoji
# render as monochrome line icons (project rule: never colorful emoji). The
# black line-art SVGs are lazily fetched from the OpenMoji CDN on first use and
# cached on disk, so:
#   - the client only ever talks to our own origin (no CDN dep, no CSP change),
#   - the repo isn't bloated with thousands of SVG files,
#   - it works offline once an emoji has been seen once.
# Unknown/unreachable codepoints return a transparent SVG (not 404), so the CSS
# mask shows nothing rather than a solid currentColor box.
import asyncio
import logging
import re
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import Response

from src.constants import EMOJI_CACHE_DIR

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(EMOJI_CACHE_DIR)
# OpenMoji "black" set = monochrome line-art SVGs. Filenames are the codepoints
# in UPPERCASE (FE0F dropped, same as we compute), '-' joined.
_OPENMOJI_BASE = "https://cdn.jsdelivr.net/npm/openmoji@15.0.0/black/svg"
# codepoints like "1f600" or "1f468-200d-1f469-200d-1f467" (lowercase hex, '-' joined)
_CODE_RE = re.compile(r"^[0-9a-f]{2,6}(?:-[0-9a-f]{2,6})*$")
_MAX_SVG_BYTES = 256 * 1024
_BLOCKED_SVG_RE = re.compile(
    br"<\s*(?:script|foreignObject|iframe|object|embed|image)\b|"
    br"\bon[a-z0-9_-]+\s*=",
    re.IGNORECASE,
)
_EXTERNAL_REF_RE = re.compile(
    br"\b(?:href|xlink:href)\s*=\s*['\"](?:https?:|//|data:|javascript:)",
    re.IGNORECASE,
)
_SVG_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "sandbox",
    "Cross-Origin-Resource-Policy": "same-origin",
}
_SVG_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
    **_SVG_SECURITY_HEADERS,
}
# Returned when a codepoint is unknown/unreachable: an empty (transparent) SVG,
# so the CSS mask renders nothing instead of a solid box. Not cached, so a later
# request can still pick up the real glyph once the CDN is reachable.
_BLANK_SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"></svg>'
_BLANK_HEADERS = {"Cache-Control": "no-store", **_SVG_SECURITY_HEADERS}


def _is_safe_svg(content: bytes) -> bool:
    if not isinstance(content, bytes) or not content:
        return False
    if len(content) > _MAX_SVG_BYTES:
        return False
    if b"<svg" not in content[:256].lower():
        return False
    if _BLOCKED_SVG_RE.search(content) or _EXTERNAL_REF_RE.search(content):
        return False
    return True


async def _fetch_and_cache(code: str, client: httpx.AsyncClient | None = None) -> bool:
    """Fetch one OpenMoji black SVG into the disk cache. Best-effort: never
    raises, returns whether a usable glyph is now cached."""
    fp = _CACHE_DIR / f"{code}.svg"
    try:
        if client is not None:
            r = await client.get(f"{_OPENMOJI_BASE}/{code.upper()}.svg")
        else:
            async with httpx.AsyncClient(timeout=8.0) as one_shot:
                r = await one_shot.get(f"{_OPENMOJI_BASE}/{code.upper()}.svg")
        if r.status_code == 200 and _is_safe_svg(r.content):
            try:
                fp.write_bytes(r.content)
            except Exception:
                pass  # cache write is best-effort
            return True
    except Exception as e:
        logger.warning("emoji fetch %s failed: %s", code, e)
    return False


# Common chat emoji (lowercase codepoints, FE0F dropped, '-' joined). Only
# used by the opt-in startup warmup (ODYSSEUS_STARTUP_WARMUPS=1) — see
# prewarm_common_emoji() below — so the first render of a frequent emoji is a
# cache hit instead of a CDN round trip.
COMMON_EMOJI_CODES = [
    "1f600", "1f601", "1f602", "1f603", "1f604", "1f606", "1f609", "1f60a",
    "1f60d", "1f60e", "1f610", "1f614", "1f61b", "1f61d", "1f622", "1f62d",
    "1f631", "1f680", "1f38a", "1f389", "1f44d", "1f44e", "1f4a1", "1f4aa",
    "1f4ac", "1f4ad", "1f4af", "1f495", "1f496", "1f499", "1f49b", "1f49c",
    "1f4c1", "1f4c4", "1f4dd", "1f50d", "1f514", "1f525", "1f5e3", "1f6e0",
    "1f7e1", "1f7e2", "1f916", "1f9e0", "2194", "23f3", "26a0", "26d4",
    "2705", "2709", "2728", "274c", "2764", "2b1c", "2b50", "1f534", "1f3a8",
]
_PREWARM_CONCURRENCY = 8


async def prewarm_common_emoji() -> int:
    """Fetch every not-yet-cached COMMON_EMOJI_CODES entry, bounded to
    `_PREWARM_CONCURRENCY` concurrent requests over one shared client.
    Returns the number newly cached. Safe to call repeatedly (skips whatever
    is already on disk)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [c for c in COMMON_EMOJI_CODES if not (_CACHE_DIR / f"{c}.svg").exists()]
    if not missing:
        return 0
    sem = asyncio.Semaphore(_PREWARM_CONCURRENCY)

    async def _bounded(code: str, client: httpx.AsyncClient) -> bool:
        async with sem:
            return await _fetch_and_cache(code, client)

    async with httpx.AsyncClient(timeout=8.0) as client:
        results = await asyncio.gather(*(_bounded(c, client) for c in missing))
    fetched = sum(1 for ok in results if ok)
    logger.info("emoji pre-warm: cached %d/%d common emoji", fetched, len(missing))
    return fetched


def setup_emoji_routes() -> APIRouter:
    router = APIRouter(prefix="/api/emoji", tags=["emoji"])

    def _blank() -> Response:
        return Response(_BLANK_SVG, media_type="image/svg+xml", headers=_BLANK_HEADERS)

    @router.get("/{code}.svg")
    async def emoji_svg(code: str):
        code = code.lower()
        if not _CODE_RE.match(code):
            return _blank()

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fp = _CACHE_DIR / f"{code}.svg"
        if fp.exists():
            try:
                content = fp.read_bytes()
                if _is_safe_svg(content):
                    return Response(content, media_type="image/svg+xml", headers=_SVG_HEADERS)
                fp.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("emoji cache read %s failed: %s", code, e)
            return _blank()

        # Cache miss: never block the request on the CDN. Serve the blank SVG
        # instantly and fetch the real glyph in the background, so the next
        # request for this emoji is a cache hit with zero render lag.
        asyncio.create_task(_fetch_and_cache(code))
        return _blank()

    return router
