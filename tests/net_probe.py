"""Network sanity probe. Compares sync `httpx` vs async `aiohttp` reachability to
the NVIDIA endpoints, to diagnose why NAT (aiohttp-based) fails while the openai
SDK (httpx sync) works."""

from __future__ import annotations

import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv()
URL = os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1") + "/models"
HEADERS = {"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}"}


def sync_probe() -> str:
    try:
        r = httpx.get(URL, headers=HEADERS, timeout=15)
        return f"httpx sync OK: {r.status_code}"
    except Exception as exc:
        return f"httpx sync FAIL: {type(exc).__name__}: {exc}"


async def async_aiohttp_probe() -> str:
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(URL, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as r:
                return f"aiohttp OK: {r.status}"
    except Exception as exc:
        return f"aiohttp FAIL: {type(exc).__name__}: {exc}"


async def async_httpx_probe() -> str:
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(URL, headers=HEADERS)
            return f"httpx async OK: {r.status_code}"
    except Exception as exc:
        return f"httpx async FAIL: {type(exc).__name__}: {exc}"


async def main() -> None:
    print(f"Endpoint: {URL}")
    print(f"HTTP_PROXY={os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')}")
    print(f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')}")
    print(f"NO_PROXY={os.environ.get('NO_PROXY') or os.environ.get('no_proxy')}")
    print()
    print(sync_probe())
    print(await async_aiohttp_probe())
    print(await async_httpx_probe())


if __name__ == "__main__":
    asyncio.run(main())
