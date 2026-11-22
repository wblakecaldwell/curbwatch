"""Minimal client for the Starling Home Hub Developer Connect API (local LAN).

By default the hub serves the API on port 3080 (HTTP) and 3443 (HTTPS, self-signed cert)
under /api/connect/v1. The API key goes in the `key` query parameter.
"""

from __future__ import annotations

import httpx

from . import config


class StarlingError(RuntimeError):
    pass


class Starling:
    def __init__(self, hub: str | None = None, key: str | None = None, https: bool = True):
        hub = hub or config.hub()
        port, scheme = (config.STARLING_HTTPS_PORT, "https") if https else (config.STARLING_HTTP_PORT, "http")
        self.base = f"{scheme}://{hub}:{port}/api/connect/v1"
        self._key = key or config.require("STARLING_API_KEY")
        # The hub uses a self-signed certificate; traffic stays on the LAN.
        self._http = httpx.Client(verify=False, timeout=30)

    def _get(self, path: str, **params) -> httpx.Response:
        r = self._http.get(f"{self.base}{path}", params={"key": self._key, **params})
        ctype = r.headers.get("content-type", "")
        if "json" in ctype:
            body = r.json()
            if isinstance(body, dict) and body.get("status") == "Error":
                raise StarlingError(f"{path}: {body.get('code')}: {body.get('message')}")
        r.raise_for_status()
        return r

    def status(self) -> dict:
        return self._get("/status").json()

    def devices(self) -> list[dict]:
        return self._get("/devices").json().get("devices", [])

    def device(self, device_id: str) -> dict:
        return self._get(f"/devices/{device_id}").json()

    def cameras(self) -> list[dict]:
        return [d for d in self.devices() if is_camera(d)]

    def snapshot(self, device_id: str, width: int | None = None) -> bytes:
        params = {"width": width} if width else {}
        r = self._get(f"/devices/{device_id}/snapshot", **params)
        if not r.headers.get("content-type", "").startswith("image/"):
            raise StarlingError(f"snapshot for {device_id} returned {r.headers.get('content-type')}")
        return r.content


def is_camera(d: dict) -> bool:
    cat = str(d.get("category", "")).lower()
    typ = str(d.get("type", "")).lower()
    return "cam" in cat or "cam" in typ or "doorbell" in cat or "doorbell" in typ
