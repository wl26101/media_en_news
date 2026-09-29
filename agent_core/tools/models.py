"""Start and stop the three local model servers the pipeline needs.

The server classes come from ai.manage_model (llama.cpp, the TTS FastAPI app,
ComfyUI for z-Image-Turbo); this module only adds the lifecycle policy the
pipeline wants: a per-model readiness budget (ComfyUI loads its model much
slower than the default wait) and reuse of servers that are already up.
"""

from __future__ import annotations

import logging

import requests
from ai.manage_model import Image, LlamaServe, TTS
from ai.settings import settings

from .. import config

logger = logging.getLogger(__name__)

# Health endpoint checked before spawning a server, so an already-running
# ComfyUI is reused instead of being started a second time.
HEALTH_PATHS = {"llm": "/health", "tts": "/health", "image": "/system_stats"}


def _ports() -> dict[str, int]:
    return {"llm": settings.llm_port, "tts": settings.tts_port, "image": settings.image_port}


class ModelServers:
    """Owns the lifetime of the llm / tts / image servers."""

    def __init__(self) -> None:
        self.servers = {"llm": LlamaServe(), "tts": TTS(), "image": Image()}
        self.running: set[str] = set()

    def is_up(self, name: str, timeout: int = 2) -> bool:
        """True when something already answers on that server's port."""
        url = f"http://127.0.0.1:{_ports()[name]}{HEALTH_PATHS[name]}"
        try:
            return requests.get(url, timeout=timeout).ok
        except Exception:
            return False

    def start(self, names: tuple[str, ...] = ("llm", "tts", "image")) -> set[str]:
        """Start the named servers; returns the ones that became ready."""
        ready: set[str] = set()
        for name in names:
            if self.is_up(name):
                logger.info("%s server already running, reusing it", name)
                self.running.add(name)
                ready.add(name)
                continue

            server = self.servers[name]
            logger.info("Starting %s server...", name)
            try:
                server.start()
            except Exception as exc:
                logger.error("Failed to start %s server: %s", name, exc)
                continue

            # every wait_ready() takes its budget in seconds as the first arg
            if server.wait_ready(config.READY_TIMEOUTS[name]):
                logger.info("%s server ready", name)
                self.running.add(name)
                ready.add(name)
            else:
                logger.error("%s server not ready after %ss", name, config.READY_TIMEOUTS[name])
                server.stop()
        return ready

    def stop(self, names: tuple[str, ...] | None = None) -> None:
        """Stop the servers this instance started (servers reused from outside
        the process are left alone: their proc handle is None)."""
        for name in names or tuple(self.servers):
            server = self.servers[name]
            if server.proc is None:
                self.running.discard(name)
                continue
            try:
                server.stop()
                logger.info("%s server stopped", name)
            except Exception as exc:
                logger.error("Failed to stop %s server: %s", name, exc)
            self.running.discard(name)
