"""The service's settings for its environment (config/service.yaml): D2Q_ENV = development (default) or production."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config" / "service.yaml"


def load() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf8")) if CONFIG.exists() else {}
    cfg = cfg or {}
    env = (os.environ.get("D2Q_ENV") or "development").strip().lower()
    if env not in ("development", "production"):
        raise ValueError(f"D2Q_ENV must be development or production, not {env!r}")
    section = cfg.get(env) or {}
    key = os.environ.get("D2Q_KEY")
    return {
        "environment": env,
        "host": str(section.get("host") or "127.0.0.1"),
        "port": int(section.get("port") or 8765),
        "key": key if key is not None else str(section.get("key") or ""),
        # browsers send the host in lower case
        "allowed_origins": [str(o).strip().lower() for o in cfg.get("allowed_origins") or [] if str(o).strip()],
    }


SETTINGS = load()
