"""map/reduce 中間結果落地，支援續跑（對應 review 2.2-21）。

長時間執行的離線批次一旦中斷即可續跑，不必全部重來——把 map 與 reduce 的結構化中間
結果以 JSON 落地，並以「輸入檔內容雜湊」當 run key，確保續跑對應到同一份輸入。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Optional

from . import config

log = logging.getLogger(__name__)


def run_key(path_a: str, path_b: str) -> str:
    """以兩檔內容雜湊組出穩定的 run key。"""
    h = hashlib.sha256()
    for p in (path_a, path_b):
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _path(key: str, stage: str) -> str:
    return os.path.join(config.CHECKPOINT_DIR, f"{key}.{stage}.json")


def save(key: str, stage: str, data: Any) -> None:
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    path = _path(key, stage)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)  # 原子寫入，避免中斷留下半截檔
    log.info("已寫入 checkpoint：%s", path)


def load(key: str, stage: str) -> Optional[Any]:
    path = _path(key, stage)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info("載入 checkpoint：%s", path)
        return data
    except (OSError, json.JSONDecodeError) as e:
        log.warning("checkpoint 讀取失敗（%s），忽略並重算", e)
        return None
