"""
weak_labels/llm_client.py

High-throughput LLM judge client for vLLM (OpenAI-compatible /v1/chat/completions).

Speed principles for long prompts (~15k–32k tokens):
- Keep output token budget tiny (decode reservation hurts throughput).
- Prefer micro-batching when prompt construction already groups passages.
- Use vLLM documented structured-output fields: guided_choice / guided_json.
- Keep a single AsyncClient + semaphore; avoid per-call allocations.
- Track global progress with ETA.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterable
from dataclasses import dataclass
import asyncio
import hashlib
import itertools
import json
import sqlite3
import threading
import time
from pathlib import Path
import re

import httpx
from loguru import logger


DEFAULT_LLM_CFG: Dict[str, Any] = {
    "base_url": "http://127.0.0.1:8000/v1/chat/completions",
    "model": "Qwen/Qwen2.5-7B-Instruct",

    "timeout_total": 120.0,
    "timeout_connect": 10.0,
    "timeout_read": 120.0,
    "timeout_write": 10.0,
    "timeout_pool": 10.0,

    "max_retries": 3,
    "retry_backoff_s": 0.05,

    "max_concurrent_requests": 16,
    "passages_per_request": 8,
    "passage_chars": 500,

    # "choice" => guided_choice (single passage, robust)
    # "json"   => guided_json (micro-batched; can still truncate on long contexts)
    "guided": {
        "mode": "choice",   # CHANGED: default to robust YES/NO classification
        "choices": ["YES", "NO"],
    },

    "httpx": {
        "max_connections": 256,
        "max_keepalive_connections": 256,
        "keepalive_expiry_s": 15.0,
        "http2": False,
    },

    "progress": {
        "log_every_s": 30.0,
    },

    "prompt_parsing": {
        "auto_expected_items": True,
        "doc_id_regex": r"\[DOC_ID=([^\]]+)\]",
    },

    "cache": {
        "enabled": True,
        "db_path": "data/prepared/llm_cache.db",
        "sqlite": {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "busy_timeout_ms": 30000,
            "cache_size_kb": 131072,
            "temp_store": "MEMORY",
        },
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _normalize_chat_completions_url(url: str) -> str:
    u = url.strip().rstrip("/")
    if u.endswith("/v1/chat/completions"):
        return u
    if u.endswith("/v1"):
        return u + "/chat/completions"
    return u


def _chunks(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i: i + n]


def _clean_text(s: str) -> str:
    return (s or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


class ProgressTracker:
    def __init__(self, total: Optional[int] = None, log_every_s: float = 30.0):
        self.total = total
        self.log_every_s = float(log_every_s)
        self._lock = threading.Lock()
        self.start_t = time.time()
        self.last_log_t = self.start_t
        self.done = 0
        self.fail = 0

    def set_total(self, total: int) -> None:
        with self._lock:
            self.total = int(total)

    def update(self, n_done: int, n_fail: int = 0) -> None:
        now = time.time()
        with self._lock:
            self.done += int(n_done)
            self.fail += int(n_fail)

            if now - self.last_log_t < self.log_every_s:
                return
            self.last_log_t = now

            elapsed = max(1e-9, now - self.start_t)
            rate = self.done / elapsed

            if self.total is not None and self.total > 0 and rate > 0:
                remaining = max(0, self.total - self.done)
                eta_s = remaining / rate
                logger.info(
                    f"LLM progress: done={self.done:,}/{self.total:,} "
                    f"fail={self.fail:,} rate={rate:.2f}/s ETA={eta_s/60:.1f}m"
                )
            else:
                logger.info(
                    f"LLM progress: done={self.done:,} fail={self.fail:,} rate={rate:.2f}/s"
                )


@dataclass(frozen=True)
class JudgeResult:
    doc_id: str
    relevance_score: int
    answerable: bool
    confidence: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "relevance_score": self.relevance_score,
            "answerable": self.answerable,
            "confidence": self.confidence,
        }


class AsyncLLMJudgeClient:
    PROMPT_VERSION = "judge_yesno_v5_guided_params_retry_http"

    def __init__(self, llm_cfg: Dict[str, Any], progress: Optional[ProgressTracker] = None):
        self.cfg = _deep_update(DEFAULT_LLM_CFG, llm_cfg or {})

        raw = self.cfg["base_url"]
        raw_urls = [u.strip() for u in raw.split(",") if u.strip()] if "," in raw else [raw.strip()]
        self.base_urls = [_normalize_chat_completions_url(u) for u in raw_urls]
        self.url_pool = itertools.cycle(self.base_urls)

        self.model = str(self.cfg["model"])
        self.max_retries = int(self.cfg["max_retries"])
        self.retry_backoff_s = float(self.cfg["retry_backoff_s"])

        self.max_concurrent_requests = int(self.cfg["max_concurrent_requests"])
        self.passages_per_request = max(1, int(self.cfg["passages_per_request"]))
        self.passage_chars = int(self.cfg["passage_chars"])

        guided = self.cfg["guided"]
        self.guided_mode = str(guided["mode"]).lower()
        self.guided_choices = list(guided.get("choices", ["YES", "NO"]))

        self.timeout_config = httpx.Timeout(
            timeout=float(self.cfg["timeout_total"]),
            connect=float(self.cfg["timeout_connect"]),
            read=float(self.cfg["timeout_read"]),
            write=float(self.cfg["timeout_write"]),
            pool=float(self.cfg["timeout_pool"]),
        )
        httpx_cfg = self.cfg["httpx"]
        self.limits = httpx.Limits(
            max_connections=int(httpx_cfg["max_connections"]),
            max_keepalive_connections=int(httpx_cfg["max_keepalive_connections"]),
            keepalive_expiry=float(httpx_cfg["keepalive_expiry_s"]),
        )

        self._client: Optional[httpx.AsyncClient] = None
        self._sema = asyncio.Semaphore(self.max_concurrent_requests)

        p_cfg = self.cfg.get("progress", {})
        self.progress = progress or ProgressTracker(total=None, log_every_s=p_cfg.get("log_every_s", 30.0))

        pp_cfg = self.cfg.get("prompt_parsing", {})
        self.auto_expected_items = bool(pp_cfg.get("auto_expected_items", True))
        self._doc_id_re = re.compile(str(pp_cfg.get("doc_id_regex", r"\[DOC_ID=([^\]]+)\]")))

        logger.info(
            f"LLMJudgeClient: servers={len(self.base_urls)}, mode={self.guided_mode}, "
            f"max_concurrent_requests={self.max_concurrent_requests}, passages_per_request={self.passages_per_request}"
        )
        for i, u in enumerate(self.base_urls):
            logger.info(f"  Server {i+1}: {u}")
        logger.info(f"  Model: {self.model}")

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            http2 = bool(self.cfg["httpx"].get("http2", False))
            self._client = httpx.AsyncClient(
                timeout=self.timeout_config,
                limits=self.limits,
                http2=http2,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _schema_yesno_array() -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "label": {"type": "string", "enum": ["YES", "NO"]},
                },
                "required": ["doc_id", "label"],
                "additionalProperties": False,
            },
        }

    def _estimate_items_from_prompt(self, prompt: str) -> int:
        if not prompt:
            return 0
        return len(self._doc_id_re.findall(prompt))

    def _prompt_single(self, query: str, passage: Dict[str, Any]) -> str:
        q = _clean_text(query)
        text = _clean_text(passage.get("text", ""))[: self.passage_chars]
        return (
            f"Query: {q}\n"
            f"Passage: {text}\n\n"
            "Can the passage answer the query?\n"
            "Reply ONLY: YES or NO."
        )

    def _prompt_group(self, query: str, passages: List[Dict[str, Any]]) -> str:
        q = _clean_text(query)
        lines = [
            f"Query: {q}",
            "",
            "For each passage below, decide if it can answer the query.",
            "Return a JSON array of objects with fields: doc_id, label (YES or NO).",
            "",
            "Passages:",
        ]
        for p in passages:
            doc_id = str(p["doc_id"])
            text = _clean_text(p.get("text", ""))[: self.passage_chars]
            lines.append(f"- doc_id: {doc_id}\n  text: {text}")
        return "\n".join(lines)

    async def _post_with_retries(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = await self._ensure_client()

        for attempt in range(self.max_retries):
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as e:
                code = e.response.status_code if e.response is not None else None
                if code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff_s * (attempt + 1))
                    continue
                raise

            except (httpx.ReadError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff_s * (attempt + 1))
                    continue
                raise e

    async def _judge_one_choice(self, query: str, passage: Dict[str, Any]) -> JudgeResult:
        async with self._sema:
            url = next(self.url_pool)
            prompt = self._prompt_single(query, passage)

            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 2,
                "guided_choice": self.guided_choices,
            }

            try:
                content = await self._post_with_retries(url, payload)
                text = str(content["choices"][0]["message"]["content"]).strip().upper()
                yes = (text == "YES")
                self.progress.update(1, 0)
                return JudgeResult(
                    doc_id=str(passage["doc_id"]),
                    relevance_score=3 if yes else 0,
                    answerable=yes,
                    confidence=1.0 if yes else 0.0,
                )
            except Exception as e:
                logger.warning(
                    f"LLM(choice) failed doc_id={passage.get('doc_id')}: {type(e).__name__}: {str(e)[:120]}"
                )
                self.progress.update(1, 1)
                return JudgeResult(doc_id=str(passage["doc_id"]), relevance_score=0, answerable=False, confidence=0.0)

    async def _judge_group_json(self, query: str, passages: List[Dict[str, Any]]) -> List[JudgeResult]:
        async with self._sema:
            url = next(self.url_pool)
            prompt = self._prompt_group(query, passages)

            schema = self._schema_yesno_array()
            max_out = max(64, 10 * len(passages))

            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_out,
                "guided_json": schema,
            }

            try:
                content = await self._post_with_retries(url, payload)
                raw = content["choices"][0]["message"]["content"]
                arr = json.loads(raw)

                out_map: Dict[str, bool] = {}
                for rec in arr:
                    did = str(rec["doc_id"])
                    lab = str(rec["label"]).strip().upper()
                    out_map[did] = (lab == "YES")

                results: List[JudgeResult] = []
                for p in passages:
                    did = str(p["doc_id"])
                    yes = bool(out_map.get(did, False))
                    results.append(JudgeResult(
                        doc_id=did,
                        relevance_score=3 if yes else 0,
                        answerable=yes,
                        confidence=1.0 if yes else 0.0,
                    ))
                self.progress.update(1, 0)
                return results

            except Exception as e:
                logger.warning(f"LLM(json) failed group_size={len(passages)}: {type(e).__name__}: {str(e)[:160]}")
                self.progress.update(1, 1)
                return [
                    JudgeResult(doc_id=str(p["doc_id"]), relevance_score=0, answerable=False, confidence=0.0)
                    for p in passages
                ]

    async def judge_batch(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not passages:
            return []

        if self.guided_mode == "choice":
            tasks = [self._judge_one_choice(query, p) for p in passages]
            results = await asyncio.gather(*tasks)
            return [r.as_dict() for r in results]

        tasks = [self._judge_group_json(query, grp) for grp in _chunks(passages, self.passages_per_request)]
        grouped = await asyncio.gather(*tasks)
        flat: List[JudgeResult] = []
        for g in grouped:
            flat.extend(g)
        return [r.as_dict() for r in flat]

    async def judge_prompt_async(
        self,
        prompt: str,
        guided_mode: Optional[str] = None,
        expected_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Judge a pre-built prompt (already contains query+passages).

        NOTE: This path is kept for backwards compatibility.
        For robustness, prefer judge_batch(query, passages) with guided.mode="choice".
        """
        mode = (guided_mode or self.guided_mode).lower()
        url = next(self.url_pool)

        if mode == "choice":
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 2,
                "guided_choice": self.guided_choices,
            }
            try:
                content = await self._post_with_retries(url, payload)
                text = str(content["choices"][0]["message"]["content"]).strip().upper()
                yes = (text == "YES")
                self.progress.update(1, 0)
                return [{
                    "doc_id": "N/A",
                    "relevance_score": 3 if yes else 0,
                    "answerable": yes,
                    "confidence": 1.0 if yes else 0.0,
                }]
            except Exception as e:
                logger.warning(f"LLM(choice) judge_prompt failed: {type(e).__name__}: {str(e)[:160]}")
                self.progress.update(1, 1)
                return []

        schema = self._schema_yesno_array()

        if expected_items is None and self.auto_expected_items:
            expected_items = self._estimate_items_from_prompt(prompt)

        n = int(expected_items) if expected_items is not None and expected_items > 0 else 128
        max_out = max(256, 8 * n)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_out,
            "guided_json": schema,
        }

        try:
            content = await self._post_with_retries(url, payload)
            raw = content["choices"][0]["message"]["content"]
            arr = json.loads(raw)

            results: List[Dict[str, Any]] = []
            for rec in arr:
                did = str(rec["doc_id"])
                lab = str(rec["label"]).strip().upper()
                yes = (lab == "YES")
                results.append({
                    "doc_id": did,
                    "relevance_score": 3 if yes else 0,
                    "answerable": yes,
                    "confidence": 1.0 if yes else 0.0,
                })
            self.progress.update(1, 0)
            return results
        except Exception as e:
            logger.warning(f"LLM(json) judge_prompt failed: {type(e).__name__}: {str(e)[:160]}")
            self.progress.update(1, 1)
            return []


class CachedLLMJudge:
    """
    Sync wrapper used by the agent: judge_batch(query, passages) -> List[dict]
    with SQLite cache and a persistent async event loop thread.

    Also supports judge_prompt(prompt, query_id, guided_mode) for Phase 2 (legacy).
    """

    def __init__(self, llm_cfg: Dict[str, Any], total_prompts: Optional[int] = None):
        self.cfg = _deep_update(DEFAULT_LLM_CFG, llm_cfg or {})

        self.enabled = bool(self.cfg["cache"]["enabled"])

        p_cfg = self.cfg.get("progress", {})
        self.progress = ProgressTracker(total=total_prompts, log_every_s=p_cfg.get("log_every_s", 30.0))

        self.client = AsyncLLMJudgeClient(self.cfg, progress=self.progress)

        cache_db = str(self.cfg["cache"]["db_path"])
        cache_path = Path(cache_db)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_db_path = str(cache_path)

        self._local = threading.local()
        self._write_lock = threading.Lock()

        self._init_sqlite()

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, args=(self._loop,), daemon=True)
        self._loop_thread.start()

        logger.info(f"LLM cache enabled={self.enabled}, db={self.cache_db_path}")

    def set_total_prompts(self, total: int) -> None:
        self.progress.set_total(total)

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _init_sqlite(self) -> None:
        sqlite_cfg = self.cfg["cache"]["sqlite"]

        conn = sqlite3.connect(self.cache_db_path)
        conn.execute(f"PRAGMA journal_mode={sqlite_cfg['journal_mode']}")
        conn.execute(f"PRAGMA synchronous={sqlite_cfg['synchronous']}")
        conn.execute(f"PRAGMA busy_timeout={int(sqlite_cfg['busy_timeout_ms'])}")
        conn.execute(f"PRAGMA cache_size={-int(sqlite_cfg['cache_size_kb'])}")
        conn.execute(f"PRAGMA temp_store={sqlite_cfg['temp_store']}")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS judgments (
                key_hash TEXT,
                doc_id TEXT,
                relevance_score INTEGER,
                answerable INTEGER,
                confidence REAL,
                PRIMARY KEY (key_hash, doc_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON judgments(key_hash)")
        conn.commit()

        cached = conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
        conn.close()
        logger.info(f"✓ SQLite cache ready: {cached:,} judgments")

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.cache_db_path, timeout=30.0, check_same_thread=False, isolation_level=None)
            conn.execute(f"PRAGMA busy_timeout={int(self.cfg['cache']['sqlite']['busy_timeout_ms'])}")
            self._local.conn = conn
        return self._local.conn

    def _key_hash(self, key_str: str) -> str:
        s = f"{self.client.model}|{self.client.PROMPT_VERSION}|{self.client.guided_mode}|{key_str}"
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

    def _read_cache(self, key_hash: str, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        conn = self._get_conn()
        out: Dict[str, Dict[str, Any]] = {}

        for did_chunk in _chunks(doc_ids, 500):
            placeholders = ",".join("?" * len(did_chunk))
            rows = conn.execute(
                f"SELECT doc_id, relevance_score, answerable, confidence FROM judgments "
                f"WHERE key_hash=? AND doc_id IN ({placeholders})",
                [key_hash] + did_chunk,
            ).fetchall()

            for doc_id, rel, ans, conf in rows:
                out[str(doc_id)] = {
                    "doc_id": str(doc_id),
                    "relevance_score": int(rel),
                    "answerable": bool(ans),
                    "confidence": float(conf),
                }
        return out

    def _write_cache(self, key_hash: str, results: List[Dict[str, Any]]) -> None:
        if not results:
            return

        with self._write_lock:
            sqlite_cfg = self.cfg["cache"]["sqlite"]
            conn = sqlite3.connect(self.cache_db_path, timeout=30.0, check_same_thread=False)
            conn.execute(f"PRAGMA busy_timeout={int(sqlite_cfg['busy_timeout_ms'])}")
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO judgments VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            key_hash,
                            str(r["doc_id"]),
                            int(r["relevance_score"]),
                            int(bool(r["answerable"])),
                            float(r["confidence"]),
                        )
                        for r in results
                    ],
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _cache_key_for_passage(query: str, passage: Dict[str, Any]) -> str:
        did = str(passage.get("doc_id", ""))
        txt = str(passage.get("text", ""))
        return f"{query}\nDOC_ID={did}\nTEXT={txt}"

    def judge_batch(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not passages:
            return []

        if not self.enabled:
            fut = asyncio.run_coroutine_threadsafe(self.client.judge_batch(query, passages), self._loop)
            return fut.result()

        # In choice mode, cache must be per passage (query+doc_id+text) to avoid collisions.
        if self.client.guided_mode == "choice":
            hits: Dict[str, Dict[str, Any]] = {}
            miss: List[Dict[str, Any]] = []

            for p in passages:
                did = str(p["doc_id"])
                kh = self._key_hash(self._cache_key_for_passage(query, p))
                rec = self._read_cache(kh, [did]).get(did)
                if rec is not None:
                    hits[did] = rec
                else:
                    miss.append(p)

            if miss:
                fut = asyncio.run_coroutine_threadsafe(self.client.judge_batch(query, miss), self._loop)
                new_results = fut.result()

                for p, r in zip(miss, new_results):
                    did = str(p["doc_id"])
                    kh = self._key_hash(self._cache_key_for_passage(query, p))
                    self._write_cache(kh, [r])
                    hits[did] = r

            return [hits[str(p["doc_id"])] for p in passages]

        # JSON mode (legacy): key per query is fine.
        key_hash = self._key_hash(query)
        doc_ids = [str(p["doc_id"]) for p in passages]

        cache_map = self._read_cache(key_hash, doc_ids)

        cached: List[Dict[str, Any]] = []
        miss: List[Dict[str, Any]] = []
        for p in passages:
            did = str(p["doc_id"])
            rec = cache_map.get(did)
            if rec is not None:
                cached.append(rec)
            else:
                miss.append(p)

        if miss:
            fut = asyncio.run_coroutine_threadsafe(self.client.judge_batch(query, miss), self._loop)
            new_results = fut.result()
            self._write_cache(key_hash, new_results)
            cached.extend(new_results)

        out_map = {r["doc_id"]: r for r in cached}
        return [out_map[str(p["doc_id"])] for p in passages]

    def judge_prompt(
        self,
        prompt: str,
        query_id: str,
        guided_mode: Optional[str] = None,
        expected_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        key_str = f"{query_id}|{guided_mode or self.client.guided_mode}|{prompt}"
        key_hash = self._key_hash(key_str)

        fut = asyncio.run_coroutine_threadsafe(
            self.client.judge_prompt_async(prompt, guided_mode, expected_items=expected_items),
            self._loop,
        )
        results = fut.result()

        self._write_cache(key_hash, results)
        return results

    def cache_stats(self) -> Dict[str, int]:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
        keys = conn.execute("SELECT COUNT(DISTINCT key_hash) FROM judgments").fetchone()[0]
        return {"total_judgments": int(total), "unique_queries": int(keys)}

    def close(self) -> None:
        try:
            asyncio.run_coroutine_threadsafe(self.client.aclose(), self._loop).result(timeout=30)
        except Exception:
            pass
        try:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        try:
            if hasattr(self._local, "conn"):
                self._local.conn.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
