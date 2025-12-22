"""
weak_labels/llm_client.py

High-throughput LLM judge client for vLLM (OpenAI-compatible /v1/chat/completions) with:
- Async HTTP + connection pooling
- Optional micro-batching (many passages per request) using structured_outputs.json
- Optional single-passage mode using structured_outputs.choice (YES/NO)
- SQLite cache (WAL) for restartability

vLLM structured outputs are passed via a `structured_outputs` object in the request body. [page:221]
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
from pathlib import Path

import httpx
from loguru import logger


# ---------------------------------------------------------------------
# Config defaults (MIRRORS what will live in configs/base.yaml under `llm:`)
# ---------------------------------------------------------------------
DEFAULT_LLM_CFG: Dict[str, Any] = {
    # Endpoint + model
    "base_url": "http://127.0.0.1:8000/v1/chat/completions",
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    # Timeouts (seconds)
    "timeout_total": 60.0,
    "timeout_connect": 10.0,
    "timeout_read": 60.0,
    "timeout_write": 10.0,
    "timeout_pool": 10.0,
    # Retries
    "max_retries": 3,
    "retry_backoff_s": 0.05,
    # Concurrency + payload sizing
    "max_concurrent_requests": 16,   # concurrent HTTP requests
    "passages_per_request": 8,       # micro-batch size (only used in guided.mode=json)
    "passage_chars": 500,            # truncate passage text
    # Structured output mode
    "guided": {
        "mode": "json",              # "json" (micro-batched) or "choice" (one passage per request)
        "choices": ["YES", "NO"],    # used if mode == "choice"
    },
    # httpx connection pool limits
    "httpx": {
        "max_connections": 256,
        "max_keepalive_connections": 256,
        "keepalive_expiry_s": 30.0,
    },
    # Cache
    "cache": {
        "enabled": True,
        "db_path": "data/prepared/llm_cache.db",
        "sqlite": {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "busy_timeout_ms": 30000,
            "cache_size_kb": 131072,   # 128MB
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
    """
    Async vLLM judge client.

    Supports two modes (set in llm_cfg["guided"]["mode"]):
    - "json": micro-batched requests using structured_outputs.json [page:221]
    - "choice": one passage per request using structured_outputs.choice [page:221]
    """

    PROMPT_VERSION = "judge_yesno_v3_cfg_only"

    def __init__(self, llm_cfg: Dict[str, Any]):
        self.cfg = _deep_update(DEFAULT_LLM_CFG, llm_cfg or {})

        base_url = self.cfg["base_url"]
        raw_urls = [u.strip() for u in base_url.split(",") if u.strip()] if "," in base_url else [base_url.strip()]
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

        logger.info(
            f"LLMJudgeClient: servers={len(self.base_urls)}, "
            f"mode={self.guided_mode}, "
            f"max_concurrent_requests={self.max_concurrent_requests}, "
            f"passages_per_request={self.passages_per_request}, "
            f"passage_chars={self.passage_chars}"
        )
        for i, u in enumerate(self.base_urls):
            logger.info(f"  Server {i+1}: {u}")
        logger.info(f"  Model: {self.model}")

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_config, limits=self.limits)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -----------------------------
    # Prompting + schemas
    # -----------------------------
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

    def _prompt_single(self, query: str, passage: Dict[str, Any]) -> str:
        q = _clean_text(query)
        text = _clean_text(passage.get("text", ""))[: self.passage_chars]
        return (
            f"Query: {q}\n"
            f"Passage: {text}\n\n"
            "Can the passage answer the query?\n"
            "Reply ONLY: YES or NO"
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

    # -----------------------------
    # HTTP call helpers
    # -----------------------------
    async def _post_with_retries(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        client = await self._ensure_client()
        for attempt in range(self.max_retries):
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.ReadError, httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff_s * (attempt + 1))
                    continue
                raise e

    # -----------------------------
    # Mode: structured_outputs.choice (1 passage/request)
    # -----------------------------
    async def _judge_one_choice(
        self,
        semaphore: asyncio.Semaphore,
        query: str,
        passage: Dict[str, Any],
    ) -> JudgeResult:
        async with semaphore:
            url = next(self.url_pool)
            prompt = self._prompt_single(query, passage)

            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 2,
                "structured_outputs": {"choice": self.guided_choices},
            }

            try:
                content = await self._post_with_retries(url, payload)
                text = str(content["choices"][0]["message"]["content"]).strip().upper()
                yes = (text == "YES")
                return JudgeResult(
                    doc_id=str(passage["doc_id"]),
                    relevance_score=3 if yes else 0,
                    answerable=yes,
                    confidence=1.0 if yes else 0.0,
                )
            except Exception as e:
                logger.warning(f"LLM(choice) failed doc_id={passage.get('doc_id')}: {type(e).__name__}: {str(e)[:120]}")
                return JudgeResult(
                    doc_id=str(passage["doc_id"]),
                    relevance_score=0,
                    answerable=False,
                    confidence=0.0,
                )

    # -----------------------------
    # Mode: structured_outputs.json (micro-batched)
    # -----------------------------
    async def _judge_group_json(
        self,
        semaphore: asyncio.Semaphore,
        query: str,
        passages: List[Dict[str, Any]],
    ) -> List[JudgeResult]:
        async with semaphore:
            url = next(self.url_pool)
            prompt = self._prompt_group(query, passages)

            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max(16, 6 * len(passages)),
                "structured_outputs": {"json": self._schema_yesno_array()},
            }

            try:
                content = await self._post_with_retries(url, payload)
                raw = str(content["choices"][0]["message"]["content"])
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
                    results.append(
                        JudgeResult(
                            doc_id=did,
                            relevance_score=3 if yes else 0,
                            answerable=yes,
                            confidence=1.0 if yes else 0.0,
                        )
                    )
                return results

            except Exception as e:
                logger.warning(f"LLM(json) failed group_size={len(passages)}: {type(e).__name__}: {str(e)[:160]}")
                return [
                    JudgeResult(doc_id=str(p["doc_id"]), relevance_score=0, answerable=False, confidence=0.0)
                    for p in passages
                ]

    async def judge_batch(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not passages:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        if self.guided_mode == "choice":
            tasks = [self._judge_one_choice(semaphore, query, p) for p in passages]
            results = await asyncio.gather(*tasks)
            return [r.as_dict() for r in results]

        tasks = [self._judge_group_json(semaphore, query, grp) for grp in _chunks(passages, self.passages_per_request)]
        grouped = await asyncio.gather(*tasks)
        flat: List[JudgeResult] = []
        for g in grouped:
            flat.extend(g)
        return [r.as_dict() for r in flat]


class CachedLLMJudge:
    """
    Sync wrapper used by the agent: judge_batch(query, passages) -> List[dict]
    with SQLite cache and a persistent async event loop thread.
    """

    def __init__(self, llm_cfg: Dict[str, Any]):
        self.cfg = _deep_update(DEFAULT_LLM_CFG, llm_cfg or {})

        self.enabled = bool(self.cfg["cache"]["enabled"])
        self.client = AsyncLLMJudgeClient(self.cfg)

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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                key_hash TEXT,
                doc_id TEXT,
                relevance_score INTEGER,
                answerable INTEGER,
                confidence REAL,
                PRIMARY KEY (key_hash, doc_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON judgments(key_hash)")
        conn.commit()

        cached = conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
        conn.close()
        logger.info(f"✓ SQLite cache ready: {cached:,} judgments")

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(
                self.cache_db_path,
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,  # autocommit for reads
            )
            conn.execute(f"PRAGMA busy_timeout={int(self.cfg['cache']['sqlite']['busy_timeout_ms'])}")
            self._local.conn = conn
        return self._local.conn

    def _key_hash(self, query: str) -> str:
        s = f"{self.client.model}|{self.client.PROMPT_VERSION}|{self.client.guided_mode}|{query}"
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

    def judge_batch(self, query: str, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not passages:
            return []

        if not self.enabled:
            fut = asyncio.run_coroutine_threadsafe(self.client.judge_batch(query, passages), self._loop)
            return fut.result()

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
