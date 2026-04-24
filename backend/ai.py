# backend/ai.py
import os
import json
import pickle
import hashlib
import sqlite3
import threading
from collections import OrderedDict
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import request, jsonify, render_template

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))  # project root
CSV_DIR     = os.path.join(BASE_DIR, "csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# TF-IDF index stays as pickle (binary blobs, fine for this use)
TFIDF_CACHE_FILE = os.path.join(RESULTS_DIR, "tfidf_index.pkl")

# NEW: SQLite QA cache (replaces qa_cache.pkl entirely)
DB_FILE = os.path.join(RESULTS_DIR, "ai_cache.db")

AI_FILE = __file__

# Relevance threshold: below this score → out-of-scope → general fallback
RELEVANCE_THRESHOLD = 0.05

# In-memory LRU cap (hot queries stay in RAM, no disk touch at all)
LRU_MAX = 128
_index_lock = threading.Lock()
DOCS = None
META = None
VECTORIZER = None
DOC_VECS = None

CSV_FILES = [
    "college.csv",
    "rank_2021.csv",
    "rank_2022.csv",
    "rank_2023.csv",
    "rank_2024.csv",
    "rank_2025.csv",
    "placement.csv",
    "reviews.csv",
]

# ======================================================================
# LAYER 1 — In-memory LRU (per-process, ~0 ms)
# ======================================================================
_lru_lock   = threading.Lock()
_lru_cache: OrderedDict = OrderedDict()


def _lru_get(key: str):
    with _lru_lock:
        if key in _lru_cache:
            _lru_cache.move_to_end(key)
            return dict(_lru_cache[key])   # return a copy
    return None


def _lru_set(key: str, value: dict):
    with _lru_lock:
        _lru_cache[key] = value
        _lru_cache.move_to_end(key)
        if len(_lru_cache) > LRU_MAX:
            _lru_cache.popitem(last=False)  # evict oldest


# ======================================================================
# LAYER 2 — SQLite persistent cache (survives restarts, ~1-2 ms)
# ======================================================================
_db_lock = threading.Lock()


def _db_connect():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")      # write-ahead log = fast
    conn.execute("PRAGMA synchronous=NORMAL")    # safe but not slow
    conn.row_factory = sqlite3.Row
    return conn


def _db_init():
    with _db_lock:
        conn = _db_connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS qa_cache (
                query_key   TEXT PRIMARY KEY,
                query_raw   TEXT NOT NULL,
                answer      TEXT NOT NULL,
                mode        TEXT NOT NULL,
                sources     TEXT NOT NULL,
                hit_count   INTEGER DEFAULT 1,
                created_at  TEXT NOT NULL,
                last_hit_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mode ON qa_cache(mode)"
        )
        conn.commit()
        conn.close()
    print(f"ai.py: SQLite cache ready → {DB_FILE}")


def _db_get(key: str):
    with _db_lock:
        conn = _db_connect()
        try:
            row = conn.execute(
                "SELECT answer, mode, sources, hit_count FROM qa_cache WHERE query_key = ?",
                (key,)
            ).fetchone()

            if row:
                conn.execute(
                    "UPDATE qa_cache SET hit_count = hit_count + 1, last_hit_at = ? "
                    "WHERE query_key = ?",
                    (datetime.utcnow().isoformat(), key)
                )
                conn.commit()
                return {
                    "ok": True,
                    "answer":  row["answer"],
                    "mode":    row["mode"],
                    "sources": json.loads(row["sources"]),
                    "_cache":  "sqlite",
                    "_hits":   row["hit_count"] + 1,
                }
        finally:
            conn.close()
    return None


def _db_set(key: str, query_raw: str, answer: str, mode: str, sources: list):
    now = datetime.utcnow().isoformat()
    with _db_lock:
        conn = _db_connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO qa_cache
                    (query_key, query_raw, answer, mode, sources,
                     hit_count, created_at, last_hit_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (key, query_raw, answer, mode, json.dumps(sources), now, now)
            )
            conn.commit()
        finally:
            conn.close()


def _db_stats():
    with _db_lock:
        conn = _db_connect()
        try:
            total   = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
            db_hits = conn.execute("SELECT COALESCE(SUM(hit_count),0) FROM qa_cache").fetchone()[0]
            top5    = conn.execute(
                "SELECT query_raw, hit_count FROM qa_cache ORDER BY hit_count DESC LIMIT 5"
            ).fetchall()
            by_mode = conn.execute(
                "SELECT mode, COUNT(*) as cnt FROM qa_cache GROUP BY mode"
            ).fetchall()
        finally:
            conn.close()

    return {
        "total_cached_queries": total,
        "total_cache_hits":     int(db_hits),
        "lru_size":             len(_lru_cache),
        "lru_max":              LRU_MAX,
        "top_queries":          [{"query": r["query_raw"], "hits": r["hit_count"]} for r in top5],
        "by_mode":              {r["mode"]: r["cnt"] for r in by_mode},
        "db_path":              DB_FILE,
    }


# ======================================================================
# Unified cache helpers
# ======================================================================
def _make_key(query: str) -> str:
    """Normalise + hash → fixed-length key; handles case/whitespace variants."""
    return hashlib.sha256(query.lower().strip().encode("utf-8")).hexdigest()


def _cache_get(key: str):
    # 1. RAM (LRU) — fastest
    hit = _lru_get(key)
    if hit:
        hit["_cache"] = "lru"
        return hit
    # 2. Disk (SQLite) — fast, persistent
    hit = _db_get(key)
    if hit:
        _lru_set(key, hit)   # warm LRU for next time
        return hit
    return None


def _cache_set(key: str, query_raw: str, payload: dict):
    _db_set(key, query_raw, payload["answer"], payload["mode"], payload.get("sources", []))
    _lru_set(key, payload)


# Initialise DB table on module load
_db_init()


# ======================================================================
# Helpers
# ======================================================================
def file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


# ======================================================================
# TF-IDF index (pickle — only for sklearn binary blobs)
# ======================================================================
def load_csvs_and_build_index(csv_dir, csv_files):
    ai_hash = file_hash(AI_FILE)

    if os.path.exists(TFIDF_CACHE_FILE):
        try:
            with open(TFIDF_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)

            valid = cache.get("ai_hash") == ai_hash
            for fname in csv_files:
                path = os.path.join(csv_dir, fname)
                if not os.path.exists(path):
                    continue
                if os.path.getmtime(path) > cache["timestamps"].get(fname, 0):
                    valid = False
                    break

            if valid:
                print("ai.py: TF-IDF index loaded from cache ✅")
                return cache["docs"], cache["meta"], cache["vectorizer"], cache["doc_vectors"]

        except Exception as e:
            print("TF-IDF cache invalid, rebuilding:", e)

    print("ai.py: Building TF-IDF index…")
    docs, meta, timestamps = [], [], {}

    for fname in csv_files:
        path = os.path.join(csv_dir, fname)
        if not os.path.exists(path):
            continue

        timestamps[fname] = os.path.getmtime(path)
        df = pd.read_csv(path, dtype=str).fillna("")

        for idx, row in df.iterrows():
            snippet = " ".join([
                f"{col}: {row[col]}"
                for col in df.columns if str(row[col]).strip()
            ])[:300]
            docs.append(snippet)
            meta.append({
                "source_file": fname,
                "row_index":   int(idx),
                "raw_row":     row.to_dict(),
            })

    vectorizer  = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)

    with open(TFIDF_CACHE_FILE, "wb") as f:
        pickle.dump({
            "docs":        docs,
            "meta":        meta,
            "vectorizer":  vectorizer,
            "doc_vectors": doc_vectors,
            "timestamps":  timestamps,
            "ai_hash":     ai_hash,
        }, f)

    print(f"ai.py: TF-IDF index built ({len(docs)} docs) ✅")
    return docs, meta, vectorizer, doc_vectors

def ensure_index_loaded():
    global DOCS
    global META
    global VECTORIZER
    global DOC_VECS

    if VECTORIZER is not None and DOC_VECS is not None:
        return

    with _index_lock:
        if VECTORIZER is not None and DOC_VECS is not None:
            return
        DOCS, META, VECTORIZER, DOC_VECS = load_csvs_and_build_index(CSV_DIR, CSV_FILES)


# ======================================================================
# Retrieval
# ======================================================================
def retrieve_top_rows(query, top_k=5):
    if not query.strip():
        return []

    ensure_index_loaded()

    qv   = VECTORIZER.transform([query])
    sims = linear_kernel(qv, DOC_VECS).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    return [
        {"score": float(sims[i]), "meta": META[i]}
        for i in top_idx
        if float(sims[i]) > 0
    ]


# ======================================================================
# Gemini API
# ======================================================================
def call_gemini(system_prompt: str, user_message: str) -> str:
    import urllib.request
    import urllib.error

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "⚠️ GEMINI_API_KEY is not set. Please check your .env file."

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-lite:generateContent?key={api_key}"
    )

    payload = json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents":           [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig":   {"maxOutputTokens": 512, "temperature": 0.7},
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print("Gemini API HTTPError:", body)
        return f"Gemini API error ({e.code}). Check your API key and quota."
    except Exception as e:
        print("Gemini API error:", e)
        return f"Could not reach Gemini API: {str(e)}"


# ======================================================================
# RAG answer (CSV context)
# ======================================================================
def rag_answer(query: str, results: list) -> str:
    context_lines = []
    for i, r in enumerate(results, 1):
        row    = r["meta"]["raw_row"]
        source = r["meta"]["source_file"].replace(".csv", "").replace("_", " ").title()
        row_str = ", ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
        context_lines.append(f"[{i}] ({source}) {row_str}")

    system_prompt = (
        "You are a helpful college counselor assistant for the edVance platform. "
        "You help students understand college rankings, placements, reviews, and related information. "
        "Answer the user's question using ONLY the data provided in the context below. "
        "Be conversational, friendly, and clear. Format numbers and ranks nicely. "
        "If the context doesn't fully answer the question, say so honestly — do not make up data. "
        "Keep answers concise (2-4 sentences unless detail is needed)."
        "\n\nContext from database:\n" + "\n".join(context_lines)
    )
    return call_gemini(system_prompt, query)


# ======================================================================
# Fallback answer (general knowledge)
# ======================================================================
def fallback_answer(query: str) -> str:
    system_prompt = (
        "You are a helpful assistant for the edVance college information platform. "
        "The user asked a question not covered in the college database. "
        "Answer helpfully from your general knowledge. "
        "If the question is about a specific college or ranking not in your knowledge, say so. "
        "Keep your answer concise and friendly. "
        "At the end, add a one-line note: "
        "'📌 Note: This answer is based on general knowledge, not the edVance database.'"
    )
    return call_gemini(system_prompt, query)


# ======================================================================
# ROUTES
# ======================================================================
def register_ai(app):

    # ------------------------------------------------------------------
    @app.route("/ai")
    def ai_page():
        return render_template("partials/ai.html")

    # ------------------------------------------------------------------
    @app.route("/ai/chat", methods=["POST"])
    def chat():
        data  = request.get_json()
        query = (data.get("query") or "").strip()

        if not query:
            return jsonify({"ok": False, "answer": "Please enter a question."})

        key = _make_key(query)

        # ── Cache lookup (LRU → SQLite) ────────────────────────────────
        cached = _cache_get(key)
        if cached:
            print(f"ai.py: [{cached.get('_cache','?')}] cache hit — {query[:60]}")
            return jsonify(cached)

        # ── Retrieve from CSV index ────────────────────────────────────
        results   = retrieve_top_rows(query, top_k=5)
        top_score = results[0]["score"] if results else 0.0

        if top_score >= RELEVANCE_THRESHOLD:
            answer  = rag_answer(query, results)
            sources = [
                {"file":  r["meta"]["source_file"],
                 "score": round(r["score"], 4),
                 "data":  r["meta"]["raw_row"]}
                for r in results
            ]
            mode = "database"
        else:
            answer  = fallback_answer(query)
            sources = []
            mode    = "general"

        payload = {"ok": True, "answer": answer, "mode": mode, "sources": sources}

        # ── Store in both cache layers ─────────────────────────────────
        _cache_set(key, query, payload)
        print(f"ai.py: [new] cached [{mode}] — {query[:60]}")

        return jsonify(payload)

    # ------------------------------------------------------------------
    @app.route("/ai/cache-stats", methods=["GET"])
    def cache_stats():
        """GET /ai/cache-stats — returns cache health as JSON."""
        return jsonify(_db_stats())

    # ------------------------------------------------------------------
    @app.route("/ai/cache-clear", methods=["POST"])
    def cache_clear():
        """
        POST /ai/cache-clear
        Optional JSON body:
          { "mode": "general" }   → clear only general-knowledge entries
          { "mode": "database" }  → clear only database entries
          {}                      → clear everything
        """
        body = request.get_json(silent=True) or {}
        mode = body.get("mode")

        with _db_lock:
            conn = _db_connect()
            try:
                if mode:
                    conn.execute("DELETE FROM qa_cache WHERE mode = ?", (mode,))
                else:
                    conn.execute("DELETE FROM qa_cache")
                conn.commit()
                remaining = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
            finally:
                conn.close()

        with _lru_lock:
            _lru_cache.clear()

        label = f"mode={mode}" if mode else "all"
        print(f"ai.py: cache cleared ({label}), {remaining} rows remain")
        return jsonify({"ok": True, "cleared": label, "remaining": remaining})

    return app
