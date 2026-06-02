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

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TFIDF_CACHE_FILE = os.path.join(RESULTS_DIR, "tfidf_index.pkl")
DB_FILE = os.path.join(RESULTS_DIR, "ai_cache.db")
AI_FILE = __file__
RELEVANCE_THRESHOLD = 0.05
LRU_MAX = 128
MAX_HISTORY_TURNS = 20
FOLLOW_UP_TRIGGERS = (
    "why",
    "how",
    "explain",
    "clarify",
    "elaborate",
    "meaning",
    "reason",
    "what do you mean",
    "tell me more",
    "can you explain",
    "what about",
)
FOLLOW_UP_HINTS = (
    " it ",
    " this ",
    " that ",
    " those ",
    " these ",
    " they ",
    " previous",
    " above",
    " you said",
    " your answer",
    " last answer",
    " same",
)
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]
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
_lru_lock = threading.Lock()
_lru_cache: OrderedDict = OrderedDict()


# Cache layer
def _lru_get(key: str):
    with _lru_lock:
        if key in _lru_cache:
            _lru_cache.move_to_end(key)
            return dict(_lru_cache[key])
    return None


def _lru_set(key: str, value: dict):
    with _lru_lock:
        _lru_cache[key] = value
        _lru_cache.move_to_end(key)
        if len(_lru_cache) > LRU_MAX:
            _lru_cache.popitem(last=False)


_db_lock = threading.Lock()


def _db_connect():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mode ON qa_cache(mode)")
        conn.commit()
        conn.close()
    print(f"ai.py: SQLite cache ready -> {DB_FILE}")


def _db_get(key: str):
    with _db_lock:
        conn = _db_connect()
        try:
            row = conn.execute(
                "SELECT answer, mode, sources, hit_count FROM qa_cache WHERE query_key = ?",
                (key,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE qa_cache SET hit_count = hit_count + 1, last_hit_at = ? "
                    "WHERE query_key = ?",
                    (datetime.utcnow().isoformat(), key),
                )
                conn.commit()
                return {
                    "ok": True,
                    "answer": row["answer"],
                    "mode": row["mode"],
                    "sources": json.loads(row["sources"]),
                    "_cache": "sqlite",
                    "_hits": row["hit_count"] + 1,
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
                (key, query_raw, answer, mode, json.dumps(sources), now, now),
            )
            conn.commit()
        finally:
            conn.close()


def _db_stats():
    with _db_lock:
        conn = _db_connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
            db_hits = conn.execute(
                "SELECT COALESCE(SUM(hit_count),0) FROM qa_cache"
            ).fetchone()[0]
            top5 = conn.execute(
                "SELECT query_raw, hit_count FROM qa_cache ORDER BY hit_count DESC LIMIT 5"
            ).fetchall()
            by_mode = conn.execute(
                "SELECT mode, COUNT(*) as cnt FROM qa_cache GROUP BY mode"
            ).fetchall()
        finally:
            conn.close()
    return {
        "total_cached_queries": total,
        "total_cache_hits": int(db_hits),
        "lru_size": len(_lru_cache),
        "lru_max": LRU_MAX,
        "top_queries": [
            {"query": r["query_raw"], "hits": r["hit_count"]} for r in top5
        ],
        "by_mode": {r["mode"]: r["cnt"] for r in by_mode},
        "db_path": DB_FILE,
    }


def _make_key(query: str, history=None) -> str:
    raw = query.lower().strip()
    if history:
        tail = json.dumps(history[-6:], sort_keys=True, ensure_ascii=False)
        raw = f"{raw}|{tail}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_history(history):
    if not isinstance(history, list):
        return []
    cleaned = []
    for item in history[-MAX_HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            cleaned.append({"role": "user", "content": content})
        elif role in {"assistant", "model", "bot"}:
            cleaned.append({"role": "assistant", "content": content})
    return cleaned


def _is_follow_up_query(query: str, history: list) -> bool:
    if not history:
        return False
    q = f" {query.lower().strip()} "
    q_compact = q.strip()
    if len(q_compact.split()) <= 4:
        return True
    if any(q_compact.startswith(trigger) or trigger in q for trigger in FOLLOW_UP_TRIGGERS):
        return True
    if any(hint in q for hint in FOLLOW_UP_HINTS):
        return True
    return False


def _cache_get(key: str):
    hit = _lru_get(key)
    if hit:
        hit["_cache"] = "lru"
        return hit
    hit = _db_get(key)
    if hit:
        _lru_set(key, hit)
        return hit
    return None


def _cache_set(key: str, query_raw: str, payload: dict):
    _db_set(
        key, query_raw, payload["answer"], payload["mode"], payload.get("sources", [])
    )
    _lru_set(key, payload)


_db_init()


def file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


# Index loader
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
                print("ai.py: TF-IDF index loaded from cache [ok]")
                return (
                    cache["docs"],
                    cache["meta"],
                    cache["vectorizer"],
                    cache["doc_vectors"],
                )
        except Exception as e:
            print("TF-IDF cache invalid, rebuilding:", e)
    print("ai.py: Building TF-IDF index...")
    docs, meta, timestamps = [], [], {}
    for fname in csv_files:
        path = os.path.join(csv_dir, fname)
        if not os.path.exists(path):
            continue
        timestamps[fname] = os.path.getmtime(path)
        df = pd.read_csv(path, dtype=str).fillna("")
        for idx, row in df.iterrows():
            snippet = " ".join(
                [f"{col}: {row[col]}" for col in df.columns if str(row[col]).strip()]
            )[:300]
            docs.append(snippet)
            meta.append(
                {
                    "source_file": fname,
                    "row_index": int(idx),
                    "raw_row": row.to_dict(),
                }
            )
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)
    with open(TFIDF_CACHE_FILE, "wb") as f:
        pickle.dump(
            {
                "docs": docs,
                "meta": meta,
                "vectorizer": vectorizer,
                "doc_vectors": doc_vectors,
                "timestamps": timestamps,
                "ai_hash": ai_hash,
            },
            f,
        )
    print(f"ai.py: TF-IDF index built ({len(docs)} docs) [ok]")
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


def retrieve_top_rows(query, top_k=5):
    if not query.strip():
        return []
    ensure_index_loaded()
    qv = VECTORIZER.transform([query])
    sims = linear_kernel(qv, DOC_VECS).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [
        {"score": float(sims[i]), "meta": META[i]}
        for i in top_idx
        if float(sims[i]) > 0
    ]


_csv_data_lock = threading.Lock()
_csv_data_cache = {}


def _read_csv_cached(file_name):
    path = os.path.join(CSV_DIR, file_name)
    if not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    with _csv_data_lock:
        cached = _csv_data_cache.get(file_name)
        if cached and cached[0] == mtime:
            return cached[1].copy()
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
        except Exception:
            return None
        _csv_data_cache[file_name] = (mtime, df)
        return df.copy()


def _query_top_n(query_l, default=5):
    for token in query_l.replace("-", " ").split():
        if token.isdigit():
            value = int(token)
            if 1 <= value <= 20:
                return value
    return default


def _has_any(text, words):
    return any(word in text for word in words)


def _looks_like_refusal(answer):
    text = str(answer or "").lower()
    return (
        "cannot provide a ranking" in text
        or "not available in the provided context" in text
        or "cannot provide a ranking based" in text
    )


def _clean_source_row(row):
    clean = {}
    for key, value in row.items():
        if pd.isna(value):
            clean[key] = ""
        else:
            clean[key] = value.item() if hasattr(value, "item") else value
    return clean


STRUCTURED_METRICS = [
    {
        "file": "reviews.csv",
        "metric": "mess_score",
        "entity": "college_name",
        "label": "mess score",
        "aliases": ["mess", "food", "hostel food", "canteen"],
        "scale": "/5",
    },
    {
        "file": "reviews.csv",
        "metric": "professor_score",
        "entity": "college_name",
        "label": "professor score",
        "aliases": ["professor", "faculty", "teacher", "teaching"],
        "scale": "/5",
    },
    {
        "file": "reviews.csv",
        "metric": "campus_score",
        "entity": "college_name",
        "label": "campus score",
        "aliases": ["campus", "college life", "environment"],
        "scale": "/5",
    },
    {
        "file": "reviews.csv",
        "metric": "infrastructure_score",
        "entity": "college_name",
        "label": "infrastructure score",
        "aliases": ["infrastructure", "infra", "labs", "building", "facilities"],
        "scale": "/5",
    },
    {
        "file": "reviews.csv",
        "metric": "placements_score",
        "entity": "college_name",
        "label": "placement review score",
        "aliases": ["placement score", "placements score", "placement review"],
        "scale": "/5",
    },
    {
        "file": "reviews.csv",
        "metric": "overall_aspect_score",
        "entity": "college_name",
        "label": "overall review score",
        "aliases": ["overall", "review score", "reviews", "review"],
        "scale": "/5",
    },
    {
        "file": "reviews.csv",
        "metric": "sentiment_score",
        "entity": "college_name",
        "label": "sentiment score",
        "aliases": ["sentiment", "positive reviews", "student sentiment"],
        "scale": "",
    },
    {
        "file": "placement.csv",
        "metric": "highest_ctc",
        "entity": "Institute",
        "label": "highest CTC",
        "aliases": ["highest ctc", "highest package", "maximum package", "max package"],
        "scale": "",
    },
    {
        "file": "placement.csv",
        "metric": "average_ctc",
        "entity": "Institute",
        "label": "average CTC",
        "aliases": ["average ctc", "avg ctc", "average package", "package", "salary", "ctc"],
        "scale": "",
    },
    {
        "file": "placement.csv",
        "metric": "median_ctc",
        "entity": "Institute",
        "label": "median CTC",
        "aliases": ["median ctc", "median package"],
        "scale": "",
    },
    {
        "file": "placement.csv",
        "metric": "placement_rating",
        "entity": "Institute",
        "label": "placement rating",
        "aliases": ["placement rating", "placements", "placement"],
        "scale": "",
    },
    {
        "file": "placement.csv",
        "metric": "inst_rank",
        "entity": "Institute",
        "label": "institution rank",
        "aliases": ["institution rank", "college rank", "top ranked", "ranked college", "rank"],
        "higher_better": False,
        "scale": "",
    },
]


def _is_structured_metric_query(query: str):
    query_l = query.lower()
    ranking_words = [
        "best",
        "top",
        "rank",
        "ranking",
        "highest",
        "lowest",
        "based on",
        "list",
        "show",
    ]
    if not _has_any(query_l, ranking_words):
        return False
    for config in STRUCTURED_METRICS:
        names = [config["metric"], config["label"], *config["aliases"]]
        if _has_any(query_l, names):
            return True
    return False


def structured_metric_answer(query: str):
    query_l = query.lower()
    ranking_words = [
        "best",
        "top",
        "rank",
        "ranking",
        "highest",
        "lowest",
        "based on",
        "list",
        "show",
    ]
    if not _has_any(query_l, ranking_words):
        return None

    metric_config = None
    for config in STRUCTURED_METRICS:
        names = [config["metric"], config["label"], *config["aliases"]]
        if _has_any(query_l, names):
            metric_config = config
            break
    if metric_config is None:
        return None

    df = _read_csv_cached(metric_config["file"])
    if df is None:
        return None
    metric = metric_config["metric"]
    entity = metric_config["entity"]
    if metric not in df.columns or entity not in df.columns:
        return None

    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    work = work[work[entity].astype(str).str.strip() != ""]
    if metric != "inst_rank":
        work = work[work[metric] > 0]
    if work.empty:
        return None

    higher_better = metric_config.get("higher_better", True)
    if "lowest" in query_l or "low " in query_l:
        higher_better = False
    if metric == "inst_rank":
        higher_better = False

    agg_metric = "max" if higher_better else "min"
    agg_map = {metric: agg_metric}
    optional_numeric = [
        "rating",
        "sentiment_score",
        "average_ctc",
        "median_ctc",
        "highest_ctc",
        "placement_rating",
    ]
    for col in optional_numeric:
        if col in work.columns and col != metric:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            agg_map[col] = "mean"
    for col in ["Program", "source", "date"]:
        if col in work.columns:
            agg_map[col] = "first"

    grouped = (
        work.groupby(entity, as_index=False)
        .agg(agg_map)
        .sort_values(metric, ascending=not higher_better)
        .head(_query_top_n(query_l))
    )
    if grouped.empty:
        return None

    label = metric_config["label"]
    scale = metric_config.get("scale", "")
    direction = "highest" if higher_better else "lowest"
    lines = [
        f"Top colleges by {label} ({direction} values first), calculated from {metric_config['file']}:"
    ]
    sources = []
    for idx, row in enumerate(grouped.to_dict("records"), 1):
        college = str(row.get(entity, "")).strip()
        score = round(float(row.get(metric, 0.0)), 2)
        detail = f"{idx}. {college} - {label} {score}{scale}"
        for extra_col, extra_label in [
            ("Program", "program"),
            ("average_ctc", "avg CTC"),
            ("highest_ctc", "highest CTC"),
            ("placement_rating", "placement rating"),
            ("rating", "rating"),
            ("sentiment_score", "sentiment"),
        ]:
            if extra_col == metric or extra_col not in row:
                continue
            value = row.get(extra_col, "")
            if str(value).strip() == "" or pd.isna(value):
                continue
            try:
                value_text = str(round(float(value), 2))
            except Exception:
                value_text = str(value).strip()
            if value_text:
                detail += f", {extra_label} {value_text}"
        lines.append(detail)
        source_match = work[work[entity].astype(str) == college]
        source_row = source_match.iloc[0].to_dict() if not source_match.empty else row
        sources.append(
            {
                "file": metric_config["file"],
                "score": score,
                "data": _clean_source_row(source_row),
            }
        )
    lines.append(f"This is a direct database answer using the `{metric}` column.")
    return {
        "answer": "\n".join(lines),
        "mode": "database",
        "sources": sources,
    }


def call_gemini(system_prompt: str, user_message: str, history=None) -> str:
    import urllib.request
    import urllib.error

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "⚠️ GEMINI_API_KEY is not set. Please check your .env file."
    contents = []
    for turn in _normalize_history(history):
        role = "user" if turn["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    payload = json.dumps(
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 512, "temperature": 0.7},
        }
    ).encode("utf-8")
    last_http_error = None
    for model_name in GEMINI_FALLBACK_MODELS:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_http_error = (e.code, body)
            if e.code in {400, 401, 403}:
                break
            print(f"Gemini model {model_name} failed ({e.code}); trying fallback.")
            continue
        except Exception as e:
            print(f"Gemini model {model_name} error:", e)
            continue
    if last_http_error:
        print("Gemini API HTTPError:", last_http_error[1])
        return f"Gemini API error ({last_http_error[0]}). Check your API key and quota."
    return "Could not reach Gemini API with the available fallback models."


def rag_answer(query: str, results: list, history=None) -> str:
    context_lines = []
    for i, r in enumerate(results, 1):
        row = r["meta"]["raw_row"]
        source = r["meta"]["source_file"].replace(".csv", "").replace("_", " ").title()
        row_str = ", ".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
        context_lines.append(f"[{i}] ({source}) {row_str}")
    system_prompt = (
        "You are a helpful college counselor assistant for the EdVance platform. "
        "You help students understand college rankings, placements, reviews, and related information. "
        "Answer the user's question using ONLY the data provided in the context below. "
        "Be conversational, friendly, and clear. Format numbers and ranks nicely. "
        "If the context doesn't fully answer the question, say so honestly — do not make up data. "
        "Keep answers concise (2-4 sentences unless detail is needed)."
        "\n\nContext from database:\n" + "\n".join(context_lines)
    )
    return call_gemini(system_prompt, query, history=history)


def conversation_answer(query: str, history: list) -> str:
    """Follow-up replies: Gemini uses chat history only (no new CSV/RAG context)."""
    system_prompt = (
        "You are a helpful assistant for the EdVance college information platform. "
        "The user is continuing an earlier conversation. Use the chat history to interpret "
        "short or vague follow-ups such as 'why', 'how', 'explain', or 'what about that'. "
        "If they refer to your previous answer, explain the reasoning behind that answer. "
        "Do not invent new placement statistics, ranks, or college facts that were not in the "
        "conversation already. Keep your answer concise and friendly."
    )
    return call_gemini(system_prompt, query, history=history)


def fallback_answer(query: str, history=None) -> str:
    """Off-database questions: Gemini general knowledge only (no CSV context)."""
    system_prompt = (
        "You are a helpful assistant for the EdVance college information platform. "
        "The user asked a question not covered in the college database. "
        "Answer helpfully from your general knowledge only. Do not pretend you searched a database. "
        "If the question is about a specific college or ranking not in your knowledge, say so. "
        "Keep your answer concise and friendly. "
        "At the end, add a one-line note: "
        "'📌 Note: This answer is based on general knowledge, not the EdVance database.'"
    )
    return call_gemini(system_prompt, query, history=history)


# AI routes
def register_ai(app):
    @app.route("/ai")
    def ai_page():
        return render_template("partials/ai.html")

    @app.route("/ai/chat", methods=["POST"])
    def chat():
        data = request.get_json(silent=True) or {}
        query = (data.get("query") or "").strip()
        history = _normalize_history(data.get("history"))
        if not query:
            return jsonify({"ok": False, "answer": "Please enter a question."})
        key = _make_key(query, history if history else None)
        use_cache = not history
        if use_cache:
            cached = _cache_get(key)
            if cached:
                if not (
                    _is_structured_metric_query(query)
                    and _looks_like_refusal(cached.get("answer", ""))
                ):
                    print(f"ai.py: [{cached.get('_cache','?')}] cache hit - {query[:60]}")
                    return jsonify(cached)
                print(f"ai.py: bypassing stale metric cache - {query[:60]}")
        if history and _is_follow_up_query(query, history):
            answer = conversation_answer(query, history)
            payload = {
                "ok": True,
                "answer": answer,
                "mode": "followup",
                "sources": [],
            }
            print(f"ai.py: [new] followup (history={len(history)}) - {query[:60]}")
            return jsonify(payload)
        metric_answer = structured_metric_answer(query)
        if metric_answer:
            payload = {"ok": True, **metric_answer}
            if use_cache:
                _cache_set(key, query, payload)
            print(f"ai.py: [new] cached [database-metric] - {query[:60]}")
            return jsonify(payload)
        results = retrieve_top_rows(query, top_k=5)
        top_score = results[0]["score"] if results else 0.0
        if top_score >= RELEVANCE_THRESHOLD:
            answer = rag_answer(query, results, history=history or None)
            sources = [
                {
                    "file": r["meta"]["source_file"],
                    "score": round(r["score"], 4),
                    "data": r["meta"]["raw_row"],
                }
                for r in results
            ]
            mode = "database"
        else:
            answer = fallback_answer(query, history=history or None)
            sources = []
            mode = "general"
        payload = {"ok": True, "answer": answer, "mode": mode, "sources": sources}
        if use_cache:
            _cache_set(key, query, payload)
        print(f"ai.py: [new] {'cached ' if use_cache else ''}[{mode}] - {query[:60]}")
        return jsonify(payload)

    @app.route("/ai/cache-stats", methods=["GET"])
    def cache_stats():
        return jsonify(_db_stats())

    @app.route("/ai/cache-clear", methods=["POST"])
    def cache_clear():
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
