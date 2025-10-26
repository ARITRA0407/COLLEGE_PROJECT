# backend/ai.py
"""
AI routes for CSV-backed chatbot with rule-based + semantic search hybrid.

Enhancements:
- structured_answer(): handles best/worst/above/below style queries.
- fallback: TF-IDF semantic retrieval across all CSVs.
- caching: stores parsed docs, meta, vectorizer, and doc_vectors in results/cache.pkl
- qa_cache: separate cache file (results/qa_cache.pkl) that stores asked Q&A pairs.
- cache invalidates if ai.py code changes or CSV files change.
- all replies include sources list, suitable for dropdown in frontend.
"""

import os
import sys
import re
import pickle
import hashlib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Flask helpers
from flask import request, jsonify, render_template

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_FILE = os.path.join(RESULTS_DIR, "cache.pkl")       # doc/vector cache
QA_CACHE_FILE = os.path.join(RESULTS_DIR, "qa_cache.pkl") # query/answer cache
AI_FILE = __file__  # this file path

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

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def file_hash(path):
    """Return SHA256 hash of a file (small helper)."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None

def csv_timestamps(csv_dir, csv_files):
    ts = {}
    for fname in csv_files:
        path = os.path.join(csv_dir, fname)
        if os.path.exists(path):
            ts[fname] = os.path.getmtime(path)
    return ts

# ----------------------------------------------------------------------
# Data loading & caching (doc index)
# ----------------------------------------------------------------------
def load_csvs_and_build_index(csv_dir, csv_files):
    """Load CSVs, build TF-IDF index, with caching for speed and ai.py hash validation."""
    ai_hash = file_hash(AI_FILE)

    # Step 1: try loading cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)

            # Validate ai.py hash and CSV modification times
            valid = True
            if cache.get("ai_hash") != ai_hash:
                valid = False
            else:
                for fname in csv_files:
                    path = os.path.join(csv_dir, fname)
                    if not os.path.exists(path):
                        continue
                    if os.path.getmtime(path) > cache["timestamps"].get(fname, 0):
                        valid = False
                        break

            if valid:
                print("ai.py: Loaded data from cache")
                return (
                    cache["docs"],
                    cache["meta"],
                    cache["vectorizer"],
                    cache["doc_vectors"],
                )
        except Exception as e:
            print("ai.py: Failed to load cache:", e)

    # Step 2: rebuild index
    print("ai.py: Rebuilding index from CSVs...")
    docs = []
    meta = []
    timestamps = {}

    for fname in csv_files:
        path = os.path.join(csv_dir, fname)
        if not os.path.exists(path):
            continue
        timestamps[fname] = os.path.getmtime(path)

        try:
            df = pd.read_csv(path, dtype=str).fillna("")
        except Exception:
            df = pd.read_csv(
                path, dtype=str, engine="python", encoding="utf-8", errors="ignore"
            ).fillna("")

        for idx, row in df.iterrows():
            parts = []
            for col in df.columns:
                val = str(row[col]).strip()
                if val:
                    parts.append(f"{col}: {val}")
            snippet = " \n ".join(parts)
            docs.append(snippet)
            meta.append(
                {
                    "source_file": fname,
                    "row_index": int(idx),
                    "display_title": row.get(df.columns[0], "")
                    if df.shape[1] > 0
                    else fname,
                    "raw_row": row.to_dict(),
                }
            )

    if not docs:
        return [], [], None, None

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=20000
    )
    doc_vectors = vectorizer.fit_transform(docs)

    # Save to cache with ai.py hash
    try:
        with open(CACHE_FILE, "wb") as f:
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
        print("ai.py: Cache saved to", CACHE_FILE)
    except Exception as e:
        print("ai.py: Failed to save cache:", e)

    return docs, meta, vectorizer, doc_vectors


DOCS, META, VECTORIZER, DOC_VECS = load_csvs_and_build_index(CSV_DIR, CSV_FILES)

# ----------------------------------------------------------------------
# QA cache (question → answer) with invalidation
# ----------------------------------------------------------------------
def load_qa_cache():
    ai_hash = file_hash(AI_FILE)
    timestamps = csv_timestamps(CSV_DIR, CSV_FILES)

    if os.path.exists(QA_CACHE_FILE):
        try:
            with open(QA_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)

            # Validate against ai.py hash and CSV timestamps
            if cache.get("ai_hash") == ai_hash:
                valid = True
                for fname, ts in timestamps.items():
                    if ts > cache.get("timestamps", {}).get(fname, 0):
                        valid = False
                        break
                if valid:
                    return cache.get("qa", {})
        except Exception:
            pass

    # If invalid or missing, reset
    print("ai.py: Resetting QA cache")
    save_qa_cache({}, ai_hash, timestamps)
    return {}

def save_qa_cache(qa_dict, ai_hash=None, timestamps=None):
    if ai_hash is None:
        ai_hash = file_hash(AI_FILE)
    if timestamps is None:
        timestamps = csv_timestamps(CSV_DIR, CSV_FILES)

    try:
        with open(QA_CACHE_FILE, "wb") as f:
            pickle.dump(
                {"qa": qa_dict, "ai_hash": ai_hash, "timestamps": timestamps}, f
            )
    except Exception as e:
        print("ai.py: Failed to save QA cache:", e)

QA_CACHE = load_qa_cache()

# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
def retrieve_top_rows(query, top_k=3):
    if VECTORIZER is None or DOC_VECS is None or not query.strip():
        return []
    qv = VECTORIZER.transform([query])
    sims = linear_kernel(qv, DOC_VECS).flatten()
    if np.all(sims == 0):
        return []
    top_idx = sims.argsort()[::-1][:top_k]
    return [
        {"score": float(sims[i]), "doc": DOCS[i], "meta": META[i]} for i in top_idx
    ]

# ----------------------------------------------------------------------
# Rule-based structured answer
# ----------------------------------------------------------------------
def structured_answer(query: str):
    q = query.lower()

    # Placement queries
    if "placement" in q or "ctc" in q or "salary" in q:
        path = os.path.join(CSV_DIR, "placement.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path).fillna("")

        # Ensure numeric columns
        for col in ["placement_percentage", "highest_ctc", "average_ctc"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # best / highest
        if any(word in q for word in ["best", "highest", "top", "maximum"]):
            if "highest_ctc" in df.columns:
                row = df.loc[df["highest_ctc"].idxmax()]
                return f"The best placement is at {row['Institute']} ({row.get('Program','')}), with highest CTC {row['highest_ctc']} LPA and average {row.get('average_ctc','N/A')} LPA."

        # worst / lowest
        if any(word in q for word in ["worst", "lowest", "minimum"]):
            if "highest_ctc" in df.columns:
                row = df.loc[df["highest_ctc"].idxmin()]
                return f"The worst placement is at {row['Institute']} ({row.get('Program','')}), with highest CTC {row['highest_ctc']} LPA and average {row.get('average_ctc','N/A')} LPA."

        # above / greater than
        match = re.search(r"(above|greater than|over)\s+(\d+)", q)
        if match and "highest_ctc" in df.columns:
            threshold = float(match.group(2))
            filt = df[
                (df["highest_ctc"] > threshold)
                | (df.get("placement_percentage", 0) > threshold)
            ]
            if filt.empty:
                return f"No colleges found with values above {threshold}."
            names = ", ".join(filt["Institute"].unique()[:5])
            return f"Colleges with values above {threshold}: {names}"

        # below / less than
        match = re.search(r"(below|less than|under)\s+(\d+)", q)
        if match and "highest_ctc" in df.columns:
            threshold = float(match.group(2))
            filt = df[
                (df["highest_ctc"] < threshold)
                | (df.get("placement_percentage", 1000) < threshold)
            ]
            if filt.empty:
                return f"No colleges found with values below {threshold}."
            names = ", ".join(filt["Institute"].unique()[:5])
            return f"Colleges with values below {threshold}: {names}"

    # Ranking queries
    if "rank" in q or "ranking" in q:
        for year in ["2025", "2024", "2023", "2022", "2021"]:
            fname = f"rank_{year}.csv"
            path = os.path.join(CSV_DIR, fname)
            if os.path.exists(path):
                df = pd.read_csv(path).fillna("")
                if "Rank" in df.columns and "Institute" in df.columns:
                    if "best" in q or "top" in q:
                        row = df.loc[df["Rank"].astype(int).idxmin()]
                        return f"The top ranked college in {year} is {row['Institute']} with rank {row['Rank']}."
                    if "worst" in q or "lowest" in q:
                        row = df.loc[df["Rank"].astype(int).idxmax()]
                        return f"The lowest ranked college in {year} is {row['Institute']} with rank {row['Rank']}."

    return None

# ----------------------------------------------------------------------
# Fallback answer synthesis
# ----------------------------------------------------------------------
def synthesize_answer(query, top_results):
    if not top_results:
        return None, "I could not find relevant information in the CSVs."

    lines = []
    for r in top_results:
        meta = r["meta"]
        row = meta["raw_row"]
        src = meta["source_file"]
        score = r["score"]
        keys_to_show = []
        for pref in [
            "Institute Name",
            "Institute",
            "Name",
            "college",
            "college_name",
            "College",
            "Website",
            "Rank",
            "Placement",
            "Rating",
            "Location",
            "City",
            "State",
        ]:
            if pref in row and row[pref]:
                keys_to_show.append(pref)
        if not keys_to_show:
            keys_to_show = [k for k, v in row.items() if v][:6]
        line = f"{src} (score={score:.3f}) → " + ", ".join(
            [f"{k}: {row.get(k,'')}" for k in keys_to_show]
        )
        lines.append(line)

    return "\n".join(lines), None

# ----------------------------------------------------------------------
# Route registration
# ----------------------------------------------------------------------
def register_ai(app):
    @app.route("/ai")
    def ai_page():
        try:
            return render_template("partials/ai.html")
        except Exception:
            return (
                "AI partial not found. Place templates/partials/ai.html and try again.",
                404,
            )

    @app.route("/ai/chat", methods=["POST"])
    def chat():
        try:
            data = request.get_json(force=True) or {}
        except Exception:
            data = {}
        query = (data.get("query") or "").strip()
        if not query:
            return (
                jsonify({"ok": False, "answer": "Empty query", "sources": []}),
                200,
            )

        # Check QA cache first
        if query in QA_CACHE:
            cached = QA_CACHE[query]
            return jsonify({
                "ok": True,
                "answer": cached["answer"],
                "sources": cached["sources"],
                "cached": True
            }), 200

        # Rule-based first
        ans = structured_answer(query)
        if ans:
            result = {
                "ok": True,
                "answer": ans,
                "sources": [{"note": "Rule-based answer (no direct CSV rows used)"}],
            }
            QA_CACHE[query] = result
            save_qa_cache(QA_CACHE)
            return jsonify(result), 200

        # Fallback semantic search
        top = retrieve_top_rows(query, top_k=6)
        answer, error = synthesize_answer(query, top)
        if error:
            return jsonify({"ok": False, "answer": error, "sources": []}), 200
        sources = [
            {
                "source_file": r["meta"]["source_file"],
                "row_index": r["meta"]["row_index"],
                "score": r["score"],
            }
            for r in top
        ]
        result = {"ok": True, "answer": answer, "sources": sources}
        QA_CACHE[query] = result
        save_qa_cache(QA_CACHE)
        return jsonify(result), 200

    return app

# Auto-register
try:
    if "app" in sys.modules:
        main_mod = sys.modules["app"]
        main_app = getattr(main_mod, "app", None)
        if main_app is not None:
            register_ai(main_app)
            print("ai.py: auto-registered routes on main app")
except Exception as e:
    print("ai.py: auto-register failed:", e)
