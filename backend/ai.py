# backend/ai.py

import os
import sys
import pickle
import hashlib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import request, jsonify, render_template

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_FILE = os.path.join(RESULTS_DIR, "cache.pkl")
QA_CACHE_FILE = os.path.join(RESULTS_DIR, "qa_cache.pkl")
AI_FILE = __file__

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
# Load CSVs and build TF-IDF index
# ----------------------------------------------------------------------
def load_csvs_and_build_index(csv_dir, csv_files):
    ai_hash = file_hash(AI_FILE)

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
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
                print("ai.py: Loaded CSV data from cache")
                return cache["docs"], cache["meta"], cache["vectorizer"], cache["doc_vectors"]

        except Exception as e:
            print("Cache load failed:", e)

    print("Rebuilding TF-IDF index...")

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
                "row_index": int(idx),
                "raw_row": row.to_dict()
            })

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "docs": docs,
            "meta": meta,
            "vectorizer": vectorizer,
            "doc_vectors": doc_vectors,
            "timestamps": timestamps,
            "ai_hash": ai_hash
        }, f)

    return docs, meta, vectorizer, doc_vectors

DOCS, META, VECTORIZER, DOC_VECS = load_csvs_and_build_index(CSV_DIR, CSV_FILES)

# ----------------------------------------------------------------------
# QA Cache
# ----------------------------------------------------------------------
def load_qa_cache():
    if os.path.exists(QA_CACHE_FILE):
        with open(QA_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_qa_cache(cache):
    with open(QA_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

QA_CACHE = load_qa_cache()

# ----------------------------------------------------------------------
# Retrieval (TF-IDF only)
# ----------------------------------------------------------------------
def retrieve_top_rows(query, top_k=3):
    if not query.strip():
        return []

    qv = VECTORIZER.transform([query])
    sims = linear_kernel(qv, DOC_VECS).flatten()

    top_idx = sims.argsort()[::-1][:top_k]

    return [
        {"score": float(sims[i]), "meta": META[i]}
        for i in top_idx
    ]

# ----------------------------------------------------------------------
# Structured Answer
# ----------------------------------------------------------------------
def structured_answer(query):
    q = query.lower()

    if "placement" in q:
        path = os.path.join(CSV_DIR, "placement.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            row = df.iloc[0]
            return f"Example placement: {row}"

    return None

# ----------------------------------------------------------------------
# Synthesizer
# ----------------------------------------------------------------------
def synthesize_answer(query, results):
    if not results:
        return "No data found."

    lines = []
    for r in results:
        row = r["meta"]["raw_row"]
        lines.append(", ".join([f"{k}:{v}" for k, v in row.items()]))

    return "\n".join(lines)

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------
def register_ai(app):

    @app.route("/ai")
    def ai_page():
        return render_template("partials/ai.html")

    @app.route("/ai/chat", methods=["POST"])
    def chat():
        data = request.get_json()
        query = data.get("query", "")

        if query in QA_CACHE:
            return jsonify(QA_CACHE[query])

        ans = structured_answer(query)
        if ans:
            QA_CACHE[query] = {"answer": ans}
            save_qa_cache(QA_CACHE)
            return jsonify({"answer": ans})

        results = retrieve_top_rows(query)
        answer = synthesize_answer(query, results)

        res = {"answer": answer}
        QA_CACHE[query] = res
        save_qa_cache(QA_CACHE)

        return jsonify(res)

    return app