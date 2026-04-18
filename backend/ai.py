# backend/ai.py
"""
AI routes for CSV-backed chatbot with rule-based + semantic search hybrid.
Optimized for CPU local inference.
- Rule-based + semantic search first
- Local LLM fallback (who/what/when/where/why/how)
- Semantic search using SentenceTransformer
- Precomputed embeddings saved to file for faster queries
- 100% offline (no API keys needed)
"""

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
# Local Transformer Backend
# ----------------------------------------------------------------------
from transformers import pipeline

try:
    LOCAL_MODEL = "google/flan-t5-small"
    _pipe = pipeline("text2text-generation", model=LOCAL_MODEL)
    print(f"ai.py: Using local transformer backend ({LOCAL_MODEL})")
except Exception as e:
    print("ai.py: Local transformer init failed:", e)
    _pipe = None

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_FILE = os.path.join(RESULTS_DIR, "cache.pkl")
QA_CACHE_FILE = os.path.join(RESULTS_DIR, "qa_cache.pkl")
EMB_FILE = os.path.join(RESULTS_DIR, "doc_embeddings.pkl")  # For precomputed embeddings
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
                print("ai.py: Loaded CSV data and TF-IDF vectors from cache")
                return cache["docs"], cache["meta"], cache["vectorizer"], cache["doc_vectors"]
        except Exception as e:
            print("ai.py: Failed to load cache:", e)

    print("ai.py: Rebuilding index from CSVs...")
    docs, meta, timestamps = [], [], {}
    for fname in csv_files:
        path = os.path.join(csv_dir, fname)
        if not os.path.exists(path):
            continue
        timestamps[fname] = os.path.getmtime(path)
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
        except Exception:
            df = pd.read_csv(path, dtype=str, engine="python", encoding="utf-8", errors="ignore").fillna("")
        for idx, row in df.iterrows():
            snippet = " \n ".join([f"{col}: {str(row[col]).strip()}" for col in df.columns if str(row[col]).strip()])[:300]
            docs.append(snippet)
            meta.append({
                "source_file": fname,
                "row_index": int(idx),
                "display_title": row.get(df.columns[0], ""),
                "raw_row": row.to_dict(),
            })

    if not docs:
        return [], [], None, None

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    doc_vectors = vectorizer.fit_transform(docs)
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump({
                "docs": docs,
                "meta": meta,
                "vectorizer": vectorizer,
                "doc_vectors": doc_vectors,
                "timestamps": timestamps,
                "ai_hash": ai_hash,
            }, f)
        print("ai.py: CSV cache saved")
    except Exception as e:
        print("ai.py: Failed to save CSV cache:", e)

    return docs, meta, vectorizer, doc_vectors

DOCS, META, VECTORIZER, DOC_VECS = load_csvs_and_build_index(CSV_DIR, CSV_FILES)

# ----------------------------------------------------------------------
# QA cache
# ----------------------------------------------------------------------
def load_qa_cache():
    ai_hash = file_hash(AI_FILE)
    timestamps = csv_timestamps(CSV_DIR, CSV_FILES)
    if os.path.exists(QA_CACHE_FILE):
        try:
            with open(QA_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
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
    save_qa_cache({}, ai_hash, timestamps)
    return {}

def save_qa_cache(qa_dict, ai_hash=None, timestamps=None):
    if ai_hash is None:
        ai_hash = file_hash(AI_FILE)
    if timestamps is None:
        timestamps = csv_timestamps(CSV_DIR, CSV_FILES)
    try:
        with open(QA_CACHE_FILE, "wb") as f:
            pickle.dump({"qa": qa_dict, "ai_hash": ai_hash, "timestamps": timestamps}, f)
    except Exception as e:
        print("ai.py: Failed to save QA cache:", e)

QA_CACHE = load_qa_cache()

# ----------------------------------------------------------------------
# Optional Semantic Search with Precomputed Embeddings
# ----------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer, util
    sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    USE_SEMANTIC = True
    print("ai.py: Semantic model loaded (all-MiniLM-L6-v2)")

    # Precompute or load embeddings
    if os.path.exists(EMB_FILE):
        print("ai.py: Loading precomputed embeddings from file...")
        with open(EMB_FILE, "rb") as f:
            DOC_EMBS = pickle.load(f)
        print("ai.py: Precomputed embeddings loaded")
    else:
        print("ai.py: Computing embeddings for all documents...")
        DOC_EMBS = sem_model.encode(DOCS, convert_to_tensor=True)
        with open(EMB_FILE, "wb") as f:
            pickle.dump(DOC_EMBS, f)
        print(f"ai.py: Precomputed embeddings saved to {EMB_FILE}")

except Exception as e:
    sem_model = None
    USE_SEMANTIC = False
    DOC_EMBS = None
    print("ai.py: SentenceTransformer not available, using TF-IDF fallback:", e)

# ----------------------------------------------------------------------
# Build list of college names for precise filtering
# ----------------------------------------------------------------------
COLLEGE_NAMES = set()
for fname in ["college.csv", "reviews.csv"]:
    path = os.path.join(CSV_DIR, fname)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
            if "college_name" in df.columns:
                COLLEGE_NAMES.update([str(c).lower() for c in df["college_name"].unique()])
            elif "Institute" in df.columns:
                COLLEGE_NAMES.update([str(c).lower() for c in df["Institute"].unique()])
        except Exception:
            continue

# ----------------------------------------------------------------------
# Retrieve top rows with precise college filtering
# ----------------------------------------------------------------------
def retrieve_top_rows(query, top_k=3):
    if not query.strip():
        return []

    query_lower = query.lower()
    # Only keep docs that match a college name
    filtered_docs, filtered_meta, filtered_embs = [], [], []
    for i, doc in enumerate(DOCS):
        doc_lower = doc.lower()
        if any(col in doc_lower for col in COLLEGE_NAMES):
            filtered_docs.append(doc)
            filtered_meta.append(META[i])
            if USE_SEMANTIC and DOC_EMBS is not None:
                filtered_embs.append(DOC_EMBS[i])
    if not filtered_docs:
        # fallback to all
        filtered_docs, filtered_meta, filtered_embs = DOCS, META, DOC_EMBS

    # Semantic search
    if USE_SEMANTIC and DOC_EMBS is not None:
        try:
            q_emb = sem_model.encode(query, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(q_emb, filtered_embs)[0].cpu().numpy()
            top_idx = sims.argsort()[::-1][:top_k]
            return [{"score": float(sims[i]), "doc": filtered_docs[i], "meta": filtered_meta[i]} for i in top_idx]
        except Exception as e:
            print("Semantic retrieval failed, fallback:", e)

    # TF-IDF fallback
    if VECTORIZER is None or DOC_VECS is None:
        return []

    qv = VECTORIZER.transform([query])
    sims = linear_kernel(qv, DOC_VECS).flatten()
    if np.all(sims == 0):
        return []
    top_idx = sims.argsort()[::-1][:top_k]
    return [{"score": float(sims[i]), "doc": DOCS[i], "meta": META[i]} for i in top_idx]

# ----------------------------------------------------------------------
# Structured answer
# ----------------------------------------------------------------------
def structured_answer(query: str):
    q = query.lower()
    if "placement" in q or "ctc" in q or "salary" in q:
        path = os.path.join(CSV_DIR, "placement.csv")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path).fillna("")
        for col in ["placement_percentage", "highest_ctc", "average_ctc"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if any(word in q for word in ["best", "highest", "top", "maximum"]):
            if "highest_ctc" in df.columns:
                row = df.loc[df["highest_ctc"].idxmax()]
                return f"The best placement is at {row['Institute']} ({row.get('Program','')}), with highest CTC {row['highest_ctc']} LPA."
        if any(word in q for word in ["worst", "lowest", "minimum"]):
            if "highest_ctc" in df.columns:
                row = df.loc[df["highest_ctc"].idxmin()]
                return f"The worst placement is at {row['Institute']} ({row.get('Program','')}), with highest CTC {row['highest_ctc']} LPA."
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
# Synthesizer and LLM
# ----------------------------------------------------------------------
def synthesize_answer(query, top_results):
    if not top_results:
        return None, "No relevant CSV data found."
    lines = []
    for r in top_results:
        meta = r["meta"]
        row = meta["raw_row"]
        src = meta["source_file"]
        score = r["score"]
        keys = [k for k in ["Institute", "Program", "Rank", "Placement", "City", "State", "college_name"] if k in row and row[k]]
        if not keys:
            keys = list(row.keys())[:6]
        line = f"{src} (score={score:.3f}) → " + ", ".join(f"{k}: {row[k]}" for k in keys)
        lines.append(line)
    return "\n".join(lines), None

def build_prompt_from_context(query, top_results, max_chars=1500):
    prompt = [
        "You are a helpful assistant that answers user questions using the provided CSV data snippets.",
        f"User question: {query}",
        "Context snippets:",
    ]
    total = 0
    for i, r in enumerate(top_results):
        meta = r["meta"]
        src = meta["source_file"]
        ridx = meta["row_index"]
        doc = r.get("doc", "")
        piece = f"[{i}] source={src} row={ridx} score={r['score']:.3f}\n{doc}"
        if len(piece) + total > max_chars:
            piece = piece[:max(0, max_chars - total - 20)] + "..."
            prompt.append(piece)
            break
        prompt.append(piece)
        total += len(piece)
    prompt.append("Answer concisely, citing [index] sources when relevant.")
    return "\n\n".join(prompt)

def llm_generate_answer(query, top_results):
    if _pipe is None:
        return None, "Local transformer backend unavailable."
    prompt = build_prompt_from_context(query, top_results)
    try:
        out = _pipe(prompt, max_new_tokens=64, do_sample=False)
        text = out[0]["generated_text"] if isinstance(out, list) else str(out)
        return text.strip(), None
    except Exception as e:
        return None, f"LLM generation failed: {e}"

# ----------------------------------------------------------------------
# Route registration
# ----------------------------------------------------------------------
def register_ai(app):
    @app.route("/ai")
    def ai_page():
        try:
            return render_template("partials/ai.html")
        except Exception:
            return "AI partial not found.", 404

    @app.route("/ai/chat", methods=["POST"])
    def chat():
        try:
            data = request.get_json(force=True) or {}
        except Exception:
            data = {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"ok": False, "answer": "Empty query", "sources": []}), 200

        general = ["who", "what", "when", "where", "why", "how"]
        college = ["rank", "placement", "college", "institute"]
        ql = query.lower()

        if any(ql.startswith(x) for x in general) and not any(x in ql for x in college):
            llm_text, llm_err = llm_generate_answer(query, [])
            if llm_text:
                return jsonify({"ok": True, "answer": llm_text, "sources": []}), 200
            return jsonify({"ok": False, "answer": llm_err or "No info found.", "sources": []}), 200

        if not any(x in ql for x in college):
            return jsonify({"ok": False, "answer": "This question seems outside the scope of college data.", "sources": []}), 200

        if query in QA_CACHE:
            c = QA_CACHE[query]
            return jsonify({"ok": True, "answer": c["answer"], "sources": c.get("sources", []), "cached": True}), 200

        ans = structured_answer(query)
        if ans:
            r = {"ok": True, "answer": ans, "sources": [{"note": "Rule-based answer"}]}
            QA_CACHE[query] = r
            save_qa_cache(QA_CACHE)
            return jsonify(r), 200

        top = retrieve_top_rows(query, top_k=3)
        answer, error = synthesize_answer(query, top)
        if answer and not error:
            max_score = max([r["score"] for r in top]) if top else 0
            if max_score > 0.05 and len(answer) > 20:
                s = [{"source_file": r["meta"]["source_file"], "row_index": r["meta"]["row_index"], "score": r["score"]} for r in top]
                r = {"ok": True, "answer": answer, "sources": s}
                QA_CACHE[query] = r
                save_qa_cache(QA_CACHE)
                return jsonify(r), 200

        llm_text, llm_err = llm_generate_answer(query, top)
        if llm_text:
            s = [{"source_file": r["meta"]["source_file"], "row_index": r["meta"]["row_index"], "score": r["score"]} for r in top]
            r = {"ok": True, "answer": llm_text, "sources": s}
            QA_CACHE[query] = r
            save_qa_cache(QA_CACHE)
            return jsonify(r), 200

        return jsonify({"ok": False, "answer": llm_err or error or "No relevant data found.", "sources": []}), 200

    return app

# ----------------------------------------------------------------------
# Auto-register
# ----------------------------------------------------------------------
try:
    if "app" in sys.modules:
        main_mod = sys.modules["app"]
        main_app = getattr(main_mod, "app", None)
        if main_app:
            register_ai(main_app)
            print("ai.py: auto-registered routes")
except Exception as e:
    print("ai.py: auto-register failed:", e)
