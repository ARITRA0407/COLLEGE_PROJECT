try:
    from dotenv import load_dotenv
except Exception:

    def load_dotenv():
        return False


load_dotenv()
import hashlib
import json
import os
import sys
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import safe_join
import pandas as pd
import random


def _env_flag(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default=0.0):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except Exception:
        return default


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
sys.path.append(BACKEND_DIR)
RESULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "results")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "videos")
os.makedirs(RESULT_REPORT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
try:
    from calibrated_weights import get_weight_vector
except Exception:

    def get_weight_vector(weight_key, names=None, data_root_dir=None):
        defaults = {
            "skill_assessment_score": {
                "bayesian_skill": 0.34,
                "observed_accuracy": 0.22,
                "confidence_alignment": 0.14,
                "time_efficiency": 0.10,
                "consistency": 0.08,
                "momentum": 0.06,
                "challenge_index": 0.06,
            }
        }
        weights = defaults.get(weight_key, {})
        if names is not None:
            names = list(names)
            weights = {name: float(weights.get(name, 0.0)) for name in names}
            total = sum(weights.values()) or 1.0
            return {name: value / total for name, value in weights.items()}
        return weights


# App config
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = (
    os.path.join(PROJECT_ROOT, "static")
    if os.path.exists(os.path.join(PROJECT_ROOT, "static"))
    else None
)
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CSV_FOLDER = os.path.join(PROJECT_ROOT, "csv")
user_data = {}
quiz_recommender = None
college_recommender = None
recommender = None
CollegeRecommender = None
HybridCareerRecommender = None
top_module = None
ai_module = None
d_map = {"Easy": 0.2, "Medium": 0.5, "Hard": 0.8}
_quiz_recommender_lock = threading.Lock()
_college_recommender_lock = threading.Lock()
_top_module_lock = threading.Lock()


def _ensure_model_diagrams_generated():
    try:
        from model_diagrams import generate_model_diagrams

        result = generate_model_diagrams(force=False)
        if result.get("generated"):
            print("Model diagrams generated:", ", ".join(result["generated"]))
    except Exception as exc:
        print("Warning: model diagram generation skipped:", exc)


_ensure_model_diagrams_generated()
_ai_module_lock = threading.Lock()
_quiz_payload_lock = threading.Lock()
_quiz_payload_prefetch_lock = threading.Lock()
_warmup_lock = threading.Lock()
_quiz_model_warmup_lock = threading.Lock()
_home_metrics_lock = threading.Lock()
_quiz_payload_cache = {"signature": None, "payload": None}
_quiz_payload_prefetch_state = {"signature": None, "event": None}
_warmup_started = False
_quiz_model_warmup_started = False
AUTO_MODEL_WARMUP = _env_flag("ENABLE_MODEL_WARMUP", False)
QUIZ_RESULT_WAIT_TIMEOUT = _env_float("QUIZ_RESULT_WAIT_TIMEOUT", 4.5)
QUIZ_MODEL_WARMUP_ON_BOOT = _env_flag("ENABLE_QUIZ_MODEL_WARMUP", True)
QUIZ_PERSIST_REPORTS = _env_flag("ENABLE_QUIZ_REPORT_PERSISTENCE", False)
# Data load
try:
    q_df = pd.read_csv(os.path.join(CSV_FOLDER, "QUESTIONS.csv"), encoding="latin1")
    career_df = pd.read_csv(
        os.path.join(CSV_FOLDER, "CAREER.csv"), encoding="utf-8-sig"
    )
    q_df.columns = q_df.columns.str.strip()
    career_df.columns = career_df.columns.str.strip()
    career_df.columns = [col.strip().replace("\ufeff", "") for col in career_df.columns]
    job_col = None
    for col in career_df.columns:
        if col.lower() == "job":
            job_col = col
            break
    if job_col:
        career_df.rename(columns={job_col: "Job"}, inplace=True)
        career_df["Job"] = career_df["Job"].astype(str).fillna("Unknown Job")
    d_map = {"Easy": 0.2, "Medium": 0.5, "Hard": 0.8}
    print("Quiz CSV loaded successfully from:", CSV_FOLDER)
except Exception as e:
    print("Quiz CSV load error:", e)
    q_df, career_df = None, None


def _format_compact_number(value):
    try:
        value = int(value)
    except Exception:
        return "0"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M+"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K+"
    return str(value)


def _count_csv_rows(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
            return max(sum(1 for _ in file_obj) - 1, 0)
    except Exception:
        return 0


def _build_homepage_metrics():
    metrics = {
        "college_count": 0,
        "college_display": "0",
        "rank_rows_count": 0,
        "rank_rows_display": "0",
        "rank_year_count": 0,
        "rank_years_display": "0 Years",
        "rank_year_range": "",
        "skill_category_count": 0,
        "skill_category_display": "0",
        "ml_model_count": 3,
        "ml_model_display": "3",
    }
    try:
        college_path = os.path.join(CSV_FOLDER, "college.csv")
        college_df = pd.read_csv(
            college_path,
            usecols=lambda col: str(col).strip().lower() == "institute",
            encoding="latin1",
        )
        if not college_df.empty:
            institute_col = college_df.columns[0]
            metrics["college_count"] = int(
                college_df[institute_col].dropna().astype(str).str.strip().nunique()
            )
            metrics["college_display"] = str(metrics["college_count"])
    except Exception:
        pass
    rank_files = sorted(
        file_name
        for file_name in os.listdir(CSV_FOLDER)
        if file_name.startswith("rank_") and file_name.endswith(".csv")
    )
    metrics["rank_year_count"] = len(rank_files)
    metrics["rank_years_display"] = (
        f"{len(rank_files)} Years" if rank_files else "0 Years"
    )
    rank_years = []
    for file_name in rank_files:
        metrics["rank_rows_count"] += _count_csv_rows(
            os.path.join(CSV_FOLDER, file_name)
        )
        try:
            rank_years.append(int(file_name.split("_")[-1].split(".")[0]))
        except Exception:
            continue
    metrics["rank_rows_display"] = _format_compact_number(metrics["rank_rows_count"])
    if rank_years:
        metrics["rank_year_range"] = f"{min(rank_years)}-{max(rank_years)}"
    if q_df is not None and "Category" in q_df.columns:
        metrics["skill_category_count"] = int(
            q_df["Category"].astype(str).str.strip().nunique()
        )
        metrics["skill_category_display"] = str(metrics["skill_category_count"])
    return metrics


HOME_PAGE_METRICS = None


def get_homepage_metrics():
    global HOME_PAGE_METRICS
    if HOME_PAGE_METRICS is not None:
        return HOME_PAGE_METRICS
    with _home_metrics_lock:
        if HOME_PAGE_METRICS is None:
            HOME_PAGE_METRICS = _build_homepage_metrics()
    return HOME_PAGE_METRICS


def get_quiz_recommender():
    global quiz_recommender
    global HybridCareerRecommender
    if quiz_recommender is not None or career_df is None:
        return quiz_recommender
    with _quiz_recommender_lock:
        if quiz_recommender is not None or career_df is None:
            return quiz_recommender
        if HybridCareerRecommender is None:
            try:
                from quiz_recommender import HybridCareerRecommender as quiz_model_class

                HybridCareerRecommender = quiz_model_class
            except Exception as import_error:
                print("Error importing quiz recommender module:", import_error)
                return None
        try:
            quiz_recommender = HybridCareerRecommender(career_df, RESULT_REPORT_DIR)
            print("Quiz recommender initialized.")
        except Exception as recommender_error:
            quiz_recommender = None
            print("Quiz recommender initialization failed:", recommender_error)
    return quiz_recommender


def _start_quiz_model_warmup():
    global _quiz_model_warmup_started
    if career_df is None or quiz_recommender is not None:
        return
    with _quiz_model_warmup_lock:
        if (
            _quiz_model_warmup_started
            or quiz_recommender is not None
            or career_df is None
        ):
            return
        _quiz_model_warmup_started = True

    def _warmup_task():
        global _quiz_model_warmup_started
        try:
            if get_quiz_recommender() is not None:
                print("Quiz recommender quiz-session warmup complete.")
                return
        except Exception as exc:
            print("Quiz recommender quiz-session warmup failed:", exc)
        with _quiz_model_warmup_lock:
            if quiz_recommender is None:
                _quiz_model_warmup_started = False

    threading.Thread(
        target=_warmup_task, name="quiz-recommender-session-warmup", daemon=True
    ).start()


def get_college_recommender():
    global college_recommender
    global CollegeRecommender
    global recommender
    if college_recommender is not None:
        return college_recommender
    with _college_recommender_lock:
        if college_recommender is not None:
            return college_recommender
        if CollegeRecommender is None:
            try:
                from recommendation import CollegeRecommender as college_model_class

                CollegeRecommender = college_model_class
            except Exception as import_error:
                print("Error importing recommendation module:", import_error)
                return None
        try:
            college_recommender = CollegeRecommender(data_root_dir=PROJECT_ROOT)
            recommender = college_recommender
            print("Recommender initialized.")
        except Exception as recommender_error:
            college_recommender = None
            recommender = None
            print("Error initializing recommender:", recommender_error)
    return college_recommender


def _start_background_warmup():
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

    def _warmup_task():
        try:
            quiz_model = get_quiz_recommender()
            if quiz_model is not None:
                print("Quiz recommender warmup complete.")
        except Exception as exc:
            print("Quiz recommender warmup failed:", exc)
        try:
            college_model = get_college_recommender()
            if college_model is not None:
                print("College recommender warmup complete.")
        except Exception as exc:
            print("College recommender warmup failed:", exc)

    threading.Thread(
        target=_warmup_task, name="recommender-warmup", daemon=True
    ).start()


def _warm_quiz_model_on_boot():
    if (
        not QUIZ_MODEL_WARMUP_ON_BOOT
        or career_df is None
        or quiz_recommender is not None
    ):
        return
    try:
        if get_quiz_recommender() is not None:
            print("Quiz recommender boot warmup complete.")
    except Exception as exc:
        print("Quiz recommender boot warmup failed:", exc)


def _quiz_payload_signature():
    snapshot = {
        "category_order": user_data.get("category_order", []),
        "answer_log": user_data.get("answer_log", []),
        "state": user_data.get("state", {}),
    }
    try:
        payload = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        payload = str(snapshot).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reset_quiz_payload_cache():
    with _quiz_payload_lock:
        _quiz_payload_cache["signature"] = None
        _quiz_payload_cache["payload"] = None
    with _quiz_payload_prefetch_lock:
        _quiz_payload_prefetch_state["signature"] = None
        _quiz_payload_prefetch_state["event"] = None


def _get_cached_quiz_payload(signature=None):
    with _quiz_payload_lock:
        cached_signature = _quiz_payload_cache.get("signature")
        cached_payload = _quiz_payload_cache.get("payload")
    if signature is not None and cached_signature != signature:
        return None
    return cached_payload


def _store_cached_quiz_payload(signature, payload):
    with _quiz_payload_lock:
        _quiz_payload_cache["signature"] = signature
        _quiz_payload_cache["payload"] = payload


def _attach_quiz_report(payload, store_history=False):
    if not payload:
        return payload
    if payload.get("report_paths"):
        return payload
    if not QUIZ_PERSIST_REPORTS:
        hydrated_payload = dict(payload)
        hydrated_payload["report_paths"] = {}
        hydrated_payload["report_file"] = ""
        return hydrated_payload
    report_paths = {}
    active_quiz_recommender = get_quiz_recommender()
    if active_quiz_recommender is not None:
        _, report_paths = active_quiz_recommender.build_report(
            category_metrics=payload.get("result", {}),
            recommendations=payload.get("rec", []),
            summary_metrics=payload.get("summary_metrics", {}),
            store_history=store_history,
        )
    hydrated_payload = dict(payload)
    hydrated_payload["report_paths"] = report_paths
    hydrated_payload["report_file"] = (
        os.path.basename(report_paths.get("latest", "")) if report_paths else ""
    )
    return hydrated_payload


def _refresh_quiz_payload_cache(
    persist_report=False, store_history=False, wait_timeout=2.5
):
    signature = _quiz_payload_signature()
    cached_payload = _get_cached_quiz_payload(signature)
    if cached_payload is not None:
        has_report = bool(cached_payload.get("report_paths"))
        if not persist_report or has_report:
            return cached_payload
        cached_payload = _attach_quiz_report(
            cached_payload, store_history=store_history
        )
        _store_cached_quiz_payload(signature, cached_payload)
        return cached_payload
    wait_event = None
    with _quiz_payload_prefetch_lock:
        if _quiz_payload_prefetch_state.get("signature") == signature:
            wait_event = _quiz_payload_prefetch_state.get("event")
    if wait_event is not None and wait_timeout > 0:
        wait_event.wait(timeout=wait_timeout)
        cached_payload = _get_cached_quiz_payload(signature)
        if cached_payload is not None:
            has_report = bool(cached_payload.get("report_paths"))
            if not persist_report or has_report:
                return cached_payload
            cached_payload = _attach_quiz_report(
                cached_payload, store_history=store_history
            )
            _store_cached_quiz_payload(signature, cached_payload)
            return cached_payload
    payload = _build_quiz_result_payload(
        persist_report=persist_report, store_history=store_history
    )
    _store_cached_quiz_payload(signature, payload)
    return payload


def _schedule_quiz_payload_refresh(persist_report=False, store_history=False):
    if not user_data.get("answer_log"):
        return
    signature = _quiz_payload_signature()
    cached_payload = _get_cached_quiz_payload(signature)
    if cached_payload is not None:
        has_report = bool(cached_payload.get("report_paths"))
        if not persist_report or has_report:
            return
    with _quiz_payload_prefetch_lock:
        if _quiz_payload_prefetch_state.get("signature") == signature:
            return
        _quiz_payload_prefetch_state["signature"] = signature
        _quiz_payload_prefetch_state["event"] = threading.Event()

    def _prefetch_worker():
        event = None
        with _quiz_payload_prefetch_lock:
            if _quiz_payload_prefetch_state.get("signature") == signature:
                event = _quiz_payload_prefetch_state.get("event")
        try:
            payload = _build_quiz_result_payload(
                persist_report=persist_report, store_history=store_history
            )
            _store_cached_quiz_payload(signature, payload)
        except Exception as exc:
            print("Quiz payload prefetch failed:", exc)
        finally:
            if event is not None:
                event.set()
            with _quiz_payload_prefetch_lock:
                if _quiz_payload_prefetch_state.get("signature") == signature:
                    _quiz_payload_prefetch_state["signature"] = None
                    _quiz_payload_prefetch_state["event"] = None

    threading.Thread(
        target=_prefetch_worker, name="quiz-payload-prefetch", daemon=True
    ).start()


if hasattr(app, "before_serving") and AUTO_MODEL_WARMUP:

    @app.before_serving
    def _before_serving_warmup():
        _start_background_warmup()

elif AUTO_MODEL_WARMUP:

    @app.before_request
    def _before_request_warmup():
        _start_background_warmup()


_warm_quiz_model_on_boot()


def get_question(category, difficulty, asked):
    category = str(category).strip()
    df = q_df[
        (q_df["Category"].astype(str).str.strip() == category)
        & (~q_df["Question ID"].isin(asked))
    ]
    if df.empty:
        return None
    df_diff = df[df["Difficulty"] == difficulty]
    if not df_diff.empty:
        df = df_diff
    q = df.sample(1).iloc[0].to_dict()
    options = [
        str(q.get("Option1", "") or ""),
        str(q.get("Option2", "") or ""),
        str(q.get("Option3", "") or ""),
    ]
    weights = [
        float(q.get("Weight1", 0) or 0),
        float(q.get("Weight2", 0) or 0),
        float(q.get("Weight3", 0) or 0),
    ]
    clean_options = []
    clean_weights = []
    for o, w in zip(options, weights):
        o = str(o).strip()
        if o and o.lower() != "nan":
            clean_options.append(o)
            clean_weights.append(w)
    if len(clean_options) == 0:
        clean_options = ["A", "B", "C"]
        clean_weights = [1, 0, 0]
    combined = list(zip(clean_options, clean_weights))
    random.shuffle(combined)
    opt, wgt = zip(*combined)
    q["Option1"] = opt[0]
    q["Weight1"] = wgt[0]
    q["Option2"] = opt[1] if len(opt) > 1 else ""
    q["Weight2"] = wgt[1] if len(opt) > 1 else 0
    q["Option3"] = opt[2] if len(opt) > 2 else ""
    q["Weight3"] = wgt[2] if len(opt) > 2 else 0
    q["Correct Answer"] = opt[list(wgt).index(max(wgt))]
    return q


def _safe_mean(values, default=0.0):
    clean_values = [float(v) for v in values if v is not None]
    return (sum(clean_values) / len(clean_values)) if clean_values else default


def _clip(value, low=0.0, high=1.0):
    try:
        value = float(value)
    except Exception:
        value = low
    return max(low, min(high, value))


# Quiz scoring
def _priority_categories():
    return [
        "Logical_Reasoning",
        "Math_Reasoning",
        "Analytical_Reasoning",
        "Coding_Skill",
        "Coding_Int",
        "Data_Mining",
        "AI_Int",
        "System_Opt",
        "DB_Design",
        "Web_Arch",
        "Low_Level",
        "Cloud_Ops",
        "Design_Int",
        "User_Empathy",
        "Risk_Eval",
        "Verbal_Reasoning",
        "Crypto_Focus",
    ]


def _assign_rank_levels(final):
    priority = _priority_categories()
    sorted_items = sorted(
        final.items(),
        key=lambda item: (
            -item[1]["score"],
            priority.index(item[0]) if item[0] in priority else 999,
        ),
    )
    top_6 = {category for category, _ in sorted_items[:6]}
    mid_5 = {category for category, _ in sorted_items[6:11]}
    for category in final.keys():
        if category in top_6:
            final[category]["level"] = "High"
        elif category in mid_5:
            final[category]["level"] = "Medium"
        else:
            final[category]["level"] = "Low"
    return dict(sorted(final.items(), key=lambda item: item[1]["score"], reverse=True))


def _fallback_career_match(final_scores):
    if career_df is None:
        return []

    def match(user_levels, row):
        score = 0
        total = 0
        lvl = {"low": 1, "medium": 2, "high": 3}
        for col in career_df.columns:
            if col == "Job":
                continue
            total += 1
            user_level = user_levels.get(col, {"level": "Low"})["level"].lower()
            required_level = str(row[col]).lower()
            if lvl.get(user_level, 1) >= lvl.get(required_level, 1):
                score += 1
        return (score / total) * 100 if total else 0

    df = career_df.copy()
    df["score"] = df.apply(lambda row: match(final_scores, row), axis=1)
    df = df[df["score"] > 20].sort_values("score", ascending=False)
    return df.head(5).to_dict(orient="records")


def _build_quiz_result_payload(persist_report=False, store_history=False):
    from ai_engine import get_skill_summary

    categories = user_data.get("category_order", [])
    answer_log = user_data.get("answer_log", [])
    state = user_data.get("state", {})
    skill_summary = get_skill_summary(state)
    final = {}
    time_avg = {}
    for category in categories:
        category_answers = [
            item
            for item in answer_log
            if str(item.get("category", "")).strip() == category
        ]
        telemetry = state.get("telemetry", {}).get(category, {})
        bayesian_skill = float(skill_summary.get("categories", {}).get(category, 0.5))
        observed_accuracy = _safe_mean(
            [item.get("normalized_score") for item in category_answers], default=0.0
        )
        avg_time = _safe_mean(
            [item.get("time") for item in category_answers], default=0.0
        )
        time_efficiency = _safe_mean(
            [item.get("time_factor") for item in category_answers], default=0.5
        )
        confidence_alignment = _safe_mean(
            [item.get("confidence_alignment") for item in category_answers],
            default=float(telemetry.get("confidence_alignment", 0.5)),
        )
        consistency = _safe_mean(
            [item.get("consistency") for item in category_answers],
            default=float(telemetry.get("consistency", 0.5)),
        )
        momentum = _safe_mean(
            [item.get("momentum") for item in category_answers],
            default=float(telemetry.get("momentum", 0.5)),
        )
        uncertainty = _safe_mean(
            [item.get("uncertainty") for item in category_answers],
            default=float(telemetry.get("uncertainty", 0.5)),
        )
        reliability = _safe_mean(
            [item.get("reliability") for item in category_answers],
            default=float(telemetry.get("reliability", 0.5)),
        )
        challenge_index = _safe_mean(
            [
                (float(d_map.get(item.get("difficulty", "Medium"), 0.5)) / 0.8)
                * float(item.get("normalized_score", 0.0))
                for item in category_answers
            ],
            default=0.0,
        )
        skill_weights = get_weight_vector(
            "skill_assessment_score",
            names=[
                "bayesian_skill",
                "observed_accuracy",
                "confidence_alignment",
                "time_efficiency",
                "consistency",
                "momentum",
                "challenge_index",
            ],
            data_root_dir=PROJECT_ROOT,
        )
        blended_score = (
            (float(skill_weights.get("bayesian_skill", 0.0)) * bayesian_skill)
            + (float(skill_weights.get("observed_accuracy", 0.0)) * observed_accuracy)
            + (
                float(skill_weights.get("confidence_alignment", 0.0))
                * confidence_alignment
            )
            + (float(skill_weights.get("time_efficiency", 0.0)) * time_efficiency)
            + (float(skill_weights.get("consistency", 0.0)) * consistency)
            + (float(skill_weights.get("momentum", 0.0)) * momentum)
            + (float(skill_weights.get("challenge_index", 0.0)) * challenge_index)
        )
        blended_score = _clip(blended_score * (1.0 - (0.12 * uncertainty)))
        final[category] = {
            "score": round(blended_score, 3),
            "level": "Low",
            "bayesian_skill": round(bayesian_skill, 3),
            "observed_accuracy": round(observed_accuracy, 3),
            "time_efficiency": round(time_efficiency, 3),
            "confidence_alignment": round(confidence_alignment, 3),
            "consistency": round(consistency, 3),
            "momentum": round(momentum, 3),
            "uncertainty": round(uncertainty, 3),
            "reliability": round(reliability, 3),
            "challenge_index": round(challenge_index, 3),
            "attempts": len(category_answers),
        }
        time_avg[category] = round(avg_time, 2)
    final = _assign_rank_levels(final)
    ordered_items = list(final.items())
    summary_metrics = {
        "questions_answered": len(answer_log),
        "avg_skill": round(float(skill_summary.get("avg_skill", 0.0)), 3),
        "avg_reliability": round(float(skill_summary.get("avg_reliability", 0.0)), 3),
        "avg_uncertainty": round(float(skill_summary.get("avg_uncertainty", 0.0)), 3),
        "avg_confidence_alignment": round(
            _safe_mean(
                [item["confidence_alignment"] for item in final.values()], default=0.0
            ),
            3,
        ),
        "avg_time_efficiency": round(
            _safe_mean(
                [item["time_efficiency"] for item in final.values()], default=0.0
            ),
            3,
        ),
        "top_strengths": [item[0] for item in ordered_items[:3]],
        "growth_areas": (
            [item[0] for item in ordered_items[-2:]] if ordered_items else []
        ),
    }
    active_quiz_recommender = get_quiz_recommender()
    report_paths = {}
    if active_quiz_recommender is not None:
        recommendations = active_quiz_recommender.recommend(
            user_scores={key: value["score"] for key, value in final.items()},
            user_levels={key: value["level"] for key, value in final.items()},
            category_metrics=final,
            top_n=5,
        )
        if persist_report:
            _, report_paths = active_quiz_recommender.build_report(
                category_metrics=final,
                recommendations=recommendations,
                summary_metrics=summary_metrics,
                store_history=store_history,
            )
        model_metrics = getattr(active_quiz_recommender, "model_metrics", {})
    else:
        recommendations = _fallback_career_match(final)
        model_metrics = {}
    return {
        "result": final,
        "rec": recommendations,
        "time_avg": time_avg,
        "summary_metrics": summary_metrics,
        "report_paths": report_paths,
        "report_file": (
            os.path.basename(report_paths.get("latest", "")) if report_paths else ""
        ),
        "model_metrics": model_metrics,
    }


@app.route("/start", methods=["POST"])
def start_quiz():
    if q_df is None:
        return jsonify([])
    q_df["Category"] = q_df["Category"].astype(str).str.strip()
    categories = sorted(q_df["Category"].unique())
    user_data.clear()
    _reset_quiz_payload_cache()
    user_data.update(
        {
            "asked": [],
            "scores": {c: [] for c in categories},
            "time": {c: [] for c in categories},
            "last_diff": {c: "Medium" for c in categories},
            "state": {},
            "category_order": categories,
            "answer_log": [],
            "report_paths": {},
        }
    )
    _start_quiz_model_warmup()
    questions = []
    for c in categories:
        q = get_question(c, "Medium", user_data["asked"])
        if q:
            user_data["asked"].append(q["Question ID"])
            questions.append(q)
    return jsonify(questions)


@app.route("/next", methods=["POST"])
def next_question():
    if q_df is None:
        return jsonify([])
    answers = request.json
    if not answers:
        return jsonify([])
    new_questions = []
    from ai_engine import process_answer

    for ans in answers:
        c = str(ans.get("category", "")).strip()
        w = float(ans.get("weight", 0))
        all_w = [
            float(item) for item in ans.get("all_weights", [w]) if item is not None
        ]
        time_sec = float(ans.get("time", 5))
        confidence = float(ans.get("confidence", 50)) / 100
        question_id = str(ans.get("question_id", "")).strip()
        question_text = str(ans.get("question_text", "")).strip()
        difficulty = str(ans.get("difficulty", "Medium")).strip() or "Medium"
        selected_option = str(ans.get("selected_option", "")).strip()
        try:
            max_weight = max(all_w) if all_w else (w if w > 0 else 1.0)
            correct = w >= max_weight if all_w else False
        except:
            max_weight = w if w > 0 else 1.0
            correct = False
        normalized_score = _clip((w / max_weight) if max_weight else 0.0)
        prev_diff = user_data["last_diff"].get(c, "Medium")
        result = process_answer(
            state=user_data["state"],
            category=c,
            score=normalized_score,
            time_taken=time_sec,
            correct=correct,
            last_diff=prev_diff,
            confidence=confidence,
        )
        user_data["state"] = result["updated_state"]
        next_diff = result["next_difficulty"]
        user_data["scores"].setdefault(c, [])
        user_data["time"].setdefault(c, [])
        user_data["last_diff"].setdefault(c, "Medium")
        user_data["time"][c].append(time_sec)
        user_data["scores"][c].append(max(0.0, min(1.0, normalized_score * confidence)))
        user_data.setdefault("answer_log", []).append(
            {
                "question_id": question_id,
                "question_text": question_text,
                "category": c,
                "difficulty": difficulty,
                "selected_option": selected_option,
                "selected_weight": round(w, 4),
                "max_weight": round(max_weight, 4),
                "normalized_score": round(normalized_score, 4),
                "correct": bool(correct),
                "time": round(time_sec, 3),
                "confidence": round(confidence, 4),
                "confidence_alignment": result.get("confidence_alignment", 0.5),
                "consistency": result.get("consistency", 0.5),
                "momentum": result.get("momentum", 0.5),
                "uncertainty": result.get("uncertainty", 0.5),
                "reliability": result.get("reliability", 0.5),
                "time_factor": result.get("time_factor", 0.5),
                "skill": result.get("skill", 0.5),
                "next_difficulty": next_diff,
            }
        )
        q = get_question(c, next_diff, user_data["asked"])
        if q:
            user_data["asked"].append(q["Question ID"])
            user_data["last_diff"][c] = next_diff
            new_questions.append(q)
    if not new_questions:
        remaining = q_df[~q_df["Question ID"].isin(user_data["asked"])]
        if not remaining.empty:
            q = remaining.sample(1).iloc[0].to_dict()
            user_data["asked"].append(q["Question ID"])
            new_questions.append(q)
    answer_count = len(user_data.get("answer_log", []))
    if answer_count >= 15:
        _start_quiz_model_warmup()
        _schedule_quiz_payload_refresh(persist_report=False, store_history=False)
    return jsonify(new_questions)


@app.route("/quiz-finalize", methods=["POST"])
def quiz_finalize():
    if career_df is None:
        return jsonify({"status": "error", "message": "Career data not loaded"}), 500
    if not user_data.get("answer_log"):
        return jsonify({"status": "error", "message": "No quiz progress found"}), 400
    _start_quiz_model_warmup()
    _schedule_quiz_payload_refresh(persist_report=False, store_history=False)
    payload = _refresh_quiz_payload_cache(
        persist_report=QUIZ_PERSIST_REPORTS,
        store_history=False,
        wait_timeout=QUIZ_RESULT_WAIT_TIMEOUT,
    )
    user_data["report_paths"] = payload.get("report_paths", {})
    return jsonify(
        {
            "status": "ready",
            "report_file": payload.get("report_file", ""),
            "questions_answered": payload.get("summary_metrics", {}).get(
                "questions_answered", 0
            ),
        }
    )


@app.route("/quiz-result")
def quiz_result():
    if career_df is None:
        return "Career data not loaded", 500
    _start_quiz_model_warmup()
    _schedule_quiz_payload_refresh(persist_report=False, store_history=False)
    payload = _refresh_quiz_payload_cache(
        persist_report=QUIZ_PERSIST_REPORTS,
        store_history=False,
        wait_timeout=QUIZ_RESULT_WAIT_TIMEOUT,
    )
    user_data["report_paths"] = payload.get("report_paths", {})
    return render_template(
        "partials/result.html",
        result=payload["result"],
        rec=payload["rec"],
        time_avg=payload["time_avg"],
        summary_metrics=payload["summary_metrics"],
        report_paths=payload["report_paths"],
        report_file=payload["report_file"],
        model_metrics=payload["model_metrics"],
    )


try:
    from explore import register_explore

    register_explore(app)
except Exception as _e:
    print("Warning: could not register explore routes:", _e)
try:
    import top as top_module

    print("Imported top module (for top-ranked colleges).")
except Exception as _e:
    top_module = None
    print("Could not import top module (top endpoints may not be available):", _e)
if top_module is not None:
    try:

        @app.route("/top")
        def top_preview():
            try:
                return render_template("partials/top.html")
            except Exception as e:
                return f"<h2>Template error</h2><pre>{e}</pre>", 500

        @app.route("/top/data")
        def top_data():
            try:
                if hasattr(top_module, "load_top10"):
                    data = top_module.load_top10()
                    return jsonify(data)
                else:
                    return (
                        jsonify({"error": "top module missing load_top10 function"}),
                        500,
                    )
            except Exception as e:
                print("Error in /top/data:", e)
                return jsonify({"error": str(e)}), 500

    except Exception as _e:
        print("Warning: could not register /top endpoints:", _e)
try:
    import ai as ai_module

    print("Imported ai module (AI endpoints available).")
    try:
        if hasattr(ai_module, "register_ai"):
            ai_module.register_ai(app)
            print("Registered AI routes via ai.register_ai(app).")
    except Exception as _e_inner:
        print("AI module present but register_ai(app) failed:", _e_inner)
except Exception as _e:
    ai_module = None
    print("Could not import ai module (AI routes may not be available):", _e)
CSV_FOLDER = os.path.join(PROJECT_ROOT, "csv")


@app.route("/csv/<path:filename>")
def serve_csv(filename):
    try:
        requested = safe_join(CSV_FOLDER, filename)
        if not requested or not os.path.exists(requested):
            return "Not found", 404
        return send_from_directory(CSV_FOLDER, filename, conditional=True)
    except Exception as e:
        print("Error serving csv:", e)
        abort(404)


@app.route("/videos/<path:filename>")
def serve_video(filename):
    try:
        requested = safe_join(VIDEO_DIR, filename)
        if not requested or not os.path.exists(requested):
            return "Not found", 404
        return send_from_directory(VIDEO_DIR, filename, conditional=True)
    except Exception as e:
        print("Error serving video:", e)
        abort(404)


@app.route("/favicon.ico")
def favicon():
    if STATIC_DIR is None:
        return "", 204
    image_dir = os.path.join(STATIC_DIR, "images")
    icon_name = "logo.png"
    icon_path = os.path.join(image_dir, icon_name)
    if os.path.exists(icon_path):
        return send_from_directory(image_dir, icon_name, mimetype="image/png")
    return "", 204


# Page routes
@app.route("/")
def index():
    try:
        home_page_metrics = get_homepage_metrics()
        home_stats = [
            {
                "value": home_page_metrics.get("college_display", "0"),
                "label": "Colleges Profiled",
            },
            {
                "value": home_page_metrics.get("rank_rows_display", "0"),
                "label": "WBJEE Rank Records",
            },
            {
                "value": home_page_metrics.get("rank_years_display", "0 Years"),
                "label": "Admission Trend Window",
            },
            {
                "value": home_page_metrics.get("skill_category_display", "0"),
                "label": "Skill Categories",
            },
            {
                "value": home_page_metrics.get("ml_model_display", "3"),
                "label": "Hybrid ML Models",
            },
        ]
        return render_template(
            "index.html",
            home_metrics=home_page_metrics,
            home_stats=home_stats,
            hero_video_url="/videos/FrontVideo.mp4",
            hero_poster_url="/videos/FrontVideoPoster.png",
        )
    except Exception as e:
        return f"<h2>Template error</h2><pre>{e}</pre>", 500


@app.route("/comparison")
def comparison():
    try:
        return render_template("partials/comparision.html")
    except Exception:
        error_html = """
        <!doctype html>
        <html>
          <head><meta charset="utf-8"/><title>Comparison page missing</title></head>
          <body style="font-family: system-ui, -apple-system, 'Segoe UI', Roboto, Arial; padding:20px;">
            <h2 style="color:#c53030;">Comparison template not found</h2>
            <p>We tried to render the comparison page for the iframe but couldn't find the template.</p>
            <p>Please ensure the following template exists in <code>templates/partials/</code>:</p>
            <ul>
              <li><code>comparision.html</code></li>
            </ul>
            <p>After adding the file, reload this page.</p>
          </body>
        </html>
        """
        return error_html, 500


@app.route("/explore-colleges")
def explore_colleges():
    try:
        return render_template("partials/explore.html")
    except Exception as e:
        return (
            f"<h2>Template Error</h2><p>Could not find 'partials/explore.html'.</p><pre>{e}</pre>",
            500,
        )


@app.route("/top-ranked")
def top_ranked_colleges():
    try:
        return render_template("partials/top.html")
    except Exception as e:
        return (
            f"<h2>Template Error</h2><p>Could not find 'partials/top.html'.</p><pre>{e}</pre>",
            500,
        )


@app.route("/recommendation")
def recommendation_page():
    try:
        return render_template("partials/recommendation.html")
    except Exception as e:
        return (
            f"<h2>Template Error</h2><p>Could not find 'partials/recommendation.html'.</p><pre>{e}</pre>",
            500,
        )


@app.route("/ai-guidance")
def ai_guidance():
    try:
        return render_template("partials/ai.html")
    except Exception as e:
        return (
            f"<h2>Template Error</h2><p>Could not find 'partials/ai.html'.</p><pre>{e}</pre>",
            500,
        )


@app.route("/quiz")
def quiz_page():
    try:
        return render_template("partials/quiz.html")
    except Exception as e:
        return (
            f"<h2>Template Error</h2><p>Could not find 'partials/quiz.html'.</p><pre>{e}</pre>",
            500,
        )


@app.route("/metadata", methods=["GET"])
def metadata():
    active_recommender = get_college_recommender()
    if active_recommender is None:
        return jsonify({"error": "Recommender not available"}), 503
    try:
        programs = (
            sorted(
                active_recommender.master_rank_df["Program"].dropna().unique().tolist()
            )
            if not active_recommender.master_rank_df.empty
            else []
        )
        streams = (
            sorted(
                active_recommender.master_rank_df["Stream"].dropna().unique().tolist()
            )
            if not active_recommender.master_rank_df.empty
            else []
        )
        quotas = (
            sorted(
                active_recommender.master_rank_df["Quota"].dropna().unique().tolist()
            )
            if not active_recommender.master_rank_df.empty
            else []
        )
        categories = (
            sorted(
                active_recommender.master_rank_df["Category"].dropna().unique().tolist()
            )
            if not active_recommender.master_rank_df.empty
            else []
        )
        locations = (
            sorted(active_recommender.merged_df["District"].dropna().unique().tolist())
            if not active_recommender.merged_df.empty
            else []
        )
        sort_options = [
            {
                "value": "Predicted Closing Rank",
                "label": "Predicted Closing Rank (asc)",
            },
            {"value": "Max Average CTC", "label": "Max Average CTC (desc)"},
            {"value": "placement_score", "label": "Placement Score (desc)"},
            {"value": "overall_aspect_score", "label": "Overall Score (desc)"},
            {"value": "professor_score", "label": "Professor Score (desc)"},
            {"value": "mess_score", "label": "Mess Score (desc)"},
        ]
        return jsonify(
            {
                "programs": programs,
                "streams": streams,
                "quotas": quotas,
                "categories": categories,
                "locations": locations,
                "sort_options": sort_options,
            }
        )
    except Exception as e:
        print("Error in /metadata:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/metadata/filtered", methods=["GET"])
def metadata_filtered():
    active_recommender = get_college_recommender()
    if active_recommender is None:
        return jsonify({"error": "Recommender not available"}), 503
    program = request.args.get("program", "").strip().lower()
    stream = request.args.get("stream", "").strip().lower()
    quota = request.args.get("quota", "").strip().lower()
    category = request.args.get("category", "").strip().lower()
    df = active_recommender.master_rank_df.copy()

    def col(name):
        for c in df.columns:
            if c.strip().lower() == name.lower():
                return c
        return None

    program_col = col("program")
    stream_col = col("stream")
    quota_col = col("quota")
    category_col = col("category")
    institute_col = col("institute")
    if program and program_col:
        df = df[df[program_col].astype(str).str.strip().str.lower() == program]
    if stream and stream_col:
        df = df[df[stream_col].astype(str).str.strip().str.lower() == stream]
    if quota and quota_col:
        df = df[df[quota_col].astype(str).str.strip().str.lower() == quota]
    if category and category_col:
        df = df[df[category_col].astype(str).str.strip().str.lower() == category]

    def unique_sorted(dataframe, column):
        if not column or column not in dataframe.columns:
            return []
        vals = dataframe[column].dropna().astype(str).str.strip()
        vals = vals[vals != ""].unique().tolist()
        return sorted(vals)

    streams = unique_sorted(df, stream_col)
    quotas = unique_sorted(df, quota_col)
    categories = unique_sorted(df, category_col)
    locations = []
    try:
        if institute_col and not active_recommender.merged_df.empty:
            merged = active_recommender.merged_df
            mcol = lambda n: next(
                (c for c in merged.columns if c.strip().lower() == n.lower()), None
            )
            inst_col_m = mcol("institute")
            dist_col_m = mcol("district")
            if inst_col_m and dist_col_m:
                institutes_in_filter = set(
                    df[institute_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .unique()
                )
                loc_df = merged[
                    merged[inst_col_m]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(institutes_in_filter)
                ]
                locations = unique_sorted(loc_df, dist_col_m)
    except Exception as e:
        print("metadata_filtered: location lookup error:", e)
    return jsonify(
        {
            "streams": streams,
            "quotas": quotas,
            "categories": categories,
            "locations": locations,
        }
    )


@app.route("/recommend_colleges", methods=["POST"])
def recommend_colleges():
    active_recommender = get_college_recommender()
    if active_recommender is None:
        return (
            jsonify({"status": "error", "message": "Recommender not available."}),
            503,
        )
    try:
        data = request.get_json() or {}
        user_rank = data.get("rank") or data.get("user_rank")
        user_program = data.get("program") or data.get("user_program", "")
        user_stream = data.get("stream") or data.get("user_stream", "")
        user_quota = data.get("quota") or data.get("user_quota", "")
        user_category = data.get("category") or data.get("user_category", "")
        user_location = data.get("location") or data.get("user_location", "")
        try:
            min_ctc = float(data.get("min_ctc", 0) or 0)
        except Exception:
            min_ctc = 0.0
        try:
            min_placements_score = float(data.get("min_placements_score", 0) or 0)
        except Exception:
            min_placements_score = 0.0
        target_year = int(data.get("target_year", 2026))
        top_n = int(data.get("top_n", 10))
        if (
            user_rank is None
            or str(user_rank).strip() == ""
            or str(user_program).strip() == ""
        ):
            return (
                jsonify(
                    {"status": "error", "message": "Required fields: rank and program."}
                ),
                400,
            )
        result = active_recommender.recommend(
            user_rank=user_rank,
            user_program=user_program,
            user_stream=user_stream,
            user_quota=user_quota,
            user_category=user_category,
            user_location=user_location,
            min_ctc=min_ctc,
            min_placements_score=min_placements_score,
            target_year=target_year,
        )
        if (
            isinstance(result, dict)
            and "data" in result
            and isinstance(result["data"], list)
        ):
            result["data"] = result["data"][:top_n]
        return jsonify(result)
    except Exception as e:
        print("Error in recommendation API:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/claude-proxy", methods=["POST"])
def claude_proxy():
    import json
    import urllib.request as urlreq
    import urllib.error as urlerr

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No payload"}), 400
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
        if not GEMINI_API_KEY:
            return jsonify({"error": "GEMINI_API_KEY not set"}), 500
        messages = payload.get("messages", [])
        prompt = messages[0].get("content", "") if messages else ""
        system = payload.get("system", "")
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        gemini_payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
        body = json.dumps(gemini_payload).encode("utf-8")
        req = urlreq.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urlreq.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return jsonify({"content": [{"type": "text", "text": text}]})
    except urlerr.HTTPError as e:
        return app.response_class(
            response=e.read(), status=e.code, mimetype="application/json"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug_mode = _env_flag("FLASK_DEBUG", False)
    use_reloader = debug_mode and _env_flag("FLASK_USE_RELOADER", False)
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=use_reloader)
