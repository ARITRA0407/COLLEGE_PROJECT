import os
import sys
import json
import re
import threading
import subprocess
from collections import defaultdict

try:
    import pandas as pd
except Exception:
    pd = None
CSV_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv"
)
COLLEGE_CSV = os.path.join(CSV_DIR, "college.csv")
REVIEWS_CSV = os.path.join(CSV_DIR, "reviews.csv")
PLACEMENT_CSV = os.path.join(CSV_DIR, "placement.csv")
_LOAD_LOCK = threading.Lock()
_LOADED = False


def _safe_read_csv(path):
    if not os.path.exists(path):
        return []
    if pd is not None:
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
            return df.to_dict(orient="records")
        except Exception:
            pass
    import csv

    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k: (v if v is not None else "") for k, v in r.items()})
    except Exception:
        pass
    return rows


def _parse_latlon(val):
    if not val:
        return None, None
    s = str(val).strip()
    for sep in [",", ";", "|", " "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    return lat, lon
                except Exception:
                    pass
    m = re.findall(r"[-+]?\d{1,3}\.\d+", s)
    if len(m) >= 2:
        try:
            return float(m[0]), float(m[1])
        except Exception:
            pass
    return None, None


_COLLEGES = []
_REVIEWS = []
_PLACEMENTS = []


def _key(name):
    return (name or "").strip().lower()


COLLEGE_MAP = {}
REVIEWS_BY = defaultdict(list)
REVIEWS_RAW_BY = defaultdict(list)
PLACEMENT_BY = defaultdict(list)


def _ensure_loaded():
    global _LOADED
    global _COLLEGES
    global _REVIEWS
    global _PLACEMENTS
    global COLLEGE_MAP
    global REVIEWS_BY
    global REVIEWS_RAW_BY
    global PLACEMENT_BY
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:
            return
        _COLLEGES = _safe_read_csv(COLLEGE_CSV)
        _REVIEWS = _safe_read_csv(REVIEWS_CSV)
        _PLACEMENTS = _safe_read_csv(PLACEMENT_CSV)
        college_map = {}
        for r in _COLLEGES:
            name = (
                r.get("Institute")
                or r.get("College")
                or r.get("institute_name")
                or r.get("Name")
                or r.get("institute_name")
            )
            if not name:
                continue
            college_map[_key(name)] = r
        reviews_by = defaultdict(list)
        reviews_raw_by = defaultdict(list)
        for r in _REVIEWS:
            name = (
                r.get("Institute")
                or r.get("College")
                or r.get("institute_name")
                or r.get("name")
                or r.get("college_name")
            )
            if not name:
                continue
            key_name = _key(name)
            reviews_raw_by[key_name].append(r)
            reviews_by[key_name].append(
                {
                    "source": r.get("source")
                    or r.get("Source")
                    or r.get("reviewed_by")
                    or "",
                    "date": r.get("date") or r.get("Date") or "",
                    "rating": r.get("rating") or r.get("Rating") or "",
                    "review_text": r.get("review_text")
                    or r.get("review")
                    or r.get("text")
                    or "",
                }
            )
        placement_by = defaultdict(list)
        for r in _PLACEMENTS:
            name = (
                r.get("Institute")
                or r.get("College")
                or r.get("institute_name")
                or r.get("name")
                or r.get("college_name")
            )
            if not name:
                continue
            placement_by[_key(name)].append(r)
        COLLEGE_MAP = college_map
        REVIEWS_BY = reviews_by
        REVIEWS_RAW_BY = reviews_raw_by
        PLACEMENT_BY = placement_by
        _LOADED = True


def _aggregate_placement(rows):
    if not rows:
        return {}
    nums = {"avg_ctc": [], "median_ctc": [], "highest_ctc": [], "placed_count": []}
    ratings = []
    for r in rows:
        for k in [
            "avg_ctc",
            "average_ctc",
            "avg_ctc_in_lpa",
            "avg_ctc_lpa",
            "avg_ctc_in_inr",
        ]:
            if r.get(k):
                try:
                    nums["avg_ctc"].append(float(re.sub(r"[^\d\.\-]", "", r.get(k))))
                    break
                except Exception:
                    pass
        for k in ["median_ctc", "median"]:
            if r.get(k):
                try:
                    nums["median_ctc"].append(float(re.sub(r"[^\d\.\-]", "", r.get(k))))
                    break
                except Exception:
                    pass
        for k in ["highest_ctc", "highest"]:
            if r.get(k):
                try:
                    nums["highest_ctc"].append(
                        float(re.sub(r"[^\d\.\-]", "", r.get(k)))
                    )
                    break
                except Exception:
                    pass
        for k in ["placed_count", "placed", "num_placed"]:
            if r.get(k):
                try:
                    nums["placed_count"].append(int(re.sub(r"[^\d]", "", r.get(k))))
                    break
                except Exception:
                    pass
        for k in ["placement_rating", "placement_score", "rating"]:
            if r.get(k):
                try:
                    ratings.append(float(re.sub(r"[^\d\.\-]", "", str(r.get(k)))))
                    break
                except Exception:
                    pass
    out = {}
    if nums["avg_ctc"]:
        out["avg_ctc"] = round(sum(nums["avg_ctc"]) / len(nums["avg_ctc"]), 2)
    if nums["median_ctc"]:
        out["median_ctc"] = round(sum(nums["median_ctc"]) / len(nums["median_ctc"]), 2)
    if nums["highest_ctc"]:
        out["highest_ctc"] = max(nums["highest_ctc"])
    if nums["placed_count"]:
        out["placed_count"] = int(sum(nums["placed_count"]))
    if ratings:
        out["placement_rating"] = round(sum(ratings) / len(ratings), 2)
    return out


def _extract_placement_lists(rows):
    if not rows:
        return {
            "num_programs": 0,
            "programs": [],
            "top_recruiters": [],
            "job_profiles": [],
        }
    prog_set = set()
    recruiters = set()
    profiles = set()
    batch_re = re.compile(r"(?i)batch[\s_-]*\d{2,4}")
    for r in rows:
        prog = (
            r.get("Program")
            or r.get("program")
            or r.get("Program Name")
            or r.get("program_name")
            or r.get("Program_Name")
            or r.get("course")
        )
        if prog:
            for p in re.split(r"[;,|/]", str(prog)):
                p = p.strip()
                if p:
                    prog_set.add(p)
        for col in [
            "top_recruiter",
            "top_recruiters",
            "TopRecruiters",
            "Top_Recruiters",
            "top_recruiter_name",
            "recruiter",
            "recruiters",
        ]:
            if r.get(col):
                for part in re.split(r"[;,|/]", str(r.get(col))):
                    p = part.strip()
                    if p and not batch_re.search(p):
                        recruiters.add(p)
        for col in [
            "job_titles",
            "job_title",
            "JobTitles",
            "Job_Title",
            "job_profiles",
            "job_profile",
            "job",
        ]:
            if r.get(col):
                for part in re.split(r"[;,|/]", str(r.get(col))):
                    p = part.strip()
                    if p:
                        profiles.add(p)
    progs = sorted([x for x in prog_set if x])
    recs = sorted([x for x in recruiters if x])
    profs = sorted([x for x in profiles if x])
    return {
        "num_programs": len(progs),
        "programs": progs,
        "top_recruiters": recs,
        "job_profiles": profs,
    }


def _aggregate_review_scores(raw_rows):
    keys = [
        "sentiment_score",
        "mess_score",
        "professor_score",
        "campus_score",
        "placement_score",
        "infrastructure_score",
        "overall_aspect_score",
    ]
    variants = {}
    for k in keys:
        variants[k] = [
            k,
            k.upper(),
            k.replace("_", " "),
            k.replace("_", "").lower(),
            k.title(),
            k.capitalize(),
        ]
    sums = {k: [] for k in keys}
    for r in raw_rows:
        for k in keys:
            found = False
            for col in variants[k]:
                if r.get(col) not in (None, "", []):
                    val = r.get(col)
                    try:
                        num = float(re.sub(r"[^\d\.\-]", "", str(val)))
                        sums[k].append(num)
                        found = True
                        break
                    except Exception:
                        found = True
                        break
            if not found:
                for colname, val in r.items():
                    if colname and k in colname.lower() and val not in (None, "", []):
                        try:
                            num = float(re.sub(r"[^\d\.\-]", "", str(val)))
                            sums[k].append(num)
                            break
                        except Exception:
                            break
    out = {}
    for k in keys:
        if sums[k]:
            try:
                out[k] = round(sum(sums[k]) / len(sums[k]), 2)
            except Exception:
                out[k] = sums[k][0] if sums[k] else None
        else:
            out[k] = ""
    return out


def register_explore(app):
    from flask import jsonify, request

    if "_api_colleges" in getattr(app, "view_functions", {}):
        return

    @app.route("/explore/api/colleges")
    def _api_colleges():
        _ensure_loaded()
        names = []
        for r in _COLLEGES:
            name = (
                r.get("Institute")
                or r.get("College")
                or r.get("institute_name")
                or r.get("Name")
                or r.get("name")
            )
            if name:
                names.append(name.strip())
        names = sorted(list(set(names)))
        return jsonify({"colleges": names})

    @app.route("/explore/api/college")
    def _api_college():
        _ensure_loaded()
        q = request.args.get("name", "").strip()
        if not q:
            return jsonify({"error": "name required"}), 400
        key = _key(q)
        row = COLLEGE_MAP.get(key)
        if not row:
            for k, v in COLLEGE_MAP.items():
                if q.lower() in k:
                    row = v
                    break
        if not row:
            return jsonify({"error": "institute not found"}), 404
        inst = {}
        inst["institute_name"] = (
            row.get("Institute")
            or row.get("College")
            or row.get("institute_name")
            or row.get("Name")
            or ""
        )
        inst["district"] = row.get("District") or row.get("district") or ""
        inst["website"] = (
            row.get("Website") or row.get("website") or row.get("site") or ""
        )
        inst["logo_image"] = (
            row.get("logo_image")
            or row.get("Logo")
            or row.get("Logo_URL")
            or row.get("logo_url")
            or ""
        )
        inst["picture"] = (
            row.get("Picture")
            or row.get("picture")
            or row.get("image")
            or row.get("Photo")
            or ""
        )
        inst["programs"] = []
        inst["rank"] = row.get("rank") or row.get("Rank") or ""
        lat, lon = None, None
        for c in [
            "Latitude",
            "latitude",
            "lat",
            "Lat",
            "Location",
            "location",
            "Coordinates",
            "coordinates",
        ]:
            if row.get(c):
                lat, lon = _parse_latlon(row.get(c))
                if lat is not None:
                    break
        if (lat is None or lon is None) and (
            row.get("latitude") and row.get("longitude")
        ):
            try:
                lat = float(row.get("latitude"))
                lon = float(row.get("longitude"))
            except Exception:
                pass
        inst["latitude"] = lat
        inst["longitude"] = lon
        placement_rows = PLACEMENT_BY.get(key, [])
        placement_info = _extract_placement_lists(placement_rows)
        inst["num_programs"] = placement_info["num_programs"]
        inst["programs"] = placement_info["programs"]
        inst["top_recruiters"] = placement_info["top_recruiters"]
        inst["key_profiles"] = placement_info["job_profiles"]
        inst["placement_summary"] = _aggregate_placement(placement_rows)
        raw_rev_rows = REVIEWS_RAW_BY.get(key, [])
        review_scores = _aggregate_review_scores(raw_rev_rows)
        inst.update(review_scores)
        inst["sample_review_count"] = len(REVIEWS_BY.get(key, []))
        return jsonify(inst)

    @app.route("/explore/api/reviews")
    def _api_reviews():
        _ensure_loaded()
        from flask import request

        q = request.args.get("name", "").strip()
        if not q:
            return jsonify({"reviews": []})
        key = _key(q)
        rows = REVIEWS_BY.get(key, [])
        return jsonify({"reviews": rows})

    @app.route("/explore/api/placement")
    def _api_placement():
        _ensure_loaded()
        from flask import request

        q = request.args.get("name", "").strip()
        if not q:
            return jsonify({})
        key = _key(q)
        rows = PLACEMENT_BY.get(key, [])
        agg = _aggregate_placement(rows)
        lists = _extract_placement_lists(rows)
        agg.update(
            {
                "num_programs": lists["num_programs"],
                "programs": lists["programs"],
                "top_recruiters": lists["top_recruiters"],
                "job_profiles": lists["job_profiles"],
            }
        )
        return jsonify(agg)

    @app.route("/explore/api/reviews/refresh", methods=["POST"])
    def _api_refresh_reviews():
        payload = request.get_json(silent=True) or {}
        name = (payload.get("name") or "").strip()
        if not name:
            return jsonify({"error": "name required"}), 400
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        collect_cmd = [
            sys.executable,
            os.path.join(backend_dir, "reviewCollection.py"),
            "--provider",
            "serper",
            "--max-snippets",
            "8",
            "--mode",
            "replace",
            "--college-name",
            name,
        ]
        try:
            subprocess.run(
                collect_cmd,
                check=True,
                cwd=os.path.dirname(backend_dir),
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            return (
                jsonify(
                    {
                        "error": f"reviewCollection failed: {(e.stderr or e.stdout or '').strip()}"
                    }
                ),
                500,
            )
        update_cmd = [sys.executable, os.path.join(backend_dir, "reviewUpdate.py")]
        try:
            subprocess.run(
                update_cmd,
                check=True,
                cwd=os.path.dirname(backend_dir),
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            return (
                jsonify(
                    {
                        "error": f"reviewUpdate failed: {(e.stderr or e.stdout or '').strip()}"
                    }
                ),
                500,
            )
        global _LOADED
        _LOADED = False
        _ensure_loaded()
        return jsonify({"ok": True, "message": "Reviews and scores refreshed"})


try:
    maybe_app = sys.modules.get("app")
    if maybe_app and hasattr(maybe_app, "app"):
        register_explore(maybe_app.app)
except Exception:
    pass
__all__ = ["register_explore"]
