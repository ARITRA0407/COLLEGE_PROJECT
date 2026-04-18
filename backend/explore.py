# backend/explore.py
"""
Explore endpoints for the frontend. Reads CSV files from project-root/csv and exposes:
 - /explore/api/colleges              -> { colleges: [name1, name2, ...] }
 - /explore/api/college?name=...       -> details JSON for a single institute
 - /explore/api/reviews?name=...       -> reviews list for institute
 - /explore/api/placement?name=...     -> aggregated placement stats for institute
This module will try to auto-register routes if an `app` Flask instance is available
in `sys.modules['app']`. Otherwise use register_explore(app) to register manually.
"""
import os
import sys
import json
import re
from collections import defaultdict

# optional dependency
try:
    import pandas as pd
except Exception:
    pd = None

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'csv')
# expected files
COLLEGE_CSV = os.path.join(CSV_DIR, 'college.csv')
REVIEWS_CSV = os.path.join(CSV_DIR, 'reviews.csv')
PLACEMENT_CSV = os.path.join(CSV_DIR, 'placement.csv')

def _safe_read_csv(path):
    """Read CSV into list of dicts; use pandas if available, else fallback."""
    if not os.path.exists(path):
        return []
    if pd is not None:
        try:
            df = pd.read_csv(path, dtype=str).fillna('')
            return df.to_dict(orient='records')
        except Exception:
            pass
    # fallback: simple csv reader
    import csv
    rows = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({k: (v if v is not None else '') for k,v in r.items()})
    except Exception:
        pass
    return rows

def _parse_latlon(val):
    """Try to extract lat and lon from a string using regex.
       Handles formats like '12.345,78.901', '12.345 78.901', '12.345|78.901' or '12.345;78.901'
    """
    if not val:
        return None, None
    s = str(val).strip()
    # common separators
    for sep in [',', ';', '|', ' ']:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    return lat, lon
                except Exception:
                    pass
    # try to find two floating numbers via regex
    m = re.findall(r'[-+]?\d{1,3}\.\d+', s)
    if len(m) >= 2:
        try:
            return float(m[0]), float(m[1])
        except Exception:
            pass
    return None, None

# Load CSVs in memory for quick responses (reloaded on import)
_COLLEGES = _safe_read_csv(COLLEGE_CSV)
_REVIEWS = _safe_read_csv(REVIEWS_CSV)
_PLACEMENTS = _safe_read_csv(PLACEMENT_CSV)

# Create mapping by institute name (case-insensitive key)
def _key(name):
    return (name or '').strip().lower()

COLLEGE_MAP = {}
for r in _COLLEGES:
    name = r.get('Institute') or r.get('College') or r.get('institute_name') or r.get('Name') or r.get('institute_name')
    if not name:
        continue
    keyn = _key(name)
    COLLEGE_MAP[keyn] = r

# REVIEWS: build two structures:
#  - REVIEWS_BY: list of simplified review entries for display
#  - REVIEWS_RAW_BY: raw rows for aggregating numeric score columns
REVIEWS_BY = defaultdict(list)
REVIEWS_RAW_BY = defaultdict(list)
for r in _REVIEWS:
    # try multiple possible columns for institute name
    name = r.get('Institute') or r.get('College') or r.get('institute_name') or r.get('name') or r.get('college_name')
    if not name:
        continue
    k = _key(name)
    REVIEWS_RAW_BY[k].append(r)
    REVIEWS_BY[k].append({
        'source': r.get('source') or r.get('Source') or r.get('reviewed_by') or '',
        'date': r.get('date') or r.get('Date') or '',
        'rating': r.get('rating') or r.get('Rating') or '',
        'review_text': r.get('review_text') or r.get('review') or r.get('text') or ''
    })

PLACEMENT_BY = defaultdict(list)
for r in _PLACEMENTS:
    name = r.get('Institute') or r.get('College') or r.get('institute_name') or r.get('name') or r.get('college_name')
    if not name:
        continue
    PLACEMENT_BY[_key(name)].append(r)

# helper aggregator for placements
def _aggregate_placement(rows):
    if not rows:
        return {}
    # try numeric conversions where possible
    nums = {'avg_ctc': [], 'median_ctc': [], 'highest_ctc': [], 'placed_count': []}
    ratings = []
    for r in rows:
        for k in ['avg_ctc','average_ctc','avg_ctc_in_lpa','avg_ctc_lpa','avg_ctc_in_inr']:
            if r.get(k):
                try:
                    nums['avg_ctc'].append(float(re.sub(r'[^\d\.\-]', '', r.get(k))))
                    break
                except Exception:
                    pass
        for k in ['median_ctc','median']:
            if r.get(k):
                try:
                    nums['median_ctc'].append(float(re.sub(r'[^\d\.\-]', '', r.get(k))))
                    break
                except Exception:
                    pass
        for k in ['highest_ctc','highest']:
            if r.get(k):
                try:
                    nums['highest_ctc'].append(float(re.sub(r'[^\d\.\-]', '', r.get(k))))
                    break
                except Exception:
                    pass
        for k in ['placed_count','placed','num_placed']:
            if r.get(k):
                try:
                    nums['placed_count'].append(int(re.sub(r'[^\d]', '', r.get(k))))
                    break
                except Exception:
                    pass
        for k in ['placement_rating','placement_score','rating']:
            if r.get(k):
                try:
                    ratings.append(float(re.sub(r'[^\d\.\-]', '', str(r.get(k)))))
                    break
                except Exception:
                    pass
    out = {}
    if nums['avg_ctc']:
        out['avg_ctc'] = round(sum(nums['avg_ctc'])/len(nums['avg_ctc']), 2)
    if nums['median_ctc']:
        out['median_ctc'] = round(sum(nums['median_ctc'])/len(nums['median_ctc']), 2)
    if nums['highest_ctc']:
        out['highest_ctc'] = max(nums['highest_ctc'])
    if nums['placed_count']:
        out['placed_count'] = int(sum(nums['placed_count']))
    if ratings:
        out['placement_rating'] = round(sum(ratings)/len(ratings), 2)
    return out

# helper to aggregate top recruiters and job profiles & program count from placement rows
def _extract_placement_lists(rows):
    """
    Returns:
      {
        'num_programs': int,
        'programs': [prog1, prog2, ...],
        'top_recruiters': [r1, r2, ...],
        'job_profiles': [p1, p2, ...]
      }

    NOTE: filters out obvious batch tokens like 'Batch 2019' from recruiters.
    """
    if not rows:
        return {
            'num_programs': 0,
            'programs': [],
            'top_recruiters': [],
            'job_profiles': []
        }
    prog_set = set()
    recruiters = set()
    profiles = set()
    batch_re = re.compile(r'(?i)batch[\s_-]*\d{2,4}')  # detect "Batch 2019" etc
    for r in rows:
        # program identification - try common column names
        prog = r.get('Program') or r.get('program') or r.get('Program Name') or r.get('program_name') or r.get('Program_Name') or r.get('course')
        if prog:
            # if multiple programs listed, split by separators
            for p in re.split(r'[;,|/]', str(prog)):
                p = p.strip()
                if p:
                    prog_set.add(p)
        # recruiters column
        for col in ['top_recruiter','top_recruiters','TopRecruiters','Top_Recruiters','top_recruiter_name','recruiter','recruiters']:
            if r.get(col):
                for part in re.split(r'[;,|/]', str(r.get(col))):
                    p = part.strip()
                    if p and not batch_re.search(p):
                        recruiters.add(p)
        # job titles / profiles column
        for col in ['job_titles','job_title','JobTitles','Job_Title','job_profiles','job_profile','job']:
            if r.get(col):
                for part in re.split(r'[;,|/]', str(r.get(col))):
                    p = part.strip()
                    if p:
                        profiles.add(p)
    progs = sorted([x for x in prog_set if x])
    recs = sorted([x for x in recruiters if x])
    profs = sorted([x for x in profiles if x])
    return {
        'num_programs': len(progs),
        'programs': progs,
        'top_recruiters': recs,
        'job_profiles': profs
    }

# helper to aggregate review numeric scores from REVIEWS_RAW_BY
def _aggregate_review_scores(raw_rows):
    """
    Look for these columns (case-insensitive variants):
      sentiment_score,mess_score,professor_score,campus_score,placement_score,infrastructure_score,overall_aspect_score
    Return dictionary with averaged floats (rounded to 2 decimals) when available.
    """
    keys = [
        'sentiment_score','mess_score','professor_score','campus_score',
        'placement_score','infrastructure_score','overall_aspect_score'
    ]
    # possible column name variants map
    variants = {}
    for k in keys:
        variants[k] = [k, k.upper(), k.replace('_',' '), k.replace('_','').lower(), k.title(), k.capitalize()]

    sums = {k: [] for k in keys}
    for r in raw_rows:
        for k in keys:
            found = False
            for col in variants[k]:
                if r.get(col) not in (None, '', []):
                    val = r.get(col)
                    try:
                        # remove non-numeric chars
                        num = float(re.sub(r'[^\d\.\-]', '', str(val)))
                        sums[k].append(num)
                        found = True
                        break
                    except Exception:
                        # not parseable; skip but mark found to avoid extra scanning
                        found = True
                        break
            if not found:
                # also try any column that contains the key as substring
                for colname, val in r.items():
                    if colname and k in colname.lower() and val not in (None, '', []):
                        try:
                            num = float(re.sub(r'[^\d\.\-]', '', str(val)))
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
            out[k] = ''
    return out

# route registration helper
def register_explore(app):
    from flask import jsonify, request

    @app.route('/explore/api/colleges')
    def _api_colleges():
        # return alphabetically sorted list of institute names
        names = []
        for r in _COLLEGES:
            name = r.get('Institute') or r.get('College') or r.get('institute_name') or r.get('Name') or r.get('name')
            if name:
                names.append(name.strip())
        names = sorted(list(set(names)))
        return jsonify({'colleges': names})

    @app.route('/explore/api/college')
    def _api_college():
        q = request.args.get('name', '').strip()
        if not q:
            return jsonify({'error': 'name required'}), 400
        key = _key(q)
        row = COLLEGE_MAP.get(key)
        if not row:
            # attempt fuzzy: match where name contains query
            for k,v in COLLEGE_MAP.items():
                if q.lower() in k:
                    row = v
                    break
        if not row:
            return jsonify({'error': 'institute not found'}), 404

        # build response with normalized fields
        inst = {}
        inst['institute_name'] = row.get('Institute') or row.get('College') or row.get('institute_name') or row.get('Name') or ''
        inst['district'] = row.get('District') or row.get('district') or ''
        inst['website'] = row.get('Website') or row.get('website') or row.get('site') or ''
        # images
        inst['logo_image'] = row.get('logo_image') or row.get('Logo') or row.get('Logo_URL') or row.get('logo_url') or ''
        inst['picture'] = row.get('Picture') or row.get('picture') or row.get('image') or row.get('Photo') or ''
        # keep placeholder programs (we will override using placement.csv)
        inst['programs'] = []
        # scores from college.csv (legacy) preserved but we'll override from reviews aggregation
        inst['rank'] = row.get('rank') or row.get('Rank') or ''

        # coordinates parsing
        lat, lon = None, None
        for c in ['Latitude','latitude','lat','Lat','Location','location','Coordinates','coordinates']:
            if row.get(c):
                lat, lon = _parse_latlon(row.get(c))
                if lat is not None:
                    break
        # also try columns named 'latitude' and 'longitude'
        if (lat is None or lon is None) and (row.get('latitude') and row.get('longitude')):
            try:
                lat = float(row.get('latitude')); lon = float(row.get('longitude'))
            except Exception:
                pass
        inst['latitude'] = lat
        inst['longitude'] = lon

        # placement-derived aggregates (num programs, recruiters, job profiles, program names)
        placement_rows = PLACEMENT_BY.get(key, [])
        placement_info = _extract_placement_lists(placement_rows)
        inst['num_programs'] = placement_info['num_programs']
        inst['programs'] = placement_info['programs']
        inst['top_recruiters'] = placement_info['top_recruiters']
        inst['key_profiles'] = placement_info['job_profiles']

        # aggregated placement numbers (avg/median/high/placed_count/placement_rating)
        inst['placement_summary'] = _aggregate_placement(placement_rows)

        # aggregated review scores (from reviews.csv)
        raw_rev_rows = REVIEWS_RAW_BY.get(key, [])
        review_scores = _aggregate_review_scores(raw_rev_rows)
        # attach these fields explicitly
        inst.update(review_scores)

        # also return a small convenience list for quick frontend display
        inst['sample_review_count'] = len(REVIEWS_BY.get(key, []))

        return jsonify(inst)

    @app.route('/explore/api/reviews')
    def _api_reviews():
        from flask import request
        q = request.args.get('name', '').strip()
        if not q:
            return jsonify({'reviews': []})
        key = _key(q)
        rows = REVIEWS_BY.get(key, [])
        return jsonify({'reviews': rows})

    @app.route('/explore/api/placement')
    def _api_placement():
        from flask import request
        q = request.args.get('name', '').strip()
        if not q:
            return jsonify({})
        key = _key(q)
        rows = PLACEMENT_BY.get(key, [])
        agg = _aggregate_placement(rows)
        # also include recruiters and job profiles and program count + program names
        lists = _extract_placement_lists(rows)
        agg.update({
            'num_programs': lists['num_programs'],
            'programs': lists['programs'],
            'top_recruiters': lists['top_recruiters'],
            'job_profiles': lists['job_profiles']
        })
        return jsonify(agg)

# auto-register if possible
try:
    # if the main app module created an 'app' Flask object and has been imported,
    # register endpoints automatically
    maybe_app = sys.modules.get('app')
    if maybe_app and hasattr(maybe_app, 'app'):
        register_explore(maybe_app.app)
except Exception:
    pass

# export for explicit registration
__all__ = ['register_explore']
