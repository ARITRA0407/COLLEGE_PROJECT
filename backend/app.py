# app.py
import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import safe_join
import pandas as pd
import random


# ensure backend module path is available for imports
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current file (app.py).
PROJECT_ROOT = os.path.dirname(BACKEND_DIR) # Get the parent directory, which is the project root.
sys.path.append(BACKEND_DIR) # Add the backend directory to the Python path for importing modules.
# ================= QUIZ SYSTEM INTEGRATION ================= #
# template/static configuration
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates') # Define the path to the templates folder.
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static') if os.path.exists(os.path.join(PROJECT_ROOT, 'static')) else None # Define the path to the static folder, checking if it exists.
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR) # Initialize the Flask application with template and static paths.
# Correct CSV paths
# ================= QUIZ CSV LOAD ================= #

CSV_FOLDER = os.path.join(PROJECT_ROOT, 'csv')

try:
    q_df = pd.read_csv(os.path.join(CSV_FOLDER, "QUESTIONS.csv"), encoding="latin1")
    career_df = pd.read_csv(os.path.join(CSV_FOLDER, "CAREER.csv"), encoding="utf-8-sig")

    # CLEANING
    q_df.columns = q_df.columns.str.strip()
    career_df.columns = career_df.columns.str.strip()
    career_df.columns = [col.strip().replace("\ufeff", "") for col in career_df.columns]

    # Fix Job column
    job_col = None
    for col in career_df.columns:
        if col.lower() == "job":
            job_col = col
            break

    if job_col:
        career_df.rename(columns={job_col: "Job"}, inplace=True)
        career_df["Job"] = career_df["Job"].astype(str).fillna("Unknown Job")

    # Difficulty weight map
    d_map = {"Easy": 0.2, "Medium": 0.5, "Hard": 0.8}

    user_data = {}

    print("✅ Quiz CSV loaded successfully from:", CSV_FOLDER)

except Exception as e:
    print("❌ Quiz CSV load error:", e)
    q_df, career_df = None, None



# =========================
# 🔥 HELPER FUNCTION (FIXED)
# =========================
def get_question(category, difficulty, asked):

    category = str(category).strip()

    df = q_df[
        (q_df["Category"].astype(str).str.strip() == category) &
        (~q_df["Question ID"].isin(asked))
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
        str(q.get("Option3", "") or "")
    ]

    weights = [
        float(q.get("Weight1", 0) or 0),
        float(q.get("Weight2", 0) or 0),
        float(q.get("Weight3", 0) or 0)
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


# =========================
# 🚀 START QUIZ (FIXED FOR ALL 17 CATEGORIES)
# =========================
@app.route("/start", methods=["POST"])
def start_quiz():

    if q_df is None:
        return jsonify([])

    q_df["Category"] = q_df["Category"].astype(str).str.strip()

    # ✅ FIX: ensures ALL 17 categories are always included
    categories = sorted(q_df["Category"].unique())

    user_data.clear()

    # 🔥 IMPORTANT: pre-initialize ALL categories safely
    user_data.update({
        "asked": [],
        "scores": {c: [] for c in categories},
        "time": {c: [] for c in categories},
        "last_diff": {c: "Medium" for c in categories},
        "state": {},
        "category_order": categories
    })

    questions = []

    for c in categories:
        q = get_question(c, "Medium", user_data["asked"])

        if q:
            user_data["asked"].append(q["Question ID"])
            questions.append(q)

    return jsonify(questions)


# =========================
# ⚡ NEXT QUESTION (FIXED + CATEGORY SAFE)
# =========================
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
        all_w = ans.get("all_weights", [w])

        time_sec = float(ans.get("time", 5))
        confidence = float(ans.get("confidence", 50)) / 100

        try:
            correct = w >= max(all_w) if all_w else False
        except:
            correct = False

        prev_diff = user_data["last_diff"].get(c, "Medium")

        result = process_answer(
            state=user_data["state"],
            category=c,
            score=max(0.0, min(1.0, w)),
            time_taken=time_sec,
            correct=correct,
            last_diff=prev_diff
        )

        user_data["state"] = result["updated_state"]
        next_diff = result["next_difficulty"]

        # 🔥 FIX: ensure category always exists
        user_data["scores"].setdefault(c, [])
        user_data["time"].setdefault(c, [])
        user_data["last_diff"].setdefault(c, "Medium")

        user_data["time"][c].append(time_sec)
        user_data["scores"][c].append(max(0.0, min(1.0, w * confidence)))

        q = get_question(c, next_diff, user_data["asked"])

        if q:
            user_data["asked"].append(q["Question ID"])
            user_data["last_diff"][c] = next_diff
            new_questions.append(q)

    # fallback safety
    if not new_questions:
        remaining = q_df[~q_df["Question ID"].isin(user_data["asked"])]

        if not remaining.empty:
            q = remaining.sample(1).iloc[0].to_dict()
            user_data["asked"].append(q["Question ID"])
            new_questions.append(q)

    return jsonify(new_questions)

# =========================
# 🧠 RESULT PAGE (RANK-BASED LEVEL LOGIC)
# =========================
@app.route("/quiz-result")
def quiz_result():

    if career_df is None:
        return "Career data not loaded", 500

    final = {}
    time_avg = {}

    categories = user_data.get("category_order", [])

    # ---------------- STEP 1: compute scores ----------------
    for c in categories:

        vals = user_data.get("scores", {}).get(c, [])
        times = user_data.get("time", {}).get(c, [])

        avg = sum(vals) / len(vals) if vals else 0
        tavg = sum(times) / len(times) if times else 0

        final[c] = {
            "score": round(avg, 2),
            "level": "Low"   # temporary placeholder
        }

        time_avg[c] = round(tavg, 2)

    # ---------------- STEP 2: PRIORITY ORDER ----------------
    priority = [
        "Logical_Reasoning","Math_Reasoning","Analytical_Reasoning",
        "Coding_Skill","Coding_Int","Data_Mining","AI_Int",
        "System_Opt","DB_Design","Web_Arch","Low_Level",
        "Cloud_Ops","Design_Int","User_Empathy","Risk_Eval",
        "Verbal_Reasoning","Crypto_Focus"
    ]

    # ---------------- STEP 3: SORT ----------------
    sorted_items = sorted(
        final.items(),
        key=lambda x: (
            -x[1]["score"],  # primary: high score first
            priority.index(x[0]) if x[0] in priority else 999  # tie-break
        )
    )

    # ---------------- STEP 4: ASSIGN LEVELS BY RANK ----------------
    n = len(sorted_items)

    top_6 = set([c for c, _ in sorted_items[:6]])
    mid_5 = set([c for c, _ in sorted_items[6:11]])
    low_6 = set([c for c, _ in sorted_items[11:]])

    for c in final.keys():

        if c in top_6:
            final[c]["level"] = "High"
        elif c in mid_5:
            final[c]["level"] = "Medium"
        else:
            final[c]["level"] = "Low"

    # ---------------- STEP 5: FINAL SORT FOR UI ----------------
    final = dict(sorted(
        final.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    ))

    # ---------------- CAREER MATCH ----------------
    def match(user_levels, row):

        score = 0
        total = 0
        lvl = {"low": 1, "medium": 2, "high": 3}

        for col in career_df.columns:
            if col == "Job":
                continue

            total += 1

            u = user_levels.get(col, {"level": "Low"})["level"].lower()
            r = str(row[col]).lower()

            if lvl.get(u, 1) >= lvl.get(r, 1):
                score += 1

        return (score / total) * 100 if total else 0

    df = career_df.copy()
    df["score"] = df.apply(lambda x: match(final, x), axis=1)

    df = df[df["score"] > 20].sort_values("score", ascending=False)

    rec = df.head(5).to_dict(orient="records")

    return render_template(
        "partials/result.html",
        result=final,
        rec=rec,
        time_avg=time_avg
    )


# import recommender
try:
    from recommendation import CollegeRecommender # Attempt to import the CollegeRecommender class.
except Exception as e:
    CollegeRecommender = None # Set the class to None if the import fails.
    print("Error importing recommendation module:", e) # Print the error for debugging.

# initialize recommender (safe)
recommender = None # Initialize recommender instance to None.
if CollegeRecommender is not None: # Check if the class was successfully imported.
    try:
        recommender = CollegeRecommender(data_root_dir=PROJECT_ROOT) # Initialize the recommender, passing the project root directory.
        print("✅ Recommender initialized.") # Print success message.
    except Exception as e:
        print("❌ Error initializing recommender:", e) # Print initialization error.
        recommender = None # Reset recommender to None on failure.
else:
    print("❌ CollegeRecommender class not available (import failed).") # Print message if the class wasn't available.



# --- NEW: import explore module so it can register its routes ---
# place import here (after app exists). explore.py is written to auto-register
# routes when it finds `app` in sys.modules as `app`.
try:
    import explore  # backend/explore.py — registers /explore/api/... endpoints
except Exception as _e:
    # avoid raising during import; surface a console message for debugging
    print("Could not import explore module (routes may not be registered):", _e)

# Explicitly register explore routes (works even when app.py runs as __main__)
try:
    from explore import register_explore
    register_explore(app)
except Exception as _e:
    print("Warning: could not register explore routes:", _e)
# ----------------------------------------------------------------------

# --- NEW: import top module so we can provide an API endpoint and page ---
# This is non-destructive: if top.py is missing or fails, we still run the app.
try:
    import top as top_module  # backend/top.py - provides load_top10() helper
    print("Imported top module (for top-ranked colleges).")
except Exception as _e:
    top_module = None
    print("Could not import top module (top endpoints may not be available):", _e)

# If top_module exists, register simple routes that use it.
if top_module is not None:
    try:
        @app.route('/top')  # lightweight preview page for the top scroller (alternate to /top-ranked)
        def top_preview():
            try:
                return render_template('partials/top.html')
            except Exception as e:
                return f"<h2>Template error</h2><pre>{e}</pre>", 500

        @app.route('/top/data')
        def top_data():
            """
            Returns JSON list of top-10 colleges. Delegates to top.load_top10() when available.
            """
            try:
                if hasattr(top_module, 'load_top10'):
                    data = top_module.load_top10()
                    return jsonify(data)
                else:
                    return jsonify({'error': 'top module missing load_top10 function'}), 500
            except Exception as e:
                print("Error in /top/data:", e)
                return jsonify({'error': str(e)}), 500

    except Exception as _e:
        print("Warning: could not register /top endpoints:", _e)
# ----------------------------------------------------------------------

# --- NEW: import ai module so it can register AI routes (if present) ---
# Non-destructive: if ai.py is missing or errors, the app still runs.
try:
    import ai as ai_module  # backend/ai.py — should register endpoints like /ai and /ai/chat
    print("Imported ai module (AI endpoints available).")
    # Try to register routes explicitly if module exposes register_ai
    try:
        if hasattr(ai_module, 'register_ai'):
            ai_module.register_ai(app)
            print("Registered AI routes via ai.register_ai(app).")
    except Exception as _e_inner:
        print("AI module present but register_ai(app) failed:", _e_inner)
except Exception as _e:
    ai_module = None
    print("Could not import ai module (AI routes may not be available):", _e)

# Serve CSVs from project-root/csv at /csv/<filename>
CSV_FOLDER = os.path.join(PROJECT_ROOT, 'csv') # Define the path to the CSV data folder.

@app.route('/csv/<path:filename>')
def serve_csv(filename):
    # basic safety: only serve files that exist in the csv directory
    try:
        # safe_join ensures no path traversal
        requested = safe_join(CSV_FOLDER, filename) # Safely construct the full path to the requested file.
        if not requested or not os.path.exists(requested): # Check if the path is safe and the file exists.
            return "Not found", 404 # Return 404 if file is not found or path is unsafe.
        return send_from_directory(CSV_FOLDER, filename, conditional=True) # Serve the file from the CSV folder.
    except Exception as e:
        # avoid revealing internals in production, but helpful in dev
        print("Error serving csv:", e) # Log the error.
        abort(404) # Abort with a 404 response.

@app.route('/')
def index():
    """
    Render the main page. Expects PROJECT_ROOT/templates/index.html to exist.
    """
    try:
        return render_template('index.html') # Render the main index page.
    except Exception as e:
        # Helpful error if template missing
        return f"<h2>Template error</h2><pre>{e}</pre>", 500 # Return an error message if the template is missing.

# NEW: route to serve the full standalone comparison page used by the iframe.
@app.route('/comparison')
def comparison():
    """
    Serve the standalone comparison page so it can be embedded in an iframe.
    Renders 'partials/comparision.html'. If the template is missing,
    returns a friendly HTML error message so the iframe shows it.
    """
    try:
        return render_template('partials/comparision.html') # Render the comparison partial from the 'partials' subfolder.
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
        return error_html, 500 # Return a template error for the comparison page.

# ----------------------------------------------------------------------
# NEW NAVIGATION ROUTES TO RENDER SEPARATE PARTIAL PAGES
# These correspond to the links in partials/header.html
# ----------------------------------------------------------------------

@app.route('/explore-colleges')
def explore_colleges():
    """
    Renders the 'Explore College' page.
    The URL_FOR name is 'explore_colleges'
    Renders 'partials/explore.html'.
    """
    try:
        return render_template('partials/explore.html') # Render the 'explore.html' template from the 'partials' subfolder.
    except Exception as e:
        return f"<h2>Template Error</h2><p>Could not find 'partials/explore.html'.</p><pre>{e}</pre>", 500 # Return an error if the template is not found.

@app.route('/top-ranked')
def top_ranked_colleges():
    """
    Renders the 'Top Ranked Colleges' page.
    The URL_FOR name is 'top_ranked_colleges'
    Renders 'partials/top.html'.
    """
    try:
        return render_template('partials/top.html') # Render the 'top.html' template from the 'partials' subfolder.
    except Exception as e:
        return f"<h2>Template Error</h2><p>Could not find 'partials/top.html'.</p><pre>{e}</pre>", 500 # Return an error if the template is not found.

@app.route('/recommendation')
def recommendation_page():
    """
    Renders the 'Recommendation' page.
    The URL_FOR name is 'recommendation_page'
    Renders 'partials/recommendation.html'.
    """
    try:
        return render_template('partials/recommendation.html') # Render the 'recommendation.html' template from the 'partials' subfolder.
    except Exception as e:
        return f"<h2>Template Error</h2><p>Could not find 'partials/recommendation.html'.</p><pre>{e}</pre>", 500 # Return an error if the template is not found.

@app.route('/ai-guidance')
def ai_guidance():
    """
    Renders the 'AI Guidance' page.
    The URL_FOR name is 'ai_guidance'
    Renders 'partials/ai.html'.
    """
    try:
        return render_template('partials/ai.html') # Render the 'ai.html' template from the 'partials' subfolder.
    except Exception as e:
        return f"<h2>Template Error</h2><p>Could not find 'partials/ai.html'.</p><pre>{e}</pre>", 500 # Return an error if the template is not found.
# ----------------------------------------------------------------------

@app.route('/quiz')
def quiz_page():
    """
    Renders the Quiz page.
    """
    try:
        return render_template('partials/quiz.html')
    except Exception as e:
        return f"<h2>Template Error</h2><p>Could not find 'partials/quiz.html'.</p><pre>{e}</pre>", 500

@app.route('/metadata', methods=['GET'])
def metadata():
    """
    Returns dropdown metadata for the frontend:
      programs, streams, quotas, categories, locations, sort options
    """
    if recommender is None:
        return jsonify({'error': 'Recommender not available'}), 503 # Return error if recommender is not initialized.

    try:
        # gather lists (they are lowercased in recommender; frontend can display them)
        programs = sorted(recommender.master_rank_df['Program'].dropna().unique().tolist()) if not recommender.master_rank_df.empty else [] # Extract and sort unique program names.
        streams = sorted(recommender.master_rank_df['Stream'].dropna().unique().tolist()) if not recommender.master_rank_df.empty else [] # Extract and sort unique stream names.
        quotas = sorted(recommender.master_rank_df['Quota'].dropna().unique().tolist()) if not recommender.master_rank_df.empty else [] # Extract and sort unique quota names.
        categories = sorted(recommender.master_rank_df['Category'].dropna().unique().tolist()) if not recommender.master_rank_df.empty else [] # Extract and sort unique category names.
        locations = sorted(recommender.merged_df['District'].dropna().unique().tolist()) if not recommender.merged_df.empty else [] # Extract and sort unique location names.

        sort_options = [
            {'value': 'Predicted Closing Rank', 'label': 'Predicted Closing Rank (asc)'}, # Sort option for rank.
            {'value': 'Max Average CTC', 'label': 'Max Average CTC (desc)'}, # Sort option for average CTC.
            {'value': 'placement_score', 'label': 'Placement Score (desc)'}, # Sort option for placement score.
            {'value': 'overall_aspect_score', 'label': 'Overall Score (desc)'}, # Sort option for overall score.
            {'value': 'professor_score', 'label': 'Professor Score (desc)'}, # Sort option for professor score.
            {'value': 'mess_score', 'label': 'Mess Score (desc)'}, # Sort option for mess score.
        ]

        return jsonify({
            'programs': programs, # Return list of programs.
            'streams': streams, # Return list of streams.
            'quotas': quotas, # Return list of quotas.
            'categories': categories, # Return list of categories.
            'locations': locations, # Return list of locations.
            'sort_options': sort_options # Return list of sort options.
        })
    except Exception as e:
        print("Error in /metadata:", e) # Log the error.
        return jsonify({'error': str(e)}), 500 # Return a JSON error response.

@app.route('/recommend_colleges', methods=['POST'])
def recommend_colleges():
    """
    Accepts JSON:
      - rank (required)
      - program (required)
      - stream, quota, category, location (optional)
      - min_ctc (optional numeric)
      - min_placements_score (optional numeric)
      - target_year (optional int)
      - top_n (optional int)
    Returns the dictionary result from recommender.recommend()
    """
    if recommender is None:
        return jsonify({'status': 'error', 'message': 'Recommender not available.'}), 503 # Return error if recommender is unavailable.

    try:
        data = request.get_json() or {} # Get JSON data from the request body.

        # accept both naming variants
        user_rank = data.get('rank') or data.get('user_rank') # Get user rank, accepting 'rank' or 'user_rank'.
        user_program = data.get('program') or data.get('user_program', '') # Get user program.
        user_stream = data.get('stream') or data.get('user_stream', '') # Get user stream.
        user_quota = data.get('quota') or data.get('user_quota', '') # Get user quota.
        user_category = data.get('category') or data.get('user_category', '') # Get user category.
        user_location = data.get('location') or data.get('user_location', '') # Get user location.

        # optional numeric filters
        try:
            min_ctc = float(data.get('min_ctc', 0) or 0) # Convert min_ctc to float, default is 0.
        except Exception:
            min_ctc = 0.0 # Set min_ctc to 0.0 on conversion error.
        try:
            min_placements_score = float(data.get('min_placements_score', 0) or 0) # Convert min_placements_score to float, default is 0.
        except Exception:
            min_placements_score = 0.0 # Set min_placements_score to 0.0 on conversion error.

        target_year = int(data.get('target_year', 2026)) # Convert target_year to int, default is 2026.
        top_n = int(data.get('top_n', 10)) # Convert top_n to int, default is 10.

        # validate required
        if user_rank is None or str(user_rank).strip() == '' or str(user_program).strip() == '': # Check if required fields (rank and program) are missing.
            return jsonify({'status': 'error', 'message': 'Required fields: rank and program.'}), 400 # Return 400 error for missing required fields.

        # call recommender
        result = recommender.recommend( # Call the recommend method with all user inputs.
            user_rank=user_rank,
            user_program=user_program,
            user_stream=user_stream,
            user_quota=user_quota,
            user_category=user_category,
            user_location=user_location,
            min_ctc=min_ctc,
            min_placements_score=min_placements_score,
            target_year=target_year
        )

        # trim results to top_n if present
        if isinstance(result, dict) and 'data' in result and isinstance(result['data'], list): # Check if the result is valid and contains a list of data.
            result['data'] = result['data'][:top_n] # Trims the list of recommendations to the top_n results.

        return jsonify(result) # Return the final recommendation result as JSON.

    except Exception as e:
        print("Error in recommendation API:", e) # Log the error.
        return jsonify({'status': 'error', 'message': str(e)}), 500 # Return a JSON error response.

# if __name__ == '__main__':
#     app.run(debug=True) # Run the Flask application in debug mode.

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
