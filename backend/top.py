# backend/top.py
"""
Flask endpoints to serve the Top 10 colleges scroller.

Usage:
- Place this file as project-root/backend/top.py
- Ensure your Flask app's templates folder includes templates/partials/top.html (the file above).
- You can run this module directly for testing: `python backend/top.py`
  which will run a small Flask app on 127.0.0.1:5001 serving:
    - GET /top        -> renders the top.html template
    - GET /top/data   -> returns JSON list of top-10 colleges (fields: rank, Institute, Website, Picture, District)
"""

import os
import json
from flask import Flask, jsonify, render_template, send_from_directory
import pandas as pd

# Update these paths if your CSVs are stored elsewhere relative to this file.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CSV_DIR = os.path.join(BASE_DIR, 'csv')

PLACEMENT_CSV = os.path.join(CSV_DIR, 'placement.csv')
COLLEGE_CSV = os.path.join(CSV_DIR, 'college.csv')

def load_top10():
    """
    Loads placement.csv and college.csv and returns top 10 entries by inst_rank.
    Returns a list of dicts with keys: rank, Institute, Website, Picture, District
    """
    # Read CSVs
    # Use try/except to provide reasonable fallback if files missing
    try:
        placement = pd.read_csv(PLACEMENT_CSV)
    except Exception as e:
        raise RuntimeError(f"Failed to read placement CSV at {PLACEMENT_CSV}: {e}")

    try:
        college = pd.read_csv(COLLEGE_CSV)
    except Exception as e:
        raise RuntimeError(f"Failed to read college CSV at {COLLEGE_CSV}: {e}")

    # Normalize column names (in case of stray whitespace)
    placement.columns = [c.strip() for c in placement.columns]
    college.columns = [c.strip() for c in college.columns]

    # Ensure inst_rank exists and can be numeric
    if 'inst_rank' not in placement.columns:
        # fallback: if rank column named differently, try lowercase match
        found = [c for c in placement.columns if 'rank' in c.lower()]
        if found:
            placement = placement.rename(columns={found[0]: 'inst_rank'})
        else:
            raise RuntimeError("placement.csv doesn't contain 'inst_rank' column")

    # Convert inst_rank to numeric for sorting. Coerce errors to NaN and drop them.
    placement['inst_rank'] = pd.to_numeric(placement['inst_rank'], errors='coerce')

    # Join using 'Institute' column if present; else try inst_key.
    join_col = None
    if 'Institute' in placement.columns and 'Institute' in college.columns:
        join_col = 'Institute'
    elif 'inst_key' in placement.columns and 'inst_key' in college.columns:
        join_col = 'inst_key'
    else:
        # fallback: try to find a likely shared column
        common = set(placement.columns).intersection(set(college.columns))
        if len(common) > 0:
            join_col = list(common)[0]
        else:
            raise RuntimeError("Could not find a common join column between placement.csv and college.csv")

    # Merge dataframes
    merged = pd.merge(placement, college, on=join_col, how='left', suffixes=('', '_c'))

    # Pick rows with numeric inst_rank and sort ascending (rank 1 is top)
    merged = merged[merged['inst_rank'].notna()]
    merged = merged.sort_values(by='inst_rank', ascending=True)

    # Take top 10 unique institutes (in case duplicates across years exist)
    # Keep the first occurrence per institute name
    if 'Institute' in merged.columns:
        merged = merged.drop_duplicates(subset=['Institute'])
    top = merged.head(10).copy()

    # Prepare output fields
    out = []
    for _, row in top.iterrows():
        # Extract Website and Picture columns reliably - prefer exactly-named columns
        website_val = ''
        if 'Website' in row and pd.notna(row.get('Website')):
            website_val = str(row.get('Website')).strip()
        else:
            # fallback: try lowercase or other variants
            for c in ['website', 'Website_url', 'URL', 'Site']:
                if c in row and pd.notna(row.get(c)):
                    website_val = str(row.get(c)).strip()
                    break

        picture_val = ''
        if 'Picture' in row and pd.notna(row.get('Picture')):
            picture_val = str(row.get('Picture')).strip()
        else:
            # try other common names
            for c in ['picture', 'Image', 'image', 'logo', 'photo']:
                if c in row and pd.notna(row.get(c)):
                    picture_val = str(row.get(c)).strip()
                    break

        inst_name = ''
        if 'Institute' in row and pd.notna(row.get('Institute')):
            inst_name = str(row.get('Institute')).strip()
        else:
            inst_name = str(row.get(join_col, '')).strip()

        item = {
            'rank': int(row['inst_rank']) if pd.notna(row['inst_rank']) else None,
            'Institute': inst_name,
            'Website': website_val,
            'Picture': picture_val,
            'District': row.get('District', '') if 'District' in row else '',
        }
        out.append(item)

    return out


def create_app(test_config=None):
    app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
    # If your project uses a different static folder, adjust as necessary.
    app.config['JSON_SORT_KEYS'] = False

    @app.route('/top')
    def top_page():
        # Render the partial template. If you prefer full page, you may
        # wrap this partial in your site layout; here we render the partial as a standalone page for preview.
        try:
            return render_template('partials/top.html')
        except Exception as e:
            return f"Template render error: {e}", 500

    @app.route('/top/data')
    def top_data():
        try:
            data = load_top10()
            return jsonify(data)
        except Exception as e:
            # Return an error JSON to help debugging
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == '__main__':
    # Run a small dev server for quick preview
    app = create_app()
    app.run(host='127.0.0.1', port=5001, debug=True)
