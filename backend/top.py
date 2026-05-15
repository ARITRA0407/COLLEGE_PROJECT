import os
import json
from flask import Flask, jsonify, render_template, send_from_directory
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_DIR = os.path.join(BASE_DIR, "csv")
PLACEMENT_CSV = os.path.join(CSV_DIR, "placement.csv")
COLLEGE_CSV = os.path.join(CSV_DIR, "college.csv")


def load_top10():
    try:
        placement = pd.read_csv(PLACEMENT_CSV)
    except Exception as e:
        raise RuntimeError(f"Failed to read placement CSV at {PLACEMENT_CSV}: {e}")
    try:
        college = pd.read_csv(COLLEGE_CSV)
    except Exception as e:
        raise RuntimeError(f"Failed to read college CSV at {COLLEGE_CSV}: {e}")
    placement.columns = [c.strip() for c in placement.columns]
    college.columns = [c.strip() for c in college.columns]
    if "inst_rank" not in placement.columns:
        found = [c for c in placement.columns if "rank" in c.lower()]
        if found:
            placement = placement.rename(columns={found[0]: "inst_rank"})
        else:
            raise RuntimeError("placement.csv doesn't contain 'inst_rank' column")
    placement["inst_rank"] = pd.to_numeric(placement["inst_rank"], errors="coerce")
    join_col = None
    if "Institute" in placement.columns and "Institute" in college.columns:
        join_col = "Institute"
    elif "inst_key" in placement.columns and "inst_key" in college.columns:
        join_col = "inst_key"
    else:
        common = set(placement.columns).intersection(set(college.columns))
        if len(common) > 0:
            join_col = list(common)[0]
        else:
            raise RuntimeError(
                "Could not find a common join column between placement.csv and college.csv"
            )
    merged = pd.merge(placement, college, on=join_col, how="left", suffixes=("", "_c"))
    merged = merged[merged["inst_rank"].notna()]
    merged = merged.sort_values(by="inst_rank", ascending=True)
    if "Institute" in merged.columns:
        merged = merged.drop_duplicates(subset=["Institute"])
    top = merged.head(10).copy()
    out = []
    for _, row in top.iterrows():
        website_val = ""
        if "Website" in row and pd.notna(row.get("Website")):
            website_val = str(row.get("Website")).strip()
        else:
            for c in ["website", "Website_url", "URL", "Site"]:
                if c in row and pd.notna(row.get(c)):
                    website_val = str(row.get(c)).strip()
                    break
        picture_val = ""
        if "Picture" in row and pd.notna(row.get("Picture")):
            picture_val = str(row.get("Picture")).strip()
        else:
            for c in ["picture", "Image", "image", "logo", "photo"]:
                if c in row and pd.notna(row.get(c)):
                    picture_val = str(row.get(c)).strip()
                    break
        inst_name = ""
        if "Institute" in row and pd.notna(row.get("Institute")):
            inst_name = str(row.get("Institute")).strip()
        else:
            inst_name = str(row.get(join_col, "")).strip()
        item = {
            "rank": int(row["inst_rank"]) if pd.notna(row["inst_rank"]) else None,
            "Institute": inst_name,
            "Website": website_val,
            "Picture": picture_val,
            "District": row.get("District", "") if "District" in row else "",
        }
        out.append(item)
    return out


def create_app(test_config=None):
    app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
    app.config["JSON_SORT_KEYS"] = False

    @app.route("/top")
    def top_page():
        try:
            return render_template("partials/top.html")
        except Exception as e:
            return f"Template render error: {e}", 500

    @app.route("/top/data")
    def top_data():
        try:
            data = load_top10()
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=True)
