import itertools
import json
import os
import pickle
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

try:
    from calibrated_weights import (
        get_saved_weight_vector,
        get_weight_vector,
    )
except Exception:

    def get_saved_weight_vector(*args, **kwargs):
        return None

    def get_weight_vector(weight_key, names=None, data_root_dir=None):
        defaults = {
            "college_quality_score": {
                "ctc": 0.45,
                "placement": 0.25,
                "overall": 0.30,
            },
            "college_recommendation_score": {
                "accessibility": 0.46,
                "quality": 0.28,
                "rule_boost": 0.26,
            },
        }
        weights = defaults.get(weight_key, {})
        if names is not None:
            names = list(names)
            weights = {name: float(weights.get(name, 0.0)) for name in names}
            total = sum(weights.values()) or 1.0
            return {name: value / total for name, value in weights.items()}
        return weights


try:
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeRegressor

    SKLEARN_AVAILABLE = True
except Exception:
    GradientBoostingRegressor = None
    RandomForestRegressor = None
    mean_absolute_error = None
    mean_absolute_percentage_error = None
    mean_squared_error = None
    median_absolute_error = None
    r2_score = None
    train_test_split = None
    LabelEncoder = None
    DecisionTreeRegressor = None
    SKLEARN_AVAILABLE = False


# College engine
class CollegeRecommender:

    CACHE_VERSION = 4
    ASSOCIATION_RULES_VERSION = 2
    RULES_FILENAME = "association_rules.csv"
    CACHE_FILENAME = "college_recommender_cache.pkl"
    METRICS_FILENAMES = {
        "association_rule_mining": "association_rule_mining_metrics.json",
        "decision_tree": "decision_tree_metrics.json",
        "random_forest": "random_forest_metrics.json",
        "gradient_boosting": "gradient_boosting_metrics.json",
        "hybrid_ensemble": "hybrid_ensemble_metrics.json",
    }
    STALE_METRICS_FILENAMES = (
        "heuristic_metrics.json",
        "extra_trees_metrics.json",
        "apriori_metrics.json",
    )
    MODEL_FEATURE_CATEGORICAL = ["Institute", "Program", "Stream", "Quota", "Category"]
    MODEL_FEATURE_NUMERIC = [
        "Predicted Closing Rank",
        "Opening Rank",
        "Max Average CTC",
        "placements_score_filter",
        "overall_aspect_score_filter",
        "mess_score",
        "professor_score",
        "campus_score",
        "infrastructure_score",
    ]

    def __init__(self, data_root_dir="."):
        self.data_root_dir = os.path.abspath(data_root_dir)
        self.results_dir = os.path.join(self.data_root_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.cache_path = os.path.join(self.results_dir, self.CACHE_FILENAME)
        self.rules_path = os.path.join(self.results_dir, self.RULES_FILENAME)
        self.dataframes = {}
        self.label_encoders = {}
        self.trained_models = {}
        self.model_metrics = {}
        self.association_rule_metrics = {}
        self.hybrid_model_weights = {}
        self.training_feature_columns = list(
            self.MODEL_FEATURE_CATEGORICAL + self.MODEL_FEATURE_NUMERIC
        )
        self.model_stack = {
            "association_rules": True,
            "hybrid_models": [],
            "hybrid_weights": {},
            "cache_loaded": False,
        }
        if self._load_prepared_cache():
            self.model_stack["cache_loaded"] = True
            self._persist_model_metrics()
            return
        self._load_all_data()
        self._prepare_master_rank_df()
        self._prepare_quality_data()
        self.dataframes = {}
        try:
            self._ensure_rules()
        except Exception as exc:
            print("Warning: association rules generation failed:", exc)
            self.assoc_rules_df = self._empty_rules_frame()
            self.association_rule_metrics = {
                "type": "association_rule_mining",
                "algorithm": "apriori_style_pair_rules",
                "rules_version": self.ASSOCIATION_RULES_VERSION,
                "reason": str(exc),
                "metrics": {},
                "top_rules": [],
            }
            self.model_metrics["association_rule_mining"] = (
                self.association_rule_metrics
            )
        try:
            self._prepare_ml_assets()
        except Exception as exc:
            print("Warning: hybrid ensemble preparation failed:", exc)
            self.label_encoders = {}
            self.trained_models = {}
            self.model_metrics = {}
            if self.association_rule_metrics:
                self.model_metrics["association_rule_mining"] = (
                    self.association_rule_metrics
                )
            self.hybrid_model_weights = {}
        try:
            self._persist_model_metrics()
        except Exception as exc:
            print("Warning: model metrics persistence failed:", exc)
        try:
            self._save_prepared_cache()
        except Exception as exc:
            print("Warning: recommender cache save failed:", exc)
        if getattr(self, "merged_df", pd.DataFrame()).empty:
            print("WARNING: master rank data is empty. Recommendations will fail.")

    def _get_file_path(self, file_name):
        return os.path.join(self.data_root_dir, "csv", file_name)

    def _legacy_rule_paths(self):
        return [
            os.path.join(self.data_root_dir, "csv", self.RULES_FILENAME),
            os.path.join(self.data_root_dir, "csv", "associates_rules.csv"),
        ]

    def _empty_rules_frame(self):
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "antecedent_support",
                "consequent_support",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        )

    def _load_saved_association_rule_metrics(self):
        metrics_path = os.path.join(
            self.results_dir,
            self.METRICS_FILENAMES["association_rule_mining"],
        )
        if not os.path.exists(metrics_path):
            return None
        try:
            with open(metrics_path, "r", encoding="utf-8") as file_obj:
                return json.load(file_obj)
        except Exception:
            return None

    def _load_rules_frame(self, path):
        if not os.path.exists(path):
            return None
        try:
            rules_df = pd.read_csv(
                path,
                dtype={
                    "antecedent": "string",
                    "consequent": "string",
                    "antecedent_support": "float32",
                    "consequent_support": "float32",
                    "support": "float32",
                    "confidence": "float32",
                    "lift": "float32",
                    "leverage": "float32",
                    "conviction": "float32",
                },
            )
        except Exception:
            return None
        expected_columns = set(self._empty_rules_frame().columns)
        if not expected_columns.issubset(set(rules_df.columns)):
            return None
        return rules_df.reindex(columns=list(self._empty_rules_frame().columns))

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if pd.isna(value):
            return None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return value

    def _cache_signature(self):
        source_files = [
            self._get_file_path("college.csv"),
            self._get_file_path("rank_2021.csv"),
            self._get_file_path("rank_2022.csv"),
            self._get_file_path("rank_2023.csv"),
            self._get_file_path("rank_2024.csv"),
            self._get_file_path("rank_2025.csv"),
            self._get_file_path("placement.csv"),
            self._get_file_path("reviews.csv"),
            __file__,
            os.path.join(os.path.dirname(__file__), "calibrated_weights.py"),
            os.path.join(self.results_dir, "calibrated_weights.json"),
        ]
        signature = {
            "cache_version": self.CACHE_VERSION,
            "sklearn_available": bool(SKLEARN_AVAILABLE),
        }
        for path in [self.rules_path] + self._legacy_rule_paths() + source_files:
            if os.path.exists(path):
                signature[path] = os.path.getmtime(path)
        return signature

    def _load_prepared_cache(self):
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "rb") as cache_file:
                payload = pickle.load(cache_file)
        except Exception:
            return False
        if payload.get("signature") != self._cache_signature():
            return False
        dataframe_attrs = [
            "master_rank_df",
            "merged_df",
            "full_college_df",
            "full_placement_df",
            "placement_max_ctc",
            "full_reviews_df",
            "reviews_avg_for_filter",
            "combined_quality_df",
            "assoc_rules_df",
        ]
        for attr in dataframe_attrs:
            setattr(self, attr, payload.get(attr, pd.DataFrame()))
        self.label_encoders = payload.get("label_encoders", {})
        self.trained_models = payload.get("trained_models", {})
        self.model_metrics = payload.get("model_metrics", {})
        self.association_rule_metrics = self._json_safe(
            payload.get("association_rule_metrics", {})
        )
        self.hybrid_model_weights = payload.get("hybrid_model_weights", {})
        self.training_feature_columns = payload.get(
            "training_feature_columns",
            list(self.MODEL_FEATURE_CATEGORICAL + self.MODEL_FEATURE_NUMERIC),
        )
        self.model_stack = payload.get("model_stack", self.model_stack)
        print("CollegeRecommender loaded from cache.")
        return True

    def _save_prepared_cache(self):
        payload = {
            "signature": self._cache_signature(),
            "master_rank_df": getattr(self, "master_rank_df", pd.DataFrame()),
            "merged_df": getattr(self, "merged_df", pd.DataFrame()),
            "full_college_df": getattr(self, "full_college_df", pd.DataFrame()),
            "full_placement_df": getattr(self, "full_placement_df", pd.DataFrame()),
            "placement_max_ctc": getattr(self, "placement_max_ctc", pd.DataFrame()),
            "full_reviews_df": getattr(self, "full_reviews_df", pd.DataFrame()),
            "reviews_avg_for_filter": getattr(
                self, "reviews_avg_for_filter", pd.DataFrame()
            ),
            "combined_quality_df": getattr(self, "combined_quality_df", pd.DataFrame()),
            "assoc_rules_df": getattr(self, "assoc_rules_df", pd.DataFrame()),
            "label_encoders": self.label_encoders,
            "trained_models": self.trained_models,
            "model_metrics": self.model_metrics,
            "association_rule_metrics": self.association_rule_metrics,
            "hybrid_model_weights": self.hybrid_model_weights,
            "training_feature_columns": self.training_feature_columns,
            "model_stack": self.model_stack,
        }
        with open(self.cache_path, "wb") as cache_file:
            pickle.dump(payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_all_data(self):
        file_names = [
            "college.csv",
            "rank_2021.csv",
            "rank_2022.csv",
            "rank_2023.csv",
            "rank_2024.csv",
            "rank_2025.csv",
            "placement.csv",
            "reviews.csv",
        ]
        for file_name in file_names:
            try:
                df_name = file_name.replace(".csv", "")
                file_path = self._get_file_path(file_name)
                self.dataframes[df_name] = pd.read_csv(file_path, dtype="object")
            except FileNotFoundError:
                print(f"Warning: file {file_name} not found and skipped.")
            except Exception as exc:
                print(f"Error loading {file_name}: {exc}")

    def _prepare_master_rank_df(self):
        combined_rank_data = []
        required_columns = [
            "Year",
            "Round",
            "Institute",
            "Program",
            "Stream",
            "Seat Type",
            "Quota",
            "Category",
            "Opening Rank",
            "Closing Rank",
        ]
        for df_name, df in self.dataframes.items():
            if not df_name.startswith("rank_20"):
                continue
            tmp = df.copy()
            try:
                year = int(df_name.split("_")[1])
            except Exception:
                year = pd.NA
            tmp["Year"] = year
            if "Seat Type" not in tmp.columns:
                tmp["Seat Type"] = "N/A"
            if "Stream" in tmp.columns and not tmp["Stream"].empty:
                tmp["Stream"] = (
                    tmp["Stream"]
                    .fillna("")
                    .astype(str)
                    .str.replace(
                        r"B\.E/B\.Tech.*|B\.E/B\.Arch.*|B\.Tech.*",
                        "b.e/b. tech",
                        regex=True,
                    )
                )
            for col in required_columns:
                if col not in tmp.columns:
                    tmp[col] = pd.NA
            combined_rank_data.append(tmp[required_columns])
        if not combined_rank_data:
            self.merged_df = pd.DataFrame(columns=required_columns + ["District"])
            self.master_rank_df = pd.DataFrame()
            return
        master_df = pd.concat(combined_rank_data, ignore_index=True)
        for col in ["Opening Rank", "Closing Rank", "Round", "Year"]:
            if col in master_df.columns:
                master_df[col] = pd.to_numeric(master_df[col], errors="coerce")
        for col in ["Institute", "Program", "Stream", "Quota", "Category", "Seat Type"]:
            if col in master_df.columns:
                master_df[col] = (
                    master_df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace("nan", "")
                    .fillna("")
                )
            else:
                master_df[col] = ""
        if "college" in self.dataframes:
            college_df = self.dataframes["college"][["Institute", "District"]].copy()
            college_df["Institute"] = (
                college_df["Institute"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace("nan", "")
                .fillna("")
            )
            college_df["District"] = (
                college_df["District"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace("nan", "")
                .fillna("")
            )
            self.merged_df = pd.merge(master_df, college_df, on="Institute", how="left")
        else:
            self.merged_df = master_df.copy()
            self.merged_df["District"] = ""
        if "District" in self.merged_df.columns:
            self.merged_df["District"] = (
                self.merged_df["District"]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace("nan", "")
                .fillna("")
            )
        self.master_rank_df = master_df

    def _prepare_quality_data(self):
        self.full_college_df = self.dataframes.get("college", pd.DataFrame()).copy()
        self.full_college_df = self.full_college_df.rename(
            columns={"logo_image_url": "logo_image"}
        )
        if not self.full_college_df.empty:
            expected_cols = [
                "Institute",
                "District",
                "Location",
                "Website",
                "logo_image",
                "Picture",
            ]
            for col in expected_cols:
                if col not in self.full_college_df.columns:
                    self.full_college_df[col] = ""
            self.full_college_df = self.full_college_df.reindex(columns=expected_cols)
            for col in ["Institute", "District"]:
                self.full_college_df[col] = (
                    self.full_college_df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace("nan", "")
                    .fillna("")
                )
        placement_df = self.dataframes.get("placement", pd.DataFrame()).copy()
        self.full_placement_df = pd.DataFrame()
        self.placement_max_ctc = pd.DataFrame()
        if not placement_df.empty:
            if "Institute" in placement_df.columns:
                placement_df["Institute"] = (
                    placement_df["Institute"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace("nan", "")
                    .fillna("")
                )
            if "Program" in placement_df.columns:
                placement_df["Program"] = (
                    placement_df["Program"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace("nan", "")
                    .fillna("")
                )
            placement_df = placement_df.rename(
                columns={
                    "top_recruiters": "top recruiter",
                    "job_titles": "job_title",
                    "inst_rank": "institute_rank",
                }
            )
            for col in ["average_ctc", "median_ctc", "highest_ctc", "placement_rating"]:
                if col in placement_df.columns:
                    placement_df[col] = pd.to_numeric(
                        placement_df[col], errors="coerce"
                    )
            if "average_ctc" in placement_df.columns:
                self.placement_max_ctc = (
                    placement_df.groupby(["Institute", "Program"])["average_ctc"]
                    .max()
                    .reset_index()
                    .rename(columns={"average_ctc": "Max Average CTC"})
                )
            agg_map = {}
            for col in ["average_ctc", "median_ctc", "highest_ctc", "placement_rating"]:
                if col in placement_df.columns:
                    agg_map[col] = "max"
            for col in ["top recruiter", "job_title", "institute_rank"]:
                if col in placement_df.columns:
                    agg_map[col] = "first"
            if agg_map:
                self.full_placement_df = (
                    placement_df.groupby(["Institute", "Program"])
                    .agg(agg_map)
                    .reset_index()
                )
        reviews_df = self.dataframes.get("reviews", pd.DataFrame()).copy()
        self.full_reviews_df = pd.DataFrame()
        self.reviews_avg_for_filter = pd.DataFrame()
        if not reviews_df.empty:
            reviews_df = reviews_df.rename(columns={"college_name": "Institute"})
            if "Institute" in reviews_df.columns:
                reviews_df["Institute"] = (
                    reviews_df["Institute"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace("nan", "")
                    .fillna("")
                )
            review_cols = [
                "rating",
                "sentiment_score",
                "mess_score",
                "professor_score",
                "campus_score",
                "placements_score",
                "infrastructure_score",
                "overall_aspect_score",
            ]
            for col in review_cols:
                if col in reviews_df.columns:
                    reviews_df[col] = pd.to_numeric(reviews_df[col], errors="coerce")
            present_cols = [col for col in review_cols if col in reviews_df.columns]
            if present_cols:
                self.full_reviews_df = (
                    reviews_df.groupby("Institute")
                    .agg({col: "mean" for col in present_cols})
                    .reset_index()
                )
                if "placements_score" in self.full_reviews_df.columns:
                    self.full_reviews_df = self.full_reviews_df.rename(
                        columns={"placements_score": "placement_score"}
                    )
                filter_cols = [
                    col
                    for col in [
                        "mess_score",
                        "professor_score",
                        "campus_score",
                        "placements_score",
                        "infrastructure_score",
                        "overall_aspect_score",
                    ]
                    if col in reviews_df.columns
                ]
                if filter_cols:
                    tmp = (
                        reviews_df[["Institute"] + filter_cols]
                        .groupby("Institute")[filter_cols]
                        .mean()
                        .reset_index()
                    )
                    self.reviews_avg_for_filter = tmp.rename(
                        columns={
                            "placements_score": "placements_score_filter",
                            "overall_aspect_score": "overall_aspect_score_filter",
                        }
                    )
        if not self.placement_max_ctc.empty and not self.reviews_avg_for_filter.empty:
            self.combined_quality_df = pd.merge(
                self.placement_max_ctc,
                self.reviews_avg_for_filter,
                on="Institute",
                how="left",
            )
        elif not self.placement_max_ctc.empty:
            self.combined_quality_df = self.placement_max_ctc.copy()
        elif not self.reviews_avg_for_filter.empty:
            self.combined_quality_df = self.reviews_avg_for_filter.copy()
        else:
            self.combined_quality_df = pd.DataFrame()

    def get_unique_programs(self):
        return (
            sorted(self.master_rank_df["Program"].dropna().unique().tolist())
            if not self.master_rank_df.empty
            else []
        )

    def get_unique_streams(self):
        return (
            sorted(self.master_rank_df["Stream"].dropna().unique().tolist())
            if not self.master_rank_df.empty
            else []
        )

    def get_unique_quotas(self):
        return (
            sorted(self.master_rank_df["Quota"].dropna().unique().tolist())
            if not self.master_rank_df.empty
            else []
        )

    def get_unique_categories(self):
        return (
            sorted(self.master_rank_df["Category"].dropna().unique().tolist())
            if not self.master_rank_df.empty
            else []
        )

    def get_unique_locations(self):
        return (
            sorted(self.merged_df["District"].dropna().unique().tolist())
            if not self.merged_df.empty
            else []
        )

    def _clean_user_input(self, value):
        return value.strip().lower() if isinstance(value, str) else ""

    def _predict_top_colleges_rank_only(
        self,
        program,
        stream="",
        quota="",
        category="",
        district="",
        target_year=2026,
    ):
        df = self.merged_df.copy()
        if df.empty:
            return pd.DataFrame()
        program = self._clean_user_input(program)
        stream = self._clean_user_input(stream)
        quota = self._clean_user_input(quota)
        category = self._clean_user_input(category)
        district = self._clean_user_input(district)
        for col in ["Program", "Stream", "Quota", "Category", "District", "Institute"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace("nan", "")
                .fillna("")
            )
        if program and "tfw" in program and not category:
            category = "tuition fee waiver"

        def apply_filters(df_in, prog, strm, qta, cat, dist):
            tmp = df_in.copy()
            if prog:
                exact = tmp[tmp["Program"] == prog].copy()
                tmp = (
                    exact
                    if not exact.empty
                    else tmp[tmp["Program"].str.contains(prog, na=False)].copy()
                )
            if strm and not tmp.empty:
                exact = tmp[tmp["Stream"] == strm].copy()
                if exact.empty:
                    exact = tmp[tmp["Stream"].str.contains(strm, na=False)].copy()
                if not exact.empty:
                    tmp = exact
            if qta and not tmp.empty:
                exact = tmp[tmp["Quota"] == qta].copy()
                if exact.empty:
                    exact = tmp[tmp["Quota"].str.contains(qta, na=False)].copy()
                if not exact.empty:
                    tmp = exact
            if dist and not tmp.empty:
                exact = tmp[tmp["District"] == dist].copy()
                if exact.empty:
                    exact = tmp[tmp["District"].str.contains(dist, na=False)].copy()
                if not exact.empty:
                    tmp = exact
            if cat and not tmp.empty:
                exact = tmp[tmp["Category"] == cat].copy()
                if exact.empty:
                    exact = tmp[tmp["Category"].str.contains(cat, na=False)].copy()
                if not exact.empty:
                    tmp = exact
            return tmp

        filtered_df = apply_filters(df, program, stream, quota, category, district)
        if filtered_df.empty:
            filtered_df = apply_filters(df, program, stream, quota, "", district)
        if filtered_df.empty:
            filtered_df = apply_filters(df, program, stream, "", "", district)
        if filtered_df.empty:
            filtered_df = apply_filters(df, program, stream, "", "", "")
        if filtered_df.empty:
            filtered_df = apply_filters(df, program, "", "", "", "")
        if filtered_df.empty and program:
            filtered_df = df[df["Program"].str.contains(program, na=False)].copy()
        if filtered_df.empty:
            return pd.DataFrame()
        final_ranks = filtered_df.sort_values("Round", ascending=False).drop_duplicates(
            subset=["Year", "Institute", "Stream", "Quota", "Category"],
            keep="first",
        )
        grouping_cols = ["Institute", "Program", "Stream", "Quota", "Category"]
        historical_years = (
            final_ranks["Year"].dropna().unique()
            if "Year" in final_ranks.columns
            else []
        )
        if target_year in historical_years:
            result_df = final_ranks[final_ranks["Year"] == target_year]
            if not result_df.empty:
                top_colleges = result_df.sort_values(by="Closing Rank", ascending=True)
                return top_colleges[
                    [
                        "Institute",
                        "Program",
                        "Stream",
                        "Seat Type",
                        "Quota",
                        "Category",
                        "Opening Rank",
                        "Closing Rank",
                        "District",
                    ]
                ].rename(columns={"Closing Rank": "Predicted Closing Rank"})
        prediction_results = []
        for name, group in final_ranks.groupby(grouping_cols):
            valid_ranks_group = group.dropna(subset=["Closing Rank"])
            if valid_ranks_group.empty:
                continue
            sorted_group = valid_ranks_group.sort_values(by="Year", ascending=False)
            recent = [
                float(x)
                for x in sorted_group["Closing Rank"].head(2).tolist()
                if pd.notna(x)
            ]
            if not recent:
                continue
            latest_data = group.sort_values(by="Year", ascending=False).iloc[0]
            prediction_results.append(
                {
                    "Institute": name[0],
                    "Program": name[1],
                    "Stream": name[2],
                    "Quota": name[3],
                    "Category": name[4],
                    "Predicted Closing Rank": max(1.0, float(np.mean(recent))),
                    "Opening Rank": (
                        latest_data["Opening Rank"]
                        if pd.notna(latest_data["Opening Rank"])
                        else np.nan
                    ),
                    "Seat Type": (
                        latest_data["Seat Type"]
                        if "Seat Type" in latest_data.index
                        else ""
                    ),
                    "District": (
                        latest_data["District"]
                        if "District" in latest_data.index
                        else ""
                    ),
                }
            )
        return pd.DataFrame(prediction_results)

    def _merge_quality_columns(self, df):
        if df.empty:
            return df.copy()
        if not getattr(self, "combined_quality_df", pd.DataFrame()).empty:
            merged = pd.merge(
                df, self.combined_quality_df, on=["Institute", "Program"], how="left"
            )
        else:
            merged = df.copy()
        score_cols = [
            "Max Average CTC",
            "mess_score",
            "professor_score",
            "campus_score",
            "placements_score_filter",
            "infrastructure_score",
            "overall_aspect_score_filter",
        ]
        for col in score_cols:
            if col not in merged.columns:
                merged[col] = 0
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        for col in ["Predicted Closing Rank", "Opening Rank"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
        return merged

    def _filter_top_colleges_by_metrics(self, df, min_ctc, min_placements_score=0):
        filtered_df = df.copy()
        if min_ctc > 0 and "Max Average CTC" in filtered_df.columns:
            filtered_df = filtered_df[
                pd.to_numeric(filtered_df["Max Average CTC"], errors="coerce").fillna(0)
                >= min_ctc
            ]
        if (
            min_placements_score > 0
            and "placements_score_filter" in filtered_df.columns
        ):
            filtered_df = filtered_df[
                pd.to_numeric(
                    filtered_df["placements_score_filter"], errors="coerce"
                ).fillna(0)
                >= min_placements_score
            ]
        return filtered_df

    def _normalize_series(self, series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        if numeric.empty:
            return pd.Series(dtype=float)
        min_val = float(numeric.min())
        max_val = float(numeric.max())
        if np.isclose(min_val, max_val):
            return pd.Series([1.0] * len(numeric), index=numeric.index, dtype=float)
        return (numeric - min_val) / (max_val - min_val)

    def _ensure_rules(self):
        loaded_rules = None
        saved_metrics = self._load_saved_association_rule_metrics()
        metrics_current = bool(saved_metrics) and (
            int(saved_metrics.get("rules_version", 0)) == self.ASSOCIATION_RULES_VERSION
        )
        if metrics_current:
            for path in [self.rules_path] + self._legacy_rule_paths():
                loaded_rules = self._load_rules_frame(path)
                if loaded_rules is not None:
                    break
        if loaded_rules is None:
            loaded_rules, assoc_metrics = self._generate_association_rules()
            self.association_rule_metrics = assoc_metrics
        else:
            self.association_rule_metrics = saved_metrics
        self.association_rule_metrics = self._json_safe(self.association_rule_metrics)
        self.assoc_rules_df = (
            loaded_rules if loaded_rules is not None else self._empty_rules_frame()
        )
        self.model_metrics["association_rule_mining"] = self.association_rule_metrics
        try:
            self.assoc_rules_df.to_csv(self.rules_path, index=False)
        except Exception as exc:
            print("Warning: failed to save association rules to results folder:", exc)

    def _generate_association_rules(
        self,
        min_support=0.03,
        max_itemset_size=2,
        min_confidence=0.18,
        max_rules_per_antecedent=6,
    ):
        df = getattr(self, "merged_df", pd.DataFrame())
        if df.empty:
            empty_metrics = {
                "type": "association_rule_mining",
                "algorithm": "apriori_style_pair_rules",
                "rules_version": self.ASSOCIATION_RULES_VERSION,
                "metrics": {
                    "transactions": 0,
                    "unique_items": 0,
                    "frequent_itemsets": 0,
                    "rules_generated": 0,
                    "rules_saved": 0,
                    "avg_support": 0.0,
                    "avg_confidence": 0.0,
                    "avg_lift": 0.0,
                    "max_support": 0.0,
                    "max_confidence": 0.0,
                    "max_lift": 0.0,
                },
                "configuration": {
                    "min_support": min_support,
                    "max_itemset_size": max_itemset_size,
                    "min_confidence": min_confidence,
                    "max_rules_per_antecedent": max_rules_per_antecedent,
                },
                "top_rules": [],
            }
            return self._empty_rules_frame(), empty_metrics
        attributes = ["Program", "Stream", "Quota", "Category", "District"]
        transactions = []
        unique_items = set()
        for _, row in df.iterrows():
            items = set()
            for attribute in attributes:
                value = str(row.get(attribute, "")).strip().lower()
                if value:
                    items.add(f"{attribute.lower()}={value}")
            if items:
                ordered_items = sorted(items)
                transactions.append(ordered_items)
                unique_items.update(ordered_items)
        total_transactions = len(transactions)
        if total_transactions == 0:
            empty_metrics = {
                "type": "association_rule_mining",
                "algorithm": "apriori_style_pair_rules",
                "rules_version": self.ASSOCIATION_RULES_VERSION,
                "metrics": {
                    "transactions": 0,
                    "unique_items": int(len(unique_items)),
                    "frequent_itemsets": 0,
                    "rules_generated": 0,
                    "rules_saved": 0,
                    "avg_support": 0.0,
                    "avg_confidence": 0.0,
                    "avg_lift": 0.0,
                    "max_support": 0.0,
                    "max_confidence": 0.0,
                    "max_lift": 0.0,
                },
                "configuration": {
                    "min_support": min_support,
                    "max_itemset_size": max_itemset_size,
                    "min_confidence": min_confidence,
                    "max_rules_per_antecedent": max_rules_per_antecedent,
                },
                "top_rules": [],
            }
            return self._empty_rules_frame(), empty_metrics
        itemset_counts = Counter()
        for transaction in transactions:
            for size in range(1, max_itemset_size + 1):
                for combo in itertools.combinations(transaction, size):
                    itemset_counts[frozenset(combo)] += 1
        frequent_itemsets = {
            itemset: count
            for itemset, count in itemset_counts.items()
            if (count / total_transactions) >= min_support
        }
        support_map = {
            itemset: count / total_transactions
            for itemset, count in frequent_itemsets.items()
        }
        rules = []
        for itemset in frequent_itemsets:
            if len(itemset) < 2:
                continue
            for consequent_item in itemset:
                antecedent = frozenset(itemset - {consequent_item})
                consequent = frozenset([consequent_item])
                if antecedent not in support_map or consequent not in support_map:
                    continue
                support_ab = support_map[itemset]
                support_a = support_map[antecedent]
                support_b = support_map[consequent]
                confidence = support_ab / support_a if support_a else 0.0
                if confidence < min_confidence:
                    continue
                lift = confidence / support_b if support_b else 0.0
                leverage = support_ab - (support_a * support_b)
                conviction = None
                if support_b < 1.0 and confidence < 1.0:
                    conviction = (1.0 - support_b) / max(1.0 - confidence, 1e-9)
                rules.append(
                    {
                        "antecedent": ";".join(sorted(list(antecedent))),
                        "consequent": ";".join(sorted(list(consequent))),
                        "antecedent_support": round(support_a, 6),
                        "consequent_support": round(support_b, 6),
                        "support": round(support_ab, 6),
                        "confidence": round(confidence, 6),
                        "lift": round(lift, 6),
                        "leverage": round(leverage, 6),
                        "conviction": (
                            round(float(conviction), 6)
                            if conviction is not None
                            else None
                        ),
                    }
                )
        rules_df = pd.DataFrame(rules)
        if rules_df.empty:
            empty_metrics = {
                "type": "association_rule_mining",
                "algorithm": "apriori_style_pair_rules",
                "rules_version": self.ASSOCIATION_RULES_VERSION,
                "metrics": {
                    "transactions": int(total_transactions),
                    "unique_items": int(len(unique_items)),
                    "frequent_itemsets": int(len(frequent_itemsets)),
                    "rules_generated": 0,
                    "rules_saved": 0,
                    "avg_support": 0.0,
                    "avg_confidence": 0.0,
                    "avg_lift": 0.0,
                    "max_support": 0.0,
                    "max_confidence": 0.0,
                    "max_lift": 0.0,
                },
                "configuration": {
                    "min_support": min_support,
                    "max_itemset_size": max_itemset_size,
                    "min_confidence": min_confidence,
                    "max_rules_per_antecedent": max_rules_per_antecedent,
                },
                "top_rules": [],
            }
            return self._empty_rules_frame(), empty_metrics
        rules_df = rules_df.sort_values(
            by=["confidence", "lift", "support"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        rules_generated = int(len(rules_df))
        rules_df["rule_rank"] = rules_df.groupby("antecedent").cumcount() + 1
        rules_df = rules_df[rules_df["rule_rank"] <= max_rules_per_antecedent].drop(
            columns=["rule_rank"]
        )
        rules_df = rules_df.reset_index(drop=True)
        for column in [
            "antecedent_support",
            "consequent_support",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ]:
            rules_df[column] = pd.to_numeric(rules_df[column], errors="coerce").astype(
                "float32"
            )
        top_rules = []
        for record in rules_df.head(10).to_dict("records"):
            clean_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    clean_record[key] = None
                elif isinstance(value, (np.floating, np.integer)):
                    clean_record[key] = float(value)
                else:
                    clean_record[key] = value
            top_rules.append(clean_record)
        metrics = {
            "type": "association_rule_mining",
            "algorithm": "apriori_style_pair_rules",
            "rules_version": self.ASSOCIATION_RULES_VERSION,
            "metrics": {
                "transactions": int(total_transactions),
                "unique_items": int(len(unique_items)),
                "frequent_itemsets": int(len(frequent_itemsets)),
                "rules_generated": rules_generated,
                "rules_saved": int(len(rules_df)),
                "avg_support": round(float(rules_df["support"].mean()), 6),
                "avg_confidence": round(float(rules_df["confidence"].mean()), 6),
                "avg_lift": round(float(rules_df["lift"].mean()), 6),
                "max_support": round(float(rules_df["support"].max()), 6),
                "max_confidence": round(float(rules_df["confidence"].max()), 6),
                "max_lift": round(float(rules_df["lift"].max()), 6),
            },
            "configuration": {
                "min_support": min_support,
                "max_itemset_size": max_itemset_size,
                "min_confidence": min_confidence,
                "max_rules_per_antecedent": max_rules_per_antecedent,
            },
            "top_rules": top_rules,
        }
        return rules_df, metrics

    def _compute_boosts_from_rules(self, candidates_df, user_filters):
        boosts = defaultdict(float)
        rules_df = getattr(self, "assoc_rules_df", pd.DataFrame())
        if rules_df is None or rules_df.empty:
            return boosts
        candidate_items = {}
        for _, row in candidates_df.iterrows():
            key = (row.get("Institute", ""), row.get("Program", ""))
            items = set()
            for attribute in ["Program", "Stream", "Quota", "Category", "District"]:
                value = str(row.get(attribute, "")).strip().lower()
                if value:
                    items.add(f"{attribute.lower()}={value}")
            candidate_items[key] = items
        user_items = set()
        for key, value in user_filters.items():
            cleaned = str(value).strip().lower()
            if cleaned:
                user_items.add(f"{key.lower()}={cleaned}")
        for _, rule in rules_df.iterrows():
            antecedent = set(filter(None, str(rule.get("antecedent", "")).split(";")))
            consequent = set(filter(None, str(rule.get("consequent", "")).split(";")))
            if not antecedent or not antecedent.issubset(user_items):
                continue
            try:
                boost_value = float(rule.get("confidence", 0.0)) * float(
                    rule.get("support", 0.0)
                )
            except Exception:
                boost_value = 0.0
            for candidate_key, item_set in candidate_items.items():
                if consequent.issubset(item_set):
                    boosts[candidate_key] += boost_value
        return boosts

    def _build_group_predictions_all(self):
        df = getattr(self, "master_rank_df", pd.DataFrame()).copy()
        if df.empty:
            return pd.DataFrame()
        grouping_cols = ["Institute", "Program", "Stream", "Quota", "Category"]
        df["Closing Rank"] = pd.to_numeric(df["Closing Rank"], errors="coerce")
        df["Opening Rank"] = pd.to_numeric(df["Opening Rank"], errors="coerce")
        prediction_rows = []
        for name, group in df.groupby(grouping_cols):
            valid_group = group.dropna(subset=["Closing Rank"])
            if valid_group.empty:
                continue
            recent = [
                float(x)
                for x in valid_group.sort_values("Year", ascending=False)[
                    "Closing Rank"
                ]
                .head(2)
                .tolist()
                if pd.notna(x)
            ]
            if not recent:
                continue
            latest_row = group.sort_values("Year", ascending=False).iloc[0]
            prediction_rows.append(
                {
                    "Institute": name[0],
                    "Program": name[1],
                    "Stream": name[2],
                    "Quota": name[3],
                    "Category": name[4],
                    "Predicted Closing Rank": max(1.0, float(np.mean(recent))),
                    "Latest Closing Rank": (
                        float(latest_row["Closing Rank"])
                        if pd.notna(latest_row["Closing Rank"])
                        else np.nan
                    ),
                    "Opening Rank": (
                        float(latest_row["Opening Rank"])
                        if pd.notna(latest_row["Opening Rank"])
                        else np.nan
                    ),
                }
            )
        return pd.DataFrame(prediction_rows)

    def _build_training_frame(self):
        grouped_df = self._build_group_predictions_all()
        if grouped_df.empty:
            return pd.DataFrame()
        grouped_df["Latest Closing Rank"] = pd.to_numeric(
            grouped_df["Latest Closing Rank"], errors="coerce"
        )
        grouped_df = grouped_df.dropna(subset=["Latest Closing Rank"]).copy()
        if grouped_df.empty:
            return pd.DataFrame()
        if not getattr(self, "combined_quality_df", pd.DataFrame()).empty:
            grouped_df = pd.merge(
                grouped_df,
                self.combined_quality_df,
                on=["Institute", "Program"],
                how="left",
            )
        for col in self.MODEL_FEATURE_NUMERIC:
            if col not in grouped_df.columns:
                grouped_df[col] = 0.0
            grouped_df[col] = pd.to_numeric(grouped_df[col], errors="coerce").fillna(
                0.0
            )
        for col in self.MODEL_FEATURE_CATEGORICAL:
            if col not in grouped_df.columns:
                grouped_df[col] = ""
            grouped_df[col] = (
                grouped_df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace("nan", "")
                .fillna("")
            )
        return grouped_df

    def _fit_or_transform_encoder(self, series, column_name, fit=False):
        clean_series = (
            series.astype(str).fillna("").str.strip().str.lower().replace("nan", "")
        )
        unknown_token = "__unknown__"
        if fit:
            encoder = LabelEncoder()
            values = clean_series.tolist()
            if unknown_token not in values:
                values.append(unknown_token)
            encoder.fit(values)
            self.label_encoders[column_name] = encoder
        else:
            encoder = self.label_encoders.get(column_name)
            if encoder is None:
                return pd.Series(
                    [0] * len(clean_series), index=clean_series.index, dtype=int
                )
        known_values = set(self.label_encoders[column_name].classes_)
        clean_series = clean_series.where(
            clean_series.isin(known_values), unknown_token
        )
        return pd.Series(
            self.label_encoders[column_name].transform(clean_series),
            index=clean_series.index,
            dtype=int,
        )

    def _encode_feature_frame(self, df, fit=False):
        encoded = pd.DataFrame(index=df.index)
        for col in self.MODEL_FEATURE_CATEGORICAL:
            source = (
                df[col]
                if col in df.columns
                else pd.Series([""] * len(df), index=df.index)
            )
            encoded[col] = self._fit_or_transform_encoder(source, col, fit=fit).astype(
                "int32"
            )
        for col in self.MODEL_FEATURE_NUMERIC:
            source = (
                df[col]
                if col in df.columns
                else pd.Series([0.0] * len(df), index=df.index)
            )
            encoded[col] = (
                pd.to_numeric(source, errors="coerce").fillna(0.0).astype("float32")
            )
        return encoded[self.training_feature_columns]

    def _regression_metrics(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        rmse = (
            float(np.sqrt(mean_squared_error(y_true, y_pred)))
            if mean_squared_error
            else None
        )
        safe_true = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
        mape = None
        if mean_absolute_percentage_error is not None:
            try:
                mape = float(mean_absolute_percentage_error(safe_true, y_pred))
            except Exception:
                mape = None
        return {
            "samples": int(len(y_true)),
            "mae": (
                round(float(mean_absolute_error(y_true, y_pred)), 4)
                if mean_absolute_error
                else None
            ),
            "rmse": round(rmse, 4) if rmse is not None else None,
            "r2": round(float(r2_score(y_true, y_pred)), 4) if r2_score else None,
            "median_ae": (
                round(float(median_absolute_error(y_true, y_pred)), 4)
                if median_absolute_error
                else None
            ),
            "mape": round(mape, 6) if mape is not None else None,
        }

    def _prepare_ml_assets(self):
        association_metrics = (
            self.model_metrics.get("association_rule_mining")
            or self.association_rule_metrics
        )
        training_df = self._build_training_frame()
        self.model_metrics = {}
        if association_metrics:
            self.model_metrics["association_rule_mining"] = association_metrics
        if training_df.empty or len(training_df) < 20:
            self.trained_models = {}
            self.hybrid_model_weights = {}
            self.model_metrics["hybrid_ensemble"] = {
                "type": "weighted_regression_ensemble",
                "reason": "Not enough training samples were available for the ML models.",
                "weights": {},
                "metrics": {},
            }
            self.model_stack = {
                "association_rules": True,
                "hybrid_models": [],
                "hybrid_weights": self.hybrid_model_weights,
                "cache_loaded": False,
            }
            return
        if SKLEARN_AVAILABLE:
            train_df, test_df = train_test_split(
                training_df, test_size=0.2, random_state=42
            )
        else:
            split_index = max(1, int(len(training_df) * 0.8))
            train_df = training_df.iloc[:split_index].copy()
            test_df = training_df.iloc[split_index:].copy()
            if test_df.empty:
                test_df = training_df.copy()
        heuristic_pred = (
            pd.to_numeric(test_df["Predicted Closing Rank"], errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        y_test = pd.to_numeric(test_df["Latest Closing Rank"], errors="coerce").fillna(
            0.0
        )
        heuristic_metrics = self._regression_metrics(y_test, heuristic_pred)
        if not SKLEARN_AVAILABLE:
            self.trained_models = {}
            self.hybrid_model_weights = {}
            self.model_metrics["hybrid_ensemble"] = {
                "type": "weighted_regression_ensemble",
                "reason": "scikit-learn is not available in the current runtime; using historical predicted rank fallback.",
                "weights": {},
                "metrics": heuristic_metrics,
            }
            self.model_stack = {
                "association_rules": True,
                "hybrid_models": [],
                "hybrid_weights": self.hybrid_model_weights,
                "cache_loaded": False,
            }
            return
        self.training_feature_columns = list(
            self.MODEL_FEATURE_CATEGORICAL + self.MODEL_FEATURE_NUMERIC
        )
        _ = self._encode_feature_frame(training_df, fit=True)
        X_train = self._encode_feature_frame(train_df, fit=False)
        X_test = self._encode_feature_frame(test_df, fit=False)
        y_train = pd.to_numeric(
            train_df["Latest Closing Rank"], errors="coerce"
        ).fillna(0.0)
        candidate_models = {
            "decision_tree": DecisionTreeRegressor(
                random_state=42, max_depth=10, min_samples_leaf=3
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=96,
                random_state=42,
                n_jobs=1,
                max_depth=16,
                min_samples_leaf=2,
                max_features="sqrt",
            ),
            "gradient_boosting": GradientBoostingRegressor(
                random_state=42,
                n_estimators=120,
                learning_rate=0.05,
                subsample=0.9,
                max_depth=3,
            ),
        }
        self.trained_models = {}
        validation_predictions = {}
        for model_name, model in candidate_models.items():
            try:
                model.fit(X_train, y_train)
                preds = np.clip(
                    np.asarray(model.predict(X_test), dtype=float), 1.0, None
                )
                self.trained_models[model_name] = model
                validation_predictions[model_name] = preds
                self.model_metrics[model_name] = {
                    "type": "regressor",
                    "metrics": self._regression_metrics(y_test, preds),
                }
            except Exception as exc:
                print(f"Warning: {model_name} training failed:", exc)
        saved_weights = get_saved_weight_vector(
            "college_model_ensemble",
            names=list(validation_predictions.keys()),
            data_root_dir=self.data_root_dir,
        )
        weight_source = (
            "calibrated_weights.json" if saved_weights else "inverse_validation_mae"
        )
        weights = {}
        if saved_weights:
            weights = {
                model_name: float(saved_weights.get(model_name, 0.0))
                for model_name in validation_predictions.keys()
            }
        else:
            for model_name, info in self.model_metrics.items():
                metrics_blob = info.get("metrics", {})
                mae = metrics_blob.get("mae")
                if mae is None:
                    continue
                weights[model_name] = 1.0 / max(float(mae), 1e-6)
        if not weights:
            self.hybrid_model_weights = {}
            self.model_metrics["hybrid_ensemble"] = {
                "type": "weighted_regression_ensemble",
                "reason": "No ML model completed training; using historical predicted rank fallback.",
                "weights": {},
                "metrics": heuristic_metrics,
            }
            self.model_stack = {
                "association_rules": True,
                "hybrid_models": [],
                "hybrid_weights": self.hybrid_model_weights,
                "cache_loaded": False,
            }
            return
        total_weight = sum(weights.values()) or 1.0
        self.hybrid_model_weights = {
            name: round(value / total_weight, 6) for name, value in weights.items()
        }
        hybrid_pred = np.zeros(len(test_df), dtype=float)
        for model_name, weight in self.hybrid_model_weights.items():
            preds = validation_predictions.get(model_name)
            if preds is None:
                continue
            hybrid_pred += weight * np.asarray(preds, dtype=float)
        hybrid_pred = np.clip(hybrid_pred, 1.0, None)
        self.model_metrics["hybrid_ensemble"] = {
            "type": "weighted_regression_ensemble",
            "weights": self.hybrid_model_weights,
            "weight_source": weight_source,
            "metrics": self._regression_metrics(y_test, hybrid_pred),
        }
        self.model_stack = {
            "association_rules": True,
            "hybrid_models": list(self.hybrid_model_weights.keys()),
            "hybrid_weights": self.hybrid_model_weights,
            "cache_loaded": False,
        }

    def _persist_model_metrics(self):
        for stale_file in self.STALE_METRICS_FILENAMES:
            stale_path = os.path.join(self.results_dir, stale_file)
            if os.path.exists(stale_path):
                try:
                    os.remove(stale_path)
                except OSError:
                    pass
        for model_name, file_name in self.METRICS_FILENAMES.items():
            metrics_blob = self.model_metrics.get(model_name)
            if not metrics_blob:
                metrics_blob = {
                    "type": "unavailable",
                    "reason": (
                        "scikit-learn is not available in the current Python runtime."
                        if not SKLEARN_AVAILABLE
                        else "Model training was skipped or did not complete."
                    ),
                    "metrics": {},
                }
            metrics_blob = self._json_safe(metrics_blob)
            path = os.path.join(self.results_dir, file_name)
            with open(path, "w", encoding="utf-8") as file_obj:
                json.dump(
                    {
                        "model_name": model_name,
                        "saved_from_cache": bool(self.model_stack.get("cache_loaded")),
                        "stack": self.model_stack,
                        **metrics_blob,
                    },
                    file_obj,
                    indent=2,
                )

    def _attach_hybrid_rank_predictions(self, df):
        if df.empty:
            return df.copy(), "Predicted Closing Rank"
        ranked_df = df.copy()
        ranked_df["Heuristic Closing Rank"] = pd.to_numeric(
            ranked_df.get("Predicted Closing Rank", 0),
            errors="coerce",
        ).fillna(0.0)
        if not self.hybrid_model_weights:
            ranked_df["Hybrid Predicted Rank"] = ranked_df["Heuristic Closing Rank"]
            return ranked_df, "Hybrid Predicted Rank"
        if not self.trained_models:
            ranked_df["Hybrid Predicted Rank"] = ranked_df["Heuristic Closing Rank"]
            return ranked_df, "Hybrid Predicted Rank"
        encoded = self._encode_feature_frame(ranked_df, fit=False)
        component_predictions = {
            "heuristic": ranked_df["Heuristic Closing Rank"].to_numpy(dtype=float)
        }
        for model_name, model in self.trained_models.items():
            try:
                component_predictions[model_name] = np.clip(
                    np.asarray(model.predict(encoded), dtype=float),
                    1.0,
                    None,
                )
            except Exception:
                continue
        hybrid_rank = np.zeros(len(ranked_df), dtype=float)
        used_weight = 0.0
        for model_name, weight in self.hybrid_model_weights.items():
            preds = component_predictions.get(model_name)
            if preds is None:
                continue
            hybrid_rank += weight * preds
            used_weight += weight
        if used_weight <= 0:
            hybrid_rank = component_predictions["heuristic"]
        elif not np.isclose(used_weight, 1.0):
            hybrid_rank = hybrid_rank / used_weight
        ranked_df["Hybrid Predicted Rank"] = np.clip(hybrid_rank, 1.0, None)
        return ranked_df, "Hybrid Predicted Rank"

    def _apply_hybrid_ranking(self, df, user_rank_val, rule_boosts, effective_rank_col):
        ranked_df = df.copy()

        def key_for_row(row):
            return (row.get("Institute", ""), row.get("Program", ""))

        ranked_df["Rule Boost"] = ranked_df.apply(
            lambda row: rule_boosts.get(key_for_row(row), 0.0), axis=1
        )
        ranked_df["Rule Boost Score"] = self._normalize_series(ranked_df["Rule Boost"])
        quality_ctc = self._normalize_series(
            ranked_df.get(
                "Max Average CTC", pd.Series(index=ranked_df.index, dtype=float)
            )
        )
        quality_place = self._normalize_series(
            ranked_df.get(
                "placements_score_filter", pd.Series(index=ranked_df.index, dtype=float)
            )
        )
        quality_overall = self._normalize_series(
            ranked_df.get(
                "overall_aspect_score_filter",
                pd.Series(index=ranked_df.index, dtype=float),
            )
        )
        quality_weights = get_weight_vector(
            "college_quality_score",
            names=["ctc", "placement", "overall"],
            data_root_dir=self.data_root_dir,
        )
        ranked_df["Quality Score"] = (
            (float(quality_weights.get("ctc", 0.0)) * quality_ctc)
            + (float(quality_weights.get("placement", 0.0)) * quality_place)
            + (float(quality_weights.get("overall", 0.0)) * quality_overall)
        )
        gap = pd.to_numeric(ranked_df[effective_rank_col], errors="coerce").fillna(
            0.0
        ) - float(user_rank_val)
        gap = gap.clip(lower=0.0)
        max_gap = float(gap.max()) if len(gap) else 0.0
        if max_gap <= 0:
            accessibility = pd.Series(
                [1.0] * len(ranked_df), index=ranked_df.index, dtype=float
            )
        else:
            accessibility = 1.0 - (gap / max_gap)
        ranked_df["Accessibility Score"] = accessibility
        recommendation_weights = get_weight_vector(
            "college_recommendation_score",
            names=["accessibility", "quality", "rule_boost"],
            data_root_dir=self.data_root_dir,
        )
        ranked_df["Hybrid Recommendation Score"] = (
            (
                float(recommendation_weights.get("accessibility", 0.0))
                * ranked_df["Accessibility Score"]
            )
            + (
                float(recommendation_weights.get("quality", 0.0))
                * ranked_df["Quality Score"]
            )
            + (
                float(recommendation_weights.get("rule_boost", 0.0))
                * ranked_df["Rule Boost Score"]
            )
        )
        ranked_df = ranked_df.sort_values(
            by=[
                "Hybrid Recommendation Score",
                "Rule Boost",
                "Quality Score",
                effective_rank_col,
                "Max Average CTC",
                "overall_aspect_score_filter",
            ],
            ascending=[False, False, False, True, False, False],
        )
        return ranked_df

    def _finalize_table(self, ranked_filtered_df, effective_rank_col):
        rank_col = (
            effective_rank_col
            if effective_rank_col in ranked_filtered_df.columns
            else "Predicted Closing Rank"
        )
        base_cols = [
            "Institute",
            "Program",
            "Stream",
            "Seat Type",
            "Quota",
            "Category",
            "District",
            "Opening Rank",
            "Predicted Closing Rank",
            "Heuristic Closing Rank",
            "Hybrid Predicted Rank",
            "Rule Boost",
            "Quality Score",
            "Hybrid Recommendation Score",
        ]
        for col in base_cols:
            if col not in ranked_filtered_df.columns:
                ranked_filtered_df[col] = ""
        final_df = ranked_filtered_df[base_cols].copy()
        final_df["Closing Rank"] = pd.to_numeric(
            ranked_filtered_df[rank_col], errors="coerce"
        ).fillna(0.0)
        if not getattr(self, "full_college_df", pd.DataFrame()).empty:
            final_df = pd.merge(
                final_df,
                self.full_college_df,
                on="Institute",
                how="left",
                suffixes=("", "_college"),
            )
            if "District_college" in final_df.columns:
                final_df["District"] = (
                    final_df["District"]
                    .replace("", np.nan)
                    .fillna(final_df["District_college"])
                )
                final_df = final_df.drop(columns=["District_college"])
        else:
            for col in ["Location", "Website", "logo_image", "Picture"]:
                final_df[col] = ""
        if not getattr(self, "full_placement_df", pd.DataFrame()).empty:
            final_df = pd.merge(
                final_df,
                self.full_placement_df,
                on=["Institute", "Program"],
                how="left",
            )
        if not getattr(self, "full_reviews_df", pd.DataFrame()).empty:
            final_df = pd.merge(
                final_df, self.full_reviews_df, on="Institute", how="left"
            )
        all_cols = [
            "Institute",
            "Program",
            "Stream",
            "Seat Type",
            "Quota",
            "Category",
            "District",
            "Location",
            "Website",
            "logo_image",
            "Picture",
            "Opening Rank",
            "Closing Rank",
            "Heuristic Closing Rank",
            "Hybrid Predicted Rank",
            "Hybrid Recommendation Score",
            "Rule Boost",
            "average_ctc",
            "median_ctc",
            "highest_ctc",
            "top recruiter",
            "job_title",
            "institute_rank",
            "rating",
            "sentiment_score",
            "mess_score",
            "professor_score",
            "campus_score",
            "placement_score",
            "infrastructure_score",
            "overall_aspect_score",
        ]
        for col in all_cols:
            if col not in final_df.columns:
                final_df[col] = ""
        final_df = final_df.reindex(columns=all_cols)
        numeric_cols = [
            "Opening Rank",
            "Closing Rank",
            "Heuristic Closing Rank",
            "Hybrid Predicted Rank",
            "Hybrid Recommendation Score",
            "Rule Boost",
            "average_ctc",
            "median_ctc",
            "highest_ctc",
            "institute_rank",
            "rating",
            "sentiment_score",
            "mess_score",
            "professor_score",
            "campus_score",
            "placement_score",
            "infrastructure_score",
            "overall_aspect_score",
        ]
        for col in numeric_cols:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce").fillna(0.0)
        for col in final_df.columns:
            if col not in numeric_cols:
                final_df[col] = final_df[col].fillna("")
        return final_df.head(10)

    def recommend(
        self,
        user_rank,
        user_program,
        user_stream="",
        user_quota="",
        user_category="",
        user_location="",
        min_ctc=0,
        min_placements_score=0,
        target_year=2026,
    ):
        ranked_predictions_df = self._predict_top_colleges_rank_only(
            program=user_program,
            stream=user_stream,
            quota=user_quota,
            category=user_category,
            district=user_location,
            target_year=target_year,
        )
        if ranked_predictions_df.empty:
            return {
                "status": "error",
                "message": "No historical data found for the specified filters.",
            }
        try:
            user_rank_val = float(user_rank)
        except Exception:
            return {"status": "error", "message": "Invalid value for user_rank."}
        working_df = self._merge_quality_columns(ranked_predictions_df)
        working_df, effective_rank_col = self._attach_hybrid_rank_predictions(
            working_df
        )
        ranked_results_df = working_df[
            pd.to_numeric(working_df[effective_rank_col], errors="coerce").fillna(0.0)
            >= user_rank_val
        ].copy()
        ranked_results_df = ranked_results_df.sort_values(
            by=effective_rank_col, ascending=True
        )
        if ranked_results_df.empty:
            return {
                "status": "error",
                "message": (
                    f"No colleges found with a predicted closing rank >= {user_rank_val}. "
                    "Consider increasing your expected rank (higher number) or broadening filters."
                ),
            }
        if min_ctc > 0 or min_placements_score > 0:
            ranked_results_df = self._filter_top_colleges_by_metrics(
                ranked_results_df,
                min_ctc=min_ctc,
                min_placements_score=min_placements_score,
            )
        if ranked_results_df.empty:
            return {
                "status": "warning",
                "message": "Quality filters removed all candidates. Try lowering min CTC/score or broadening other filters.",
                "data": [],
            }
        user_filters = {
            "program": self._clean_user_input(user_program),
            "stream": self._clean_user_input(user_stream),
            "quota": self._clean_user_input(user_quota),
            "category": self._clean_user_input(user_category),
            "district": self._clean_user_input(user_location),
        }
        boost_input_df = ranked_results_df.copy()
        for col in ["Program", "Stream", "Quota", "Category", "District", "Institute"]:
            if col not in boost_input_df.columns:
                boost_input_df[col] = ""
            boost_input_df[col] = (
                boost_input_df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace("nan", "")
                .fillna("")
            )
        rule_boosts = self._compute_boosts_from_rules(boost_input_df, user_filters)
        ranked_results_df = self._apply_hybrid_ranking(
            ranked_results_df,
            user_rank_val=user_rank_val,
            rule_boosts=rule_boosts,
            effective_rank_col=effective_rank_col,
        )
        final_table_candidates = self._finalize_table(
            ranked_results_df, effective_rank_col=effective_rank_col
        )
        result_list = final_table_candidates.head(10).to_dict("records")
        return {
            "status": "success",
            "message": "Top college recommendations based on historical ranks, Apriori rule boosts, and cached hybrid ensemble scoring.",
            "data": result_list,
            "model_stack": self.model_stack,
        }
