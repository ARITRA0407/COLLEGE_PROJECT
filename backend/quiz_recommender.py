import json
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from calibrated_weights import get_weight_vector
except Exception:

    def get_weight_vector(weight_key, names=None, data_root_dir=None):
        defaults = {
            "quiz_model_ensemble": {
                "random_forest": 0.55,
                "gradient_boosting": 0.45,
            },
            "career_recommendation_score": {
                "rule": 0.24,
                "cosine": 0.19,
                "latent": 0.15,
                "model_probability": 0.22,
                "readiness": 0.15,
                "reliability": 0.05,
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
    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    PCA = None
    GradientBoostingClassifier = None
    RandomForestClassifier = None
    accuracy_score = None
    f1_score = None
    log_loss = None
    top_k_accuracy_score = None
    cosine_similarity = None
    train_test_split = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False
LEVEL_TO_SCORE = {
    "low": 0.25,
    "medium": 0.60,
    "high": 0.90,
}


# Career engine
class HybridCareerRecommender:

    CACHE_FILENAME = "hybrid_quiz_recommender.pkl"

    def __init__(self, career_df, output_dir):
        self.output_dir = os.path.abspath(output_dir)
        self.data_root_dir = os.path.dirname(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.cache_path = os.path.join(self.output_dir, self.CACHE_FILENAME)
        self.career_df = self._prepare_career_df(career_df)
        self.skill_cols = [col for col in self.career_df.columns if col != "Job"]
        self.job_names = self.career_df["Job"].astype(str).tolist()
        self.prototype_matrix = self.career_df[self.skill_cols].to_numpy(dtype=float)
        self.scaler = None
        self.pca = None
        self.prototype_latent = None
        self.rf_model = None
        self.gb_model = None
        self.model_metrics = {
            "training_mode": "hybrid_fallback",
            "synthetic_samples": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "macro_f1": 0.0,
            "log_loss": None,
        }
        if self._load_cache():
            return
        self._fit_models()
        self._save_cache()

    def _prepare_career_df(self, career_df):
        df = career_df.copy()
        df.columns = [str(col).strip() for col in df.columns]
        job_col = next((col for col in df.columns if col.lower() == "job"), "Job")
        if job_col != "Job":
            df = df.rename(columns={job_col: "Job"})
        for col in df.columns:
            if col == "Job":
                df[col] = df[col].astype(str).fillna("Unknown Role")
                continue
            df[col] = df[col].apply(self._level_to_score)
        return df

    def _level_to_score(self, value):
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(np.clip(value, 0.0, 1.0))
        text = str(value).strip().lower()
        if text in LEVEL_TO_SCORE:
            return LEVEL_TO_SCORE[text]
        return 0.40

    def _cache_signature(self):
        career_hash = pd.util.hash_pandas_object(self.career_df, index=True).sum()
        weights_path = os.path.join(
            self.data_root_dir, "results", "calibrated_weights.json"
        )
        return {
            "module_mtime": (
                os.path.getmtime(__file__) if os.path.exists(__file__) else 0
            ),
            "calibrated_weights_module_mtime": os.path.getmtime(
                os.path.join(os.path.dirname(__file__), "calibrated_weights.py")
            ),
            "calibrated_weights_mtime": (
                os.path.getmtime(weights_path) if os.path.exists(weights_path) else 0
            ),
            "career_hash": str(int(career_hash)),
            "sklearn_available": bool(SKLEARN_AVAILABLE),
        }

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "rb") as cache_file:
                payload = pickle.load(cache_file)
        except Exception:
            return False
        if payload.get("signature") != self._cache_signature():
            return False
        for attr in [
            "career_df",
            "skill_cols",
            "job_names",
            "prototype_matrix",
            "scaler",
            "pca",
            "prototype_latent",
            "rf_model",
            "gb_model",
            "model_metrics",
        ]:
            if attr in payload:
                setattr(self, attr, payload[attr])
        return True

    def _save_cache(self):
        payload = {
            "signature": self._cache_signature(),
            "career_df": self.career_df,
            "skill_cols": self.skill_cols,
            "job_names": self.job_names,
            "prototype_matrix": self.prototype_matrix,
            "scaler": self.scaler,
            "pca": self.pca,
            "prototype_latent": self.prototype_latent,
            "rf_model": self.rf_model,
            "gb_model": self.gb_model,
            "model_metrics": self.model_metrics,
        }
        with open(self.cache_path, "wb") as cache_file:
            pickle.dump(payload, cache_file, protocol=pickle.HIGHEST_PROTOCOL)

    def _quiz_model_weights(self):
        return get_weight_vector(
            "quiz_model_ensemble",
            names=["random_forest", "gradient_boosting"],
            data_root_dir=self.data_root_dir,
        )

    def _career_score_weights(self):
        return get_weight_vector(
            "career_recommendation_score",
            names=[
                "rule",
                "cosine",
                "latent",
                "model_probability",
                "readiness",
                "reliability",
            ],
            data_root_dir=self.data_root_dir,
        )

    def _blend_model_probabilities(self, rf_probs, gb_probs):
        weights = self._quiz_model_weights()
        return (
            float(weights.get("random_forest", 0.0)) * np.asarray(rf_probs, dtype=float)
        ) + (
            float(weights.get("gradient_boosting", 0.0))
            * np.asarray(gb_probs, dtype=float)
        )

    def _normalize_rows(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _build_synthetic_training_data(self, samples_per_role=280):
        rng = np.random.default_rng(42)
        vectors = self.prototype_matrix
        features = []
        labels = []
        for role_index, vector in enumerate(vectors):
            features.append(vector.copy())
            labels.append(role_index)
            distances = np.linalg.norm(vectors - vector, axis=1)
            neighbor_indices = np.argsort(distances)[1:4]
            for _ in range(samples_per_role):
                noise = rng.normal(0, rng.uniform(0.04, 0.14), size=vector.shape)
                sample = vector + noise
                if len(neighbor_indices) and rng.random() < 0.35:
                    neighbor = vectors[int(rng.choice(neighbor_indices))]
                    blend = rng.uniform(0.72, 0.92)
                    sample = (
                        (blend * vector) + ((1.0 - blend) * neighbor) + (noise * 0.5)
                    )
                sample = np.clip(sample, 0.0, 1.0)
                features.append(sample)
                labels.append(role_index)
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=int)
        return features, labels

    def _fit_models(self):
        if not SKLEARN_AVAILABLE or len(self.job_names) < 3:
            return
        try:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(self.prototype_matrix)
            n_components = max(2, min(6, scaled.shape[0] - 1, scaled.shape[1]))
            self.pca = PCA(n_components=n_components, random_state=42)
            self.prototype_latent = self._normalize_rows(self.pca.fit_transform(scaled))
            X, y = self._build_synthetic_training_data()
            self.model_metrics["synthetic_samples"] = int(len(X))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.22, random_state=42, stratify=y
            )
            self.rf_model = RandomForestClassifier(
                n_estimators=320, min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            self.gb_model = GradientBoostingClassifier(random_state=42)
            self.gb_model.fit(X_train, y_train)
            rf_probs = self.rf_model.predict_proba(X_test)
            gb_probs = self.gb_model.predict_proba(X_test)
            ensemble_probs = self._blend_model_probabilities(rf_probs, gb_probs)
            ensemble_pred = np.argmax(ensemble_probs, axis=1)
            self.model_metrics = {
                "training_mode": "hybrid_ensemble",
                "synthetic_samples": int(len(X)),
                "ensemble_weights": self._quiz_model_weights(),
                "top1_accuracy": round(float(accuracy_score(y_test, ensemble_pred)), 4),
                "top3_accuracy": round(
                    float(
                        top_k_accuracy_score(
                            y_test,
                            ensemble_probs,
                            k=3,
                            labels=np.arange(len(self.job_names)),
                        )
                    ),
                    4,
                ),
                "macro_f1": round(
                    float(f1_score(y_test, ensemble_pred, average="macro")), 4
                ),
                "log_loss": round(
                    float(
                        log_loss(
                            y_test,
                            ensemble_probs,
                            labels=np.arange(len(self.job_names)),
                        )
                    ),
                    4,
                ),
            }
        except Exception:
            self.scaler = None
            self.pca = None
            self.prototype_latent = None
            self.rf_model = None
            self.gb_model = None
            self.model_metrics = {
                "training_mode": "hybrid_fallback",
                "synthetic_samples": 0,
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "macro_f1": 0.0,
                "log_loss": None,
            }

    def _latent_similarity(self, user_vector):
        if (
            not SKLEARN_AVAILABLE
            or self.scaler is None
            or self.pca is None
            or self.prototype_latent is None
        ):
            return np.zeros(len(self.job_names))
        scaled_user = self.scaler.transform([user_vector])
        latent_user = self._normalize_rows(self.pca.transform(scaled_user))
        latent_scores = np.dot(self.prototype_latent, latent_user[0])
        return np.clip(latent_scores, 0.0, 1.0)

    def _ensemble_probabilities(self, user_vector):
        if self.rf_model is None or self.gb_model is None:
            return np.zeros(len(self.job_names))
        rf_probs = self.rf_model.predict_proba([user_vector])[0]
        gb_probs = self.gb_model.predict_proba([user_vector])[0]
        ensemble_probs = self._blend_model_probabilities(rf_probs, gb_probs)
        return np.clip(ensemble_probs, 0.0, 1.0)

    def _rule_score(self, user_levels, role_vector):
        user_level_vector = np.array(
            [
                LEVEL_TO_SCORE.get(
                    str(user_levels.get(col, "Low")).strip().lower(), 0.25
                )
                for col in self.skill_cols
            ],
            dtype=float,
        )
        coverage = (user_level_vector >= role_vector).astype(float)
        return float(np.mean(coverage))

    def _readiness_score(self, user_vector, role_vector):
        gaps = np.maximum(role_vector - user_vector, 0.0)
        return float(np.clip(1.0 - np.mean(gaps), 0.0, 1.0))

    def _alignment_score(self, user_vector, role_vector):
        return float(
            np.clip(1.0 - np.mean(np.abs(user_vector - role_vector)), 0.0, 1.0)
        )

    def _career_reliability(self, category_metrics, role_vector):
        weights = role_vector / max(float(np.sum(role_vector)), 1e-8)
        reliability_values = np.array(
            [
                float(category_metrics.get(col, {}).get("reliability", 0.5))
                for col in self.skill_cols
            ],
            dtype=float,
        )
        return float(np.clip(np.dot(reliability_values, weights), 0.0, 1.0))

    def _strengths_and_gaps(self, user_vector, role_vector):
        deltas = user_vector - role_vector
        ordered_positive = np.argsort(deltas)[::-1]
        ordered_negative = np.argsort(role_vector - user_vector)[::-1]
        strengths = []
        for idx in ordered_positive:
            category = self.skill_cols[int(idx)]
            if role_vector[idx] >= 0.55 or user_vector[idx] >= 0.70:
                strengths.append(category)
            if len(strengths) == 3:
                break
        growth_areas = []
        for idx in ordered_negative:
            category = self.skill_cols[int(idx)]
            if (role_vector[idx] - user_vector[idx]) > 0.08:
                growth_areas.append(category)
            if len(growth_areas) == 2:
                break
        return strengths, growth_areas

    def recommend(self, user_scores, user_levels, category_metrics, top_n=5):
        # Score roles
        user_vector = np.array(
            [
                float(np.clip(user_scores.get(col, 0.0), 0.0, 1.0))
                for col in self.skill_cols
            ],
            dtype=float,
        )
        if np.allclose(user_vector, 0.0):
            user_vector = np.full(len(self.skill_cols), 0.25, dtype=float)
        cosine_scores = np.dot(
            self._normalize_rows(self.prototype_matrix),
            self._normalize_rows([user_vector])[0],
        )
        cosine_scores = np.clip(cosine_scores, 0.0, 1.0)
        latent_scores = self._latent_similarity(user_vector)
        ensemble_probs = self._ensemble_probabilities(user_vector)
        recommendations = []
        for idx, (_, row) in enumerate(self.career_df.iterrows()):
            role_vector = row[self.skill_cols].to_numpy(dtype=float)
            rule_score = self._rule_score(user_levels, role_vector)
            readiness_score = self._readiness_score(user_vector, role_vector)
            alignment_score = self._alignment_score(user_vector, role_vector)
            reliability_score = self._career_reliability(category_metrics, role_vector)
            strengths, growth_areas = self._strengths_and_gaps(user_vector, role_vector)
            score_weights = self._career_score_weights()
            final_score = (
                (float(score_weights.get("rule", 0.0)) * rule_score)
                + (float(score_weights.get("cosine", 0.0)) * float(cosine_scores[idx]))
                + (float(score_weights.get("latent", 0.0)) * float(latent_scores[idx]))
                + (
                    float(score_weights.get("model_probability", 0.0))
                    * float(ensemble_probs[idx])
                )
                + (float(score_weights.get("readiness", 0.0)) * readiness_score)
                + (float(score_weights.get("reliability", 0.0)) * reliability_score)
            )
            recommendations.append(
                {
                    "Job": row["Job"],
                    "score": round(float(final_score * 100), 2),
                    "match_score": round(float(final_score), 4),
                    "rule_score": round(float(rule_score), 4),
                    "cosine_score": round(float(cosine_scores[idx]), 4),
                    "latent_score": round(float(latent_scores[idx]), 4),
                    "model_probability": round(float(ensemble_probs[idx]), 4),
                    "readiness_score": round(float(readiness_score), 4),
                    "alignment_score": round(float(alignment_score), 4),
                    "reliability_score": round(float(reliability_score), 4),
                    "matched_strengths": strengths,
                    "growth_areas": growth_areas,
                }
            )
        recommendations.sort(key=lambda item: item["match_score"], reverse=True)
        return recommendations[:top_n]

    def build_report(
        self, category_metrics, recommendations, summary_metrics, store_history=False
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "generated_at": datetime.now().isoformat(),
            "model_stack": {
                "retrieval_model": "cosine_similarity + PCA_latent_matching",
                "reranker_model": self.model_metrics.get(
                    "training_mode", "hybrid_fallback"
                ),
                "training_metrics": self.model_metrics,
                "runtime_weights": {
                    "quiz_model_ensemble": self._quiz_model_weights(),
                    "career_recommendation_score": self._career_score_weights(),
                },
            },
            "summary_metrics": summary_metrics,
            "category_metrics": category_metrics,
            "recommendations": recommendations,
        }
        latest_path = os.path.join(self.output_dir, "latest_quiz_recommendation.json")
        timestamped_path = ""
        with open(latest_path, "w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, indent=2)
        if store_history:
            timestamped_path = os.path.join(
                self.output_dir, f"quiz_recommendation_{timestamp}.json"
            )
            with open(timestamped_path, "w", encoding="utf-8") as file_obj:
                json.dump(report, file_obj, indent=2)
        return report, {
            "latest": latest_path,
            "timestamped": timestamped_path,
        }
