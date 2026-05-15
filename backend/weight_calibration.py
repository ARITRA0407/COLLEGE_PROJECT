import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "calibrated_weight_search.txt"
WEIGHTS_PATH = RESULTS_DIR / "calibrated_weights.json"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from calibrated_weights import DEFAULT_WEIGHT_CONFIG


def _as_float_array(values):
    return np.asarray(values, dtype=float)


def _normalise(values, invert=False, log_scale=False):
    arr = _as_float_array(values)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if log_scale:
        arr = np.log1p(np.clip(arr, 0.0, None))
    min_val = float(np.min(arr)) if arr.size else 0.0
    max_val = float(np.max(arr)) if arr.size else 0.0
    if max_val - min_val <= 1e-12:
        norm = np.zeros_like(arr, dtype=float)
    else:
        norm = (arr - min_val) / (max_val - min_val)
    if invert:
        norm = 1.0 - norm
    return np.clip(norm, 0.0, 1.0)


def _normalise_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _clip01(values):
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _simplex_grid(names, step):
    names = list(names)
    units = int(round(1.0 / step))
    if not names:
        return

    def walk(index, remaining, chosen):
        if index == len(names) - 1:
            yield chosen + [remaining]
            return
        for value in range(remaining + 1):
            yield from walk(index + 1, remaining - value, chosen + [value])

    for raw in walk(0, units, []):
        weights = {name: raw[idx] / units for idx, name in enumerate(names)}
        if sum(weights.values()) > 0:
            yield weights


def _weights_to_array(weights, names):
    return np.asarray([float(weights.get(name, 0.0)) for name in names], dtype=float)


def _regression_metrics(y_true, y_pred):
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err))) if len(err) else 0.0
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0
    median_ae = float(np.median(np.abs(err))) if len(err) else 0.0
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2)) if len(y_true) else 0.0
    r2 = 1.0 - (float(np.sum(err**2)) / denom) if denom > 1e-12 else 0.0
    return {
        "samples": int(len(y_true)),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "median_ae": round(median_ae, 6),
        "r2": round(float(r2), 6),
    }


def _macro_f1(y_true, y_pred, labels):
    scores = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2.0 * precision * recall) / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _softmax(scores):
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    denom = np.sum(exp_scores, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp_scores / denom


def _classification_metrics(y_true, scores, probabilities=False):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] == 0:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "top3_accuracy": 0.0,
            "macro_f1": 0.0,
            "log_loss": None,
        }
    if probabilities:
        probs = np.clip(scores, 1e-12, 1.0)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
    else:
        probs = np.clip(_softmax(scores), 1e-12, 1.0)
    pred = np.argmax(scores, axis=1)
    labels = np.arange(scores.shape[1])
    accuracy = float(np.mean(pred == y_true))
    top_k = min(3, scores.shape[1])
    top_indices = np.argsort(scores, axis=1)[:, -top_k:]
    top3_accuracy = float(
        np.mean([y_true[idx] in top_indices[idx] for idx in range(len(y_true))])
    )
    macro_f1 = _macro_f1(y_true, pred, labels)
    row_probs = probs[np.arange(len(y_true)), y_true]
    log_loss = float(-np.mean(np.log(np.clip(row_probs, 1e-12, 1.0))))
    return {
        "samples": int(len(y_true)),
        "accuracy": round(accuracy, 6),
        "top3_accuracy": round(top3_accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "log_loss": round(log_loss, 6),
    }


def _evaluate_regression_grid(components, y_true, names, step, top_n=25):
    components = np.asarray(components, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    rows = []
    for weights in _simplex_grid(names, step):
        pred = components @ _weights_to_array(weights, names)
        metrics = _regression_metrics(y_true, pred)
        rows.append(
            {
                "weights": {
                    key: round(float(value), 6) for key, value in weights.items()
                },
                "metrics": metrics,
                "sort_key": (metrics["rmse"], metrics["mae"], -metrics["r2"]),
            }
        )
    rows.sort(key=lambda item: item["sort_key"])
    return rows[:top_n], len(rows)


def _evaluate_probability_grid(probabilities, y_true, names, step, top_n=25):
    rows = []
    for weights in _simplex_grid(names, step):
        blended = np.zeros_like(probabilities[0], dtype=float)
        for idx, name in enumerate(names):
            blended += float(weights.get(name, 0.0)) * probabilities[idx]
        metrics = _classification_metrics(y_true, blended, probabilities=True)
        rows.append(
            {
                "weights": {
                    key: round(float(value), 6) for key, value in weights.items()
                },
                "metrics": metrics,
                "sort_key": (
                    -metrics["accuracy"],
                    metrics["log_loss"],
                    -metrics["macro_f1"],
                ),
            }
        )
    rows.sort(key=lambda item: item["sort_key"])
    return rows[:top_n], len(rows)


def _evaluate_multiclass_score_grid(components, y_true, names, step, top_n=25):
    rows = []
    for weights in _simplex_grid(names, step):
        scores = np.tensordot(
            components, _weights_to_array(weights, names), axes=([2], [0])
        )
        metrics = _classification_metrics(y_true, scores, probabilities=False)
        rows.append(
            {
                "weights": {
                    key: round(float(value), 6) for key, value in weights.items()
                },
                "metrics": metrics,
                "sort_key": (
                    -metrics["accuracy"],
                    metrics["log_loss"],
                    -metrics["macro_f1"],
                ),
            }
        )
    rows.sort(key=lambda item: item["sort_key"])
    return rows[:top_n], len(rows)


def _format_weights(weights):
    return ", ".join(f"{key}={value:.3f}" for key, value in weights.items())


def _format_section(title, rows, tested, selection_metric, skipped_reason=None):
    lines = []
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))
    if skipped_reason:
        lines.append(f"Skipped: {skipped_reason}")
        return lines
    lines.append(f"Tested cases: {tested}")
    lines.append(f"Selection metric: {selection_metric}")
    lines.append("")
    header = f"{'rank':>4}  {'weights':<95}  {'rmse':>10}  {'mae':>10}  {'r2':>8}  {'acc':>8}  {'f1':>8}  {'logloss':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for idx, row in enumerate(rows, start=1):
        metrics = row.get("metrics", {})
        lines.append(
            f"{idx:>4}  "
            f"{_format_weights(row.get('weights', {})):<95}  "
            f"{str(metrics.get('rmse', '')):>10}  "
            f"{str(metrics.get('mae', '')):>10}  "
            f"{str(metrics.get('r2', '')):>8}  "
            f"{str(metrics.get('accuracy', '')):>8}  "
            f"{str(metrics.get('macro_f1', '')):>8}  "
            f"{str(metrics.get('log_loss', '')):>10}"
        )
    return lines


def _best_payload(rows, selection_metric, source, tested):
    if not rows:
        return None
    best = rows[0]
    return {
        "weights": best["weights"],
        "metrics": best["metrics"],
        "selection_metric": selection_metric,
        "source": source,
        "tested_cases": int(tested),
    }


def _load_college_recommender():
    try:
        from recommendation import CollegeRecommender

        return CollegeRecommender(data_root_dir=str(PROJECT_ROOT)), None
    except Exception as exc:
        return None, str(exc)


def _college_training_frame(recommender):
    try:
        training_df = recommender._build_training_frame()
    except Exception:
        training_df = pd.DataFrame()
    return training_df if isinstance(training_df, pd.DataFrame) else pd.DataFrame()


def _calibrate_college_model_ensemble(recommender, training_df):
    try:
        from recommendation import SKLEARN_AVAILABLE, train_test_split
    except Exception:
        SKLEARN_AVAILABLE = False
        train_test_split = None
    names = ["decision_tree", "random_forest", "gradient_boosting"]
    if not SKLEARN_AVAILABLE or train_test_split is None:
        return (
            [],
            0,
            "scikit-learn is not available, so DT/RF/GB validation predictions cannot be generated.",
        )
    if training_df.empty or len(training_df) < 20:
        return [], 0, "not enough college rank training rows."
    available_names = [
        name for name in names if name in getattr(recommender, "trained_models", {})
    ]
    if not available_names:
        return [], 0, "trained rank models are not available."
    try:
        _, test_df = train_test_split(training_df, test_size=0.2, random_state=42)
        X_test = recommender._encode_feature_frame(test_df, fit=False)
        y_test = (
            pd.to_numeric(test_df["Latest Closing Rank"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        predictions = []
        for model_name in available_names:
            model = recommender.trained_models[model_name]
            predictions.append(
                np.clip(np.asarray(model.predict(X_test), dtype=float), 1.0, None)
            )
        components = np.column_stack(predictions)
        rows, tested = _evaluate_regression_grid(
            components=components,
            y_true=y_test,
            names=available_names,
            step=0.05,
            top_n=30,
        )
        return rows, tested, None
    except Exception as exc:
        return [], 0, str(exc)


def _quality_components(training_df):
    ctc = _normalise(training_df.get("Max Average CTC", 0.0))
    placement = _normalise(training_df.get("placements_score_filter", 0.0))
    overall = _normalise(training_df.get("overall_aspect_score_filter", 0.0))
    components = np.column_stack([ctc, placement, overall])
    target = _normalise(
        training_df.get("Latest Closing Rank", 0.0), invert=True, log_scale=True
    )
    return components, target


def _calibrate_college_quality(training_df):
    if training_df.empty:
        return [], 0, "college training frame is empty."
    components, target = _quality_components(training_df)
    names = ["ctc", "placement", "overall"]
    rows, tested = _evaluate_regression_grid(
        components=components,
        y_true=target,
        names=names,
        step=0.05,
        top_n=30,
    )
    return rows, tested, None


def _rule_strengths(recommender, training_df):
    rules_df = getattr(recommender, "assoc_rules_df", pd.DataFrame())
    if not isinstance(rules_df, pd.DataFrame) or rules_df.empty:
        return np.zeros(len(training_df), dtype=float)
    prepared_rules = []
    for _, rule in rules_df.iterrows():
        confidence = float(rule.get("confidence", 0.0) or 0.0)
        support = float(rule.get("support", 0.0) or 0.0)
        token_a = str(rule.get("antecedent", "")).strip().lower()
        token_b = str(rule.get("consequent", "")).strip().lower()
        strength = confidence * support
        if strength > 0:
            prepared_rules.append((token_a, token_b, strength))
    values = []
    for _, row in training_df.iterrows():
        tokens = {
            f"institute={str(row.get('Institute', '')).strip().lower()}",
            f"program={str(row.get('Program', '')).strip().lower()}",
            f"stream={str(row.get('Stream', '')).strip().lower()}",
            f"quota={str(row.get('Quota', '')).strip().lower()}",
            f"category={str(row.get('Category', '')).strip().lower()}",
        }
        best = 0.0
        for token_a, token_b, strength in prepared_rules:
            if token_a in tokens or token_b in tokens:
                best = max(best, strength)
        values.append(best)
    return _normalise(values)


def _calibrate_college_recommendation(recommender, training_df, quality_weights):
    if training_df.empty:
        return [], 0, "college training frame is empty."
    quality_matrix, target = _quality_components(training_df)
    quality_names = ["ctc", "placement", "overall"]
    quality_weight_array = _weights_to_array(quality_weights, quality_names)
    quality_score = quality_matrix @ quality_weight_array
    latest = (
        pd.to_numeric(training_df.get("Latest Closing Rank", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    predicted = (
        pd.to_numeric(training_df.get("Predicted Closing Rank", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    accessibility = 1.0 - _normalise(np.abs(predicted - latest))
    rule_boost = _rule_strengths(recommender, training_df)
    components = np.column_stack(
        [
            _clip01(accessibility),
            _clip01(quality_score),
            _clip01(rule_boost),
        ]
    )
    names = ["accessibility", "quality", "rule_boost"]
    rows, tested = _evaluate_regression_grid(
        components=components,
        y_true=target,
        names=names,
        step=0.05,
        top_n=30,
    )
    return rows, tested, None


def _load_quiz_model():
    try:
        from quiz_recommender import HybridCareerRecommender

        career_df = pd.read_csv(
            PROJECT_ROOT / "csv" / "CAREER.csv", encoding="utf-8-sig"
        )
        return HybridCareerRecommender(career_df, str(RESULTS_DIR)), career_df, None
    except Exception as exc:
        return None, None, str(exc)


def _align_proba(model, X, class_count):
    probs = np.asarray(model.predict_proba(X), dtype=float)
    aligned = np.zeros((len(X), class_count), dtype=float)
    classes = getattr(model, "classes_", np.arange(probs.shape[1]))
    for src_idx, class_id in enumerate(classes):
        class_id = int(class_id)
        if 0 <= class_id < class_count:
            aligned[:, class_id] = probs[:, src_idx]
    return aligned


def _quiz_validation_data(quiz_model, samples_per_role=160):
    try:
        from quiz_recommender import SKLEARN_AVAILABLE, train_test_split
    except Exception:
        SKLEARN_AVAILABLE = False
        train_test_split = None
    if not SKLEARN_AVAILABLE or train_test_split is None:
        return None, None, "scikit-learn is not available."
    try:
        X, y = quiz_model._build_synthetic_training_data(
            samples_per_role=samples_per_role
        )
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.22,
            random_state=42,
            stratify=y,
        )
        return X_test, y_test, None
    except Exception as exc:
        return None, None, str(exc)


def _calibrate_quiz_model_ensemble(quiz_model, X_test, y_test):
    if X_test is None or y_test is None:
        return [], 0, "quiz validation data is unavailable."
    if (
        getattr(quiz_model, "rf_model", None) is None
        or getattr(quiz_model, "gb_model", None) is None
    ):
        return [], 0, "quiz Random Forest / Gradient Boosting models are unavailable."
    try:
        class_count = len(quiz_model.job_names)
        rf_probs = _align_proba(quiz_model.rf_model, X_test, class_count)
        gb_probs = _align_proba(quiz_model.gb_model, X_test, class_count)
        rows, tested = _evaluate_probability_grid(
            probabilities=[rf_probs, gb_probs],
            y_true=y_test,
            names=["random_forest", "gradient_boosting"],
            step=0.01,
            top_n=30,
        )
        return rows, tested, None
    except Exception as exc:
        return [], 0, str(exc)


def _career_components(quiz_model, X_test, quiz_model_weights):
    role_matrix = np.asarray(quiz_model.prototype_matrix, dtype=float)
    user_matrix = np.asarray(X_test, dtype=float)
    class_count = len(quiz_model.job_names)
    cosine = _normalise_rows(user_matrix) @ _normalise_rows(role_matrix).T
    cosine = _clip01(cosine)
    if (
        getattr(quiz_model, "scaler", None) is not None
        and getattr(quiz_model, "pca", None) is not None
    ):
        latent_user = _normalise_rows(
            quiz_model.pca.transform(quiz_model.scaler.transform(user_matrix))
        )
        latent = latent_user @ np.asarray(quiz_model.prototype_latent, dtype=float).T
        latent = _clip01(latent)
    else:
        latent = np.zeros((len(user_matrix), class_count), dtype=float)
    if (
        getattr(quiz_model, "rf_model", None) is not None
        and getattr(quiz_model, "gb_model", None) is not None
    ):
        rf_probs = _align_proba(quiz_model.rf_model, user_matrix, class_count)
        gb_probs = _align_proba(quiz_model.gb_model, user_matrix, class_count)
        model_probability = (
            float(quiz_model_weights.get("random_forest", 0.0)) * rf_probs
            + float(quiz_model_weights.get("gradient_boosting", 0.0)) * gb_probs
        )
    else:
        model_probability = np.zeros((len(user_matrix), class_count), dtype=float)
    user_levels = np.where(
        user_matrix >= 0.75, 0.90, np.where(user_matrix >= 0.45, 0.60, 0.25)
    )
    rule = np.mean(user_levels[:, None, :] >= role_matrix[None, :, :], axis=2)
    readiness = 1.0 - np.mean(
        np.maximum(role_matrix[None, :, :] - user_matrix[:, None, :], 0.0), axis=2
    )
    reliability_values = _clip01(0.55 + (0.45 * user_matrix))
    role_weight_den = np.sum(role_matrix, axis=1, keepdims=True)
    role_weight_den[role_weight_den == 0] = 1.0
    role_weights = role_matrix / role_weight_den
    reliability = reliability_values @ role_weights.T
    components = np.stack(
        [
            _clip01(rule),
            _clip01(cosine),
            _clip01(latent),
            _clip01(model_probability),
            _clip01(readiness),
            _clip01(reliability),
        ],
        axis=2,
    )
    return components


def _calibrate_career_score(quiz_model, X_test, y_test, quiz_model_weights):
    if X_test is None or y_test is None:
        return [], 0, "quiz validation data is unavailable."
    try:
        max_samples = 1000
        if len(X_test) > max_samples:
            rng = np.random.default_rng(42)
            selected = rng.choice(len(X_test), size=max_samples, replace=False)
            X_test = X_test[selected]
            y_test = np.asarray(y_test)[selected]
        components = _career_components(quiz_model, X_test, quiz_model_weights)
        rows, tested = _evaluate_multiclass_score_grid(
            components=components,
            y_true=y_test,
            names=[
                "rule",
                "cosine",
                "latent",
                "model_probability",
                "readiness",
                "reliability",
            ],
            step=0.10,
            top_n=30,
        )
        return rows, tested, None
    except Exception as exc:
        return [], 0, str(exc)


def _difficulty_score(value):
    text = str(value).strip().lower()
    if text == "easy":
        return 0.25
    if text == "hard":
        return 1.0
    return 0.625


def _simulate_skill_rows(q_df, samples_per_category=90):
    rng = np.random.default_rng(42)
    rows = []
    bayes_rows = []
    if q_df.empty or "Category" not in q_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    for category, group in q_df.groupby("Category"):
        group = group.reset_index(drop=True)
        if group.empty:
            continue
        for _ in range(samples_per_category):
            ability = float(rng.beta(2.0, 2.0))
            attempts = int(rng.integers(5, min(13, max(6, len(group) + 1))))
            selected_indices = rng.choice(
                len(group), size=attempts, replace=len(group) < attempts
            )
            prior = 0.5
            observed = []
            time_factors = []
            confidence_alignments = []
            challenge_values = []
            skill_values = []
            for question_index in selected_indices:
                question = group.iloc[int(question_index)]
                option_weights = []
                for col in ["Weight1", "Weight2", "Weight3"]:
                    try:
                        option_weights.append(float(question.get(col, 0.0) or 0.0))
                    except Exception:
                        option_weights.append(0.0)
                max_weight = max(option_weights) if option_weights else 1.0
                if max_weight <= 0:
                    max_weight = 1.0
                normalized_options = (
                    np.asarray(option_weights, dtype=float) / max_weight
                )
                option_scores = np.exp(-np.abs(normalized_options - ability) / 0.20)
                option_scores = option_scores / np.sum(option_scores)
                chosen_idx = int(rng.choice(len(normalized_options), p=option_scores))
                normalized_score = float(normalized_options[chosen_idx])
                difficulty = _difficulty_score(question.get("Difficulty", "Medium"))
                time_factor = float(
                    np.clip(
                        1.05
                        - (difficulty * 0.25)
                        - (abs(normalized_score - ability) * 0.55)
                        + rng.normal(0, 0.08),
                        0.30,
                        1.0,
                    )
                )
                correct = normalized_score >= max(normalized_options)
                confidence = float(
                    np.clip(0.25 + (0.65 * ability) + rng.normal(0, 0.12), 0.0, 1.0)
                )
                confidence_alignment = 1.0 - abs(confidence - (1.0 if correct else 0.0))
                likelihood = float(np.clip(normalized_score * time_factor, 0.0, 1.0))
                bayes_rows.append(
                    {
                        "prior": prior,
                        "likelihood": likelihood,
                        "target": ability,
                    }
                )
                prior = (0.70 * prior) + (0.30 * likelihood)
                observed.append(normalized_score)
                time_factors.append(time_factor)
                confidence_alignments.append(confidence_alignment)
                challenge_values.append(difficulty * normalized_score)
                skill_values.append(prior)
            observed_accuracy = float(np.mean(observed)) if observed else 0.0
            time_efficiency = float(np.mean(time_factors)) if time_factors else 0.5
            confidence_alignment = (
                float(np.mean(confidence_alignments)) if confidence_alignments else 0.5
            )
            bayesian_skill = float(skill_values[-1]) if skill_values else 0.5
            consistency = (
                float(np.clip(1.0 - min(np.std(observed) / 0.35, 1.0), 0.0, 1.0))
                if len(observed) > 1
                else 0.5
            )
            deltas = (
                np.diff(skill_values) if len(skill_values) > 1 else np.asarray([0.0])
            )
            momentum = float(np.clip(0.5 + (float(np.mean(deltas)) * 3.0), 0.0, 1.0))
            challenge_index = (
                float(np.mean(challenge_values)) if challenge_values else 0.0
            )
            uncertainty = (
                (1.0 - bayesian_skill) * 0.45
                + (1.0 - confidence_alignment) * 0.35
                + (1.0 - time_efficiency) * 0.20
            )
            target_uncertainty = float(
                np.clip(
                    (abs(observed_accuracy - ability) * 0.70)
                    + ((1.0 - confidence_alignment) * 0.20)
                    + ((1.0 - time_efficiency) * 0.10),
                    0.0,
                    1.0,
                )
            )
            target_reliability = float(
                np.clip(1.0 - abs(bayesian_skill - ability), 0.0, 1.0)
            )
            rows.append(
                {
                    "category": category,
                    "target": ability,
                    "bayesian_skill": bayesian_skill,
                    "observed_accuracy": observed_accuracy,
                    "confidence_alignment": confidence_alignment,
                    "time_efficiency": time_efficiency,
                    "consistency": consistency,
                    "momentum": momentum,
                    "challenge_index": challenge_index,
                    "uncertainty": uncertainty,
                    "skill_gap": 1.0 - bayesian_skill,
                    "confidence_gap": 1.0 - confidence_alignment,
                    "time_gap": 1.0 - time_efficiency,
                    "certainty": 1.0 - uncertainty,
                    "target_uncertainty": target_uncertainty,
                    "target_reliability": target_reliability,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bayes_rows)


def _calibrate_skill_assessment(skill_df):
    if skill_df.empty:
        return [], 0, "quiz question data is unavailable."
    names = [
        "bayesian_skill",
        "observed_accuracy",
        "confidence_alignment",
        "time_efficiency",
        "consistency",
        "momentum",
        "challenge_index",
    ]
    components = skill_df[names].to_numpy(dtype=float)
    target = skill_df["target"].to_numpy(dtype=float)
    penalty = 1.0 - (0.12 * skill_df["uncertainty"].to_numpy(dtype=float))
    rows = []
    tested = 0
    for weights in _simplex_grid(names, 0.10):
        pred = (components @ _weights_to_array(weights, names)) * penalty
        metrics = _regression_metrics(target, _clip01(pred))
        rows.append(
            {
                "weights": {
                    key: round(float(value), 6) for key, value in weights.items()
                },
                "metrics": metrics,
                "sort_key": (metrics["rmse"], metrics["mae"], -metrics["r2"]),
            }
        )
        tested += 1
    rows.sort(key=lambda item: item["sort_key"])
    return rows[:30], tested, None


def _calibrate_skill_uncertainty(skill_df):
    if skill_df.empty:
        return [], 0, "quiz question data is unavailable."
    names = ["skill_gap", "confidence_gap", "time_gap"]
    rows, tested = _evaluate_regression_grid(
        components=skill_df[names].to_numpy(dtype=float),
        y_true=skill_df["target_uncertainty"].to_numpy(dtype=float),
        names=names,
        step=0.05,
        top_n=30,
    )
    return rows, tested, None


def _calibrate_skill_reliability(skill_df):
    if skill_df.empty:
        return [], 0, "quiz question data is unavailable."
    names = ["confidence_alignment", "consistency", "certainty"]
    rows, tested = _evaluate_regression_grid(
        components=skill_df[names].to_numpy(dtype=float),
        y_true=skill_df["target_reliability"].to_numpy(dtype=float),
        names=names,
        step=0.05,
        top_n=30,
    )
    return rows, tested, None


def _calibrate_bayesian_update(bayes_df):
    if bayes_df.empty:
        return [], 0, "quiz question data is unavailable."
    names = ["prior", "likelihood"]
    rows, tested = _evaluate_regression_grid(
        components=bayes_df[names].to_numpy(dtype=float),
        y_true=bayes_df["target"].to_numpy(dtype=float),
        names=names,
        step=0.01,
        top_n=30,
    )
    return rows, tested, None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    sections = [
        "Weighted Sum Calibration Report",
        f"Generated at: {generated_at}",
        f"Project root: {PROJECT_ROOT}",
        "",
        "The JSON file is loaded automatically by app.py and backend modules.",
        "If scikit-learn is missing, ML ensemble sections are skipped instead of writing fake weights.",
    ]
    payload = {
        "generated_at": generated_at,
        "project_root": str(PROJECT_ROOT),
        "defaults": DEFAULT_WEIGHT_CONFIG,
        "weights": {},
    }
    recommender, college_error = _load_college_recommender()
    training_df = (
        _college_training_frame(recommender)
        if recommender is not None
        else pd.DataFrame()
    )
    rows, tested, skipped = (
        ([], 0, college_error)
        if recommender is None
        else _calibrate_college_model_ensemble(recommender, training_df)
    )
    sections.extend(
        _format_section(
            "1. Hybrid Recommendation Score: DT/RF/GB rank ensemble",
            rows,
            tested,
            "lowest validation RMSE, then MAE",
            skipped,
        )
    )
    best = _best_payload(
        rows, "lowest validation RMSE, then MAE", "rank model validation split", tested
    )
    if best:
        payload["weights"]["college_model_ensemble"] = best
    rows, tested, skipped = _calibrate_college_quality(training_df)
    sections.extend(
        _format_section(
            "2. College Aspect / Quality Score",
            rows,
            tested,
            "lowest RMSE against historical rank desirability",
            skipped,
        )
    )
    best = _best_payload(
        rows,
        "lowest RMSE against historical rank desirability",
        "college rank and quality CSVs",
        tested,
    )
    if best:
        payload["weights"]["college_quality_score"] = best
        college_quality_weights = best["weights"]
    else:
        college_quality_weights = DEFAULT_WEIGHT_CONFIG["college_quality_score"]
    rows, tested, skipped = (
        ([], 0, college_error)
        if recommender is None
        else _calibrate_college_recommendation(
            recommender,
            training_df,
            college_quality_weights,
        )
    )
    sections.extend(
        _format_section(
            "3. Final College Recommendation Score",
            rows,
            tested,
            "lowest RMSE against historical rank desirability",
            skipped,
        )
    )
    best = _best_payload(
        rows,
        "lowest RMSE against historical rank desirability",
        "college rank, quality, and association-rule proxies",
        tested,
    )
    if best:
        payload["weights"]["college_recommendation_score"] = best
    quiz_model, _, quiz_error = _load_quiz_model()
    X_test, y_test, quiz_validation_error = (
        (None, None, quiz_error)
        if quiz_model is None
        else _quiz_validation_data(quiz_model)
    )
    rows, tested, skipped = (
        ([], 0, quiz_validation_error)
        if quiz_validation_error
        else _calibrate_quiz_model_ensemble(quiz_model, X_test, y_test)
    )
    sections.extend(
        _format_section(
            "4. Quiz Career Model Ensemble: Random Forest / Gradient Boosting",
            rows,
            tested,
            "highest accuracy, then lowest log loss",
            skipped,
        )
    )
    best = _best_payload(
        rows,
        "highest accuracy, then lowest log loss",
        "synthetic career-profile validation split",
        tested,
    )
    if best:
        payload["weights"]["quiz_model_ensemble"] = best
        quiz_ensemble_weights = best["weights"]
    else:
        quiz_ensemble_weights = DEFAULT_WEIGHT_CONFIG["quiz_model_ensemble"]
    rows, tested, skipped = (
        ([], 0, quiz_validation_error)
        if quiz_validation_error
        else _calibrate_career_score(
            quiz_model,
            X_test,
            y_test,
            quiz_ensemble_weights,
        )
    )
    sections.extend(
        _format_section(
            "5. Career Recommendation Score",
            rows,
            tested,
            "highest top-1 accuracy, then lowest log loss",
            skipped,
        )
    )
    best = _best_payload(
        rows,
        "highest top-1 accuracy, then lowest log loss",
        "synthetic career-profile validation split",
        tested,
    )
    if best:
        payload["weights"]["career_recommendation_score"] = best
    try:
        q_df = pd.read_csv(PROJECT_ROOT / "csv" / "QUESTIONS.csv", encoding="latin1")
    except Exception:
        q_df = pd.DataFrame()
    skill_df, bayes_df = _simulate_skill_rows(q_df)
    rows, tested, skipped = _calibrate_skill_assessment(skill_df)
    sections.extend(
        _format_section(
            "6. Skill Assessment Total Score",
            rows,
            tested,
            "lowest RMSE against simulated skill ability",
            skipped,
        )
    )
    best = _best_payload(
        rows,
        "lowest RMSE against simulated skill ability",
        "QUESTIONS.csv synthetic quiz sessions",
        tested,
    )
    if best:
        payload["weights"]["skill_assessment_score"] = best
    rows, tested, skipped = _calibrate_skill_uncertainty(skill_df)
    sections.extend(
        _format_section(
            "7. Skill Uncertainty Score",
            rows,
            tested,
            "lowest RMSE against simulated uncertainty",
            skipped,
        )
    )
    best = _best_payload(
        rows,
        "lowest RMSE against simulated uncertainty",
        "QUESTIONS.csv synthetic quiz sessions",
        tested,
    )
    if best:
        payload["weights"]["skill_uncertainty_score"] = best
    rows, tested, skipped = _calibrate_skill_reliability(skill_df)
    sections.extend(
        _format_section(
            "8. Skill Reliability / Confidence Score",
            rows,
            tested,
            "lowest RMSE against simulated reliability",
            skipped,
        )
    )
    best = _best_payload(
        rows,
        "lowest RMSE against simulated reliability",
        "QUESTIONS.csv synthetic quiz sessions",
        tested,
    )
    if best:
        payload["weights"]["skill_reliability_score"] = best
    rows, tested, skipped = _calibrate_bayesian_update(bayes_df)
    sections.extend(
        _format_section(
            "9. Bayesian Skill Update Blend",
            rows,
            tested,
            "lowest RMSE against simulated skill ability",
            skipped,
        )
    )
    best = _best_payload(
        rows,
        "lowest RMSE against simulated skill ability",
        "QUESTIONS.csv synthetic quiz answer steps",
        tested,
    )
    if best:
        payload["weights"]["skill_bayesian_update"] = best
    sections.append("")
    sections.append("Untuned standard formulas")
    sections.append("------------------------")
    sections.append(
        "Cosine similarity, TF-IDF, accuracy, precision, recall, F1, and normalization are mathematical definitions, not arbitrary project weight vectors."
    )
    sections.append(
        "Review keyword weights are lexicon scores rather than normalized weighted-sum coefficients, so they are not changed by this calibration file."
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(sections).rstrip() + "\n")
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(_json_safe(payload), file_obj, indent=2)
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved runtime weights: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
