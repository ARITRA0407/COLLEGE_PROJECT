import json
import os
from copy import deepcopy


CALIBRATED_WEIGHTS_FILENAME = "calibrated_weights.json"


DEFAULT_WEIGHT_CONFIG = {
    "college_model_ensemble": {
        "decision_tree": 1.0 / 3.0,
        "random_forest": 1.0 / 3.0,
        "gradient_boosting": 1.0 / 3.0,
    },
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
    "skill_assessment_score": {
        "bayesian_skill": 0.34,
        "observed_accuracy": 0.22,
        "confidence_alignment": 0.14,
        "time_efficiency": 0.10,
        "consistency": 0.08,
        "momentum": 0.06,
        "challenge_index": 0.06,
    },
    "skill_uncertainty_score": {
        "skill_gap": 0.45,
        "confidence_gap": 0.35,
        "time_gap": 0.20,
    },
    "skill_reliability_score": {
        "confidence_alignment": 0.35,
        "consistency": 0.35,
        "certainty": 0.30,
    },
    "skill_bayesian_update": {
        "prior": 0.70,
        "likelihood": 0.30,
    },
}


_PAYLOAD_CACHE = {}


def _project_root(data_root_dir=None):
    if data_root_dir:
        return os.path.abspath(data_root_dir)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def calibrated_weights_path(data_root_dir=None):
    return os.path.join(
        _project_root(data_root_dir),
        "results",
        CALIBRATED_WEIGHTS_FILENAME,
    )


def calibrated_weights_mtime(data_root_dir=None):
    path = calibrated_weights_path(data_root_dir)
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def _load_payload(data_root_dir=None):
    path = calibrated_weights_path(data_root_dir)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _PAYLOAD_CACHE[path] = (0, {})
        return {}

    cached = _PAYLOAD_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except Exception:
        payload = {}

    _PAYLOAD_CACHE[path] = (mtime, payload)
    return payload


def _normalise(weights):
    clean = {}
    for key, value in weights.items():
        try:
            numeric = float(value)
        except Exception:
            continue
        if numeric < 0:
            numeric = 0.0
        clean[key] = numeric

    total = sum(clean.values())
    if total <= 0:
        return clean

    return {key: value / total for key, value in clean.items()}


def default_weight_vector(weight_key, names=None):
    weights = deepcopy(DEFAULT_WEIGHT_CONFIG.get(weight_key, {}))
    if names is not None:
        names = list(names)
        weights = {name: float(weights.get(name, 0.0)) for name in names}
        if sum(weights.values()) <= 0 and names:
            weights = {name: 1.0 / len(names) for name in names}
    return _normalise(weights)


def get_saved_weight_vector(weight_key, names=None, data_root_dir=None):
    payload = _load_payload(data_root_dir)
    section = payload.get("weights", {}).get(weight_key)
    if not isinstance(section, dict):
        return None

    raw_weights = section.get("weights")
    if not isinstance(raw_weights, dict):
        return None

    if names is not None:
        names = list(names)
        defaults = default_weight_vector(weight_key, names)
        weights = {
            name: float(raw_weights.get(name, defaults.get(name, 0.0)))
            for name in names
        }
    else:
        weights = dict(raw_weights)

    weights = _normalise(weights)
    if not weights or sum(weights.values()) <= 0:
        return None
    return weights


def get_weight_vector(weight_key, names=None, data_root_dir=None):
    saved = get_saved_weight_vector(weight_key, names=names, data_root_dir=data_root_dir)
    if saved is not None:
        return saved
    return default_weight_vector(weight_key, names=names)


def weighted_sum(weight_key, values, names=None, data_root_dir=None):
    if names is None:
        names = list(values.keys())
    weights = get_weight_vector(weight_key, names=names, data_root_dir=data_root_dir)
    total = 0.0
    for name in names:
        try:
            value = float(values.get(name, 0.0))
        except Exception:
            value = 0.0
        total += float(weights.get(name, 0.0)) * value
    return total
