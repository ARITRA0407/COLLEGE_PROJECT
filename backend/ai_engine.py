import math

try:
    from calibrated_weights import get_weight_vector
except Exception:

    def get_weight_vector(weight_key, names=None, data_root_dir=None):
        defaults = {
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
        weights = defaults.get(weight_key, {})
        if names is not None:
            names = list(names)
            weights = {name: float(weights.get(name, 0.0)) for name in names}
            total = sum(weights.values()) or 1.0
            return {name: value / total for name, value in weights.items()}
        return weights


def _clamp(value, low=0.0, high=1.0):
    try:
        value = float(value)
    except Exception:
        value = low
    return max(low, min(high, value))


def bayesian_update(prior, likelihood, alpha=None):
    if prior is None:
        prior = 0.5
    prior = _clamp(prior)
    likelihood = _clamp(likelihood)
    if alpha is None:
        weights = get_weight_vector(
            "skill_bayesian_update",
            names=["prior", "likelihood"],
        )
        prior_weight = float(weights.get("prior", 0.70))
        likelihood_weight = float(weights.get("likelihood", 0.30))
    else:
        prior_weight = _clamp(alpha)
        likelihood_weight = 1.0 - prior_weight
    return round((prior_weight * prior) + (likelihood_weight * likelihood), 4)


def decide_difficulty(skill_score, streak, last_difficulty="Medium"):
    skill_score = _clamp(skill_score)
    if skill_score >= 0.8 and streak >= 3:
        return "Hard"
    if skill_score >= 0.6:
        return "Medium"
    if skill_score < 0.4:
        return "Easy"
    return last_difficulty or "Medium"


def generate_explanation(category, old_diff, new_diff, score, time_taken):
    if new_diff == "Hard" and old_diff != "Hard":
        return f"High performance detected in {category}. Increasing difficulty."
    if new_diff == "Easy":
        return f"Lower performance in {category}. Reducing difficulty."
    if time_taken > 6:
        return f"Slow response in {category}. Adjusting difficulty."
    if score >= 0.75:
        return f"Strong accuracy in {category}. Maintaining challenge level."
    return f"Adaptive adjustment applied in {category}."


def update_streak(current_streak, correct):
    return current_streak + 1 if correct else 0


def compute_confidence_alignment(confidence, correct):
    confidence = _clamp(confidence)
    target = 1.0 if correct else 0.0
    return round(1.0 - abs(confidence - target), 4)


def cognitive_load(time_taken, accuracy, streak):
    time_taken = float(time_taken)
    if time_taken > 6 and accuracy < 0.5:
        return "High"
    if time_taken > 4 and accuracy < 0.6:
        return "High"
    if time_taken > 4 or accuracy < 0.65:
        return "Medium"
    if streak >= 4 and accuracy >= 0.8:
        return "Low"
    return "Medium"


def compute_consistency(history, window=5):
    recent = history[-window:] if history else []
    if len(recent) <= 1:
        return 0.5
    scores = [float(item.get("score", 0.0)) for item in recent]
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_dev = math.sqrt(max(variance, 0.0))
    return round(_clamp(1.0 - min(std_dev / 0.35, 1.0)), 4)


def compute_momentum(history, window=4):
    recent = history[-window:] if history else []
    if len(recent) <= 1:
        return 0.5
    skills = [float(item.get("skill", 0.5)) for item in recent]
    deltas = [skills[i] - skills[i - 1] for i in range(1, len(skills))]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return round(_clamp(0.5 + (avg_delta * 3.0)), 4)


def compute_uncertainty(skill_score, time_factor, confidence_alignment):
    weights = get_weight_vector(
        "skill_uncertainty_score",
        names=["skill_gap", "confidence_gap", "time_gap"],
    )
    uncertainty = (
        (1.0 - _clamp(skill_score)) * float(weights.get("skill_gap", 0.0))
        + (1.0 - _clamp(confidence_alignment))
        * float(weights.get("confidence_gap", 0.0))
        + (1.0 - _clamp(time_factor)) * float(weights.get("time_gap", 0.0))
    )
    return round(_clamp(uncertainty), 4)


def compute_reliability(confidence_alignment, consistency, uncertainty):
    weights = get_weight_vector(
        "skill_reliability_score",
        names=["confidence_alignment", "consistency", "certainty"],
    )
    reliability = (
        (_clamp(confidence_alignment) * float(weights.get("confidence_alignment", 0.0)))
        + (_clamp(consistency) * float(weights.get("consistency", 0.0)))
        + ((1.0 - _clamp(uncertainty)) * float(weights.get("certainty", 0.0)))
    )
    return round(_clamp(reliability), 4)


def estimate_skill(current_skill, score, time_factor=1.0):
    current_skill = _clamp(current_skill)
    performance = _clamp(score * time_factor)
    return bayesian_update(current_skill, performance)


def compute_time_factor(time_taken):
    time_taken = float(time_taken)
    if time_taken <= 2:
        return 1.0
    if time_taken <= 4:
        return 0.8
    if time_taken <= 6:
        return 0.6
    if time_taken <= 10:
        return 0.4
    return 0.3


def process_answer(
    state, category, score, time_taken, correct, last_diff, confidence=0.5
):
    state.setdefault("skills", {})
    state.setdefault("streak", {})
    state.setdefault("history", {})
    state.setdefault("telemetry", {})
    state["skills"].setdefault(category, 0.5)
    state["streak"].setdefault(category, 0)
    state["streak"][category] = update_streak(state["streak"][category], correct)
    time_factor = compute_time_factor(time_taken)
    old_skill = state["skills"][category]
    new_skill = estimate_skill(old_skill, score, time_factor)
    state["skills"][category] = new_skill
    next_diff = decide_difficulty(new_skill, state["streak"][category], last_diff)
    load = cognitive_load(time_taken, score, state["streak"][category])
    explanation = generate_explanation(
        category, last_diff, next_diff, score, time_taken
    )
    state["history"].setdefault(category, [])
    state["history"][category].append(
        {
            "score": round(_clamp(score), 3),
            "skill": round(new_skill, 3),
            "time": float(time_taken),
            "difficulty": next_diff,
            "correct": bool(correct),
            "confidence": round(_clamp(confidence), 3),
            "time_factor": round(time_factor, 3),
        }
    )
    recent_history = state["history"][category]
    confidence_alignment = compute_confidence_alignment(confidence, correct)
    consistency = compute_consistency(recent_history)
    momentum = compute_momentum(recent_history)
    uncertainty = compute_uncertainty(new_skill, time_factor, confidence_alignment)
    reliability = compute_reliability(confidence_alignment, consistency, uncertainty)
    state["history"][category][-1].update(
        {
            "confidence_alignment": confidence_alignment,
            "consistency": consistency,
            "momentum": momentum,
            "uncertainty": uncertainty,
            "reliability": reliability,
        }
    )
    state["telemetry"][category] = {
        "confidence_alignment": confidence_alignment,
        "consistency": consistency,
        "momentum": momentum,
        "uncertainty": uncertainty,
        "reliability": reliability,
    }
    return {
        "updated_state": state,
        "next_difficulty": next_diff,
        "explanation": explanation,
        "cognitive_load": load,
        "skill": round(new_skill, 3),
        "streak": state["streak"][category],
        "time_factor": time_factor,
        "confidence_alignment": confidence_alignment,
        "consistency": consistency,
        "momentum": momentum,
        "uncertainty": uncertainty,
        "reliability": reliability,
    }


def get_skill_summary(state):
    skills = state.get("skills", {})
    telemetry = state.get("telemetry", {})
    if not skills:
        return {
            "categories": {},
            "avg_skill": 0,
            "avg_reliability": 0,
            "avg_uncertainty": 0,
        }
    avg_skill = sum(skills.values()) / max(len(skills), 1)
    reliability_values = [
        float(item.get("reliability", 0.0)) for item in telemetry.values()
    ]
    uncertainty_values = [
        float(item.get("uncertainty", 0.0)) for item in telemetry.values()
    ]
    return {
        "categories": {k: round(v, 3) for k, v in skills.items()},
        "avg_skill": round(avg_skill, 3),
        "avg_reliability": (
            round(sum(reliability_values) / len(reliability_values), 3)
            if reliability_values
            else 0
        ),
        "avg_uncertainty": (
            round(sum(uncertainty_values) / len(uncertainty_values), 3)
            if uncertainty_values
            else 0
        ),
    }
