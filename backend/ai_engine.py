# ==============================
# 🧠 AI ENGINE - CORE MODULE (FIXED)
# ==============================

import math
import random


# =========================================================
# 1. 🧠 BAYESIAN SKILL UPDATE
# =========================================================
def bayesian_update(prior, likelihood, alpha=0.7):
    if prior is None:
        prior = 0.5
    prior = max(0, min(1, prior))
    likelihood = max(0, min(1, likelihood))
    return round((alpha * prior) + ((1 - alpha) * likelihood), 4)


# =========================================================
# 2. 🎯 DIFFICULTY DECISION ENGINE (STABLE)
# =========================================================
def decide_difficulty(skill_score, streak, last_difficulty="Medium"):

    skill_score = max(0, min(1, skill_score))

    if skill_score >= 0.8 and streak >= 3:
        return "Hard"

    if skill_score >= 0.6:
        return "Medium"

    if skill_score < 0.4:
        return "Easy"

    return last_difficulty or "Medium"


# =========================================================
# 3. 🧩 EXPLAINABILITY ENGINE
# =========================================================
def generate_explanation(category, old_diff, new_diff, score, time_taken):

    if new_diff == "Hard" and old_diff != "Hard":
        return f"🔥 High performance detected in {category}. Increasing difficulty."

    if new_diff == "Easy":
        return f"📉 Low performance in {category}. Reducing difficulty."

    if time_taken > 6:
        return f"⏱ Slow response in {category}. Adjusting difficulty."

    if score >= 0.75:
        return f"✅ Strong accuracy in {category}. Maintaining challenge level."

    return f"🔄 Adaptive adjustment applied in {category}."


# =========================================================
# 4. 🔥 STREAK SYSTEM
# =========================================================
def update_streak(current_streak, correct):
    return current_streak + 1 if correct else 0


# =========================================================
# 5. 🧠 COGNITIVE LOAD DETECTION (FIXED)
# =========================================================
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


# =========================================================
# 6. 🧠 SKILL ESTIMATOR (NORMALIZED)
# =========================================================
def estimate_skill(current_skill, score, time_factor=1.0):

    current_skill = max(0, min(1, current_skill))
    performance = max(0, min(1, score * time_factor))

    return bayesian_update(current_skill, performance)


# =========================================================
# 7. ⚙️ TIME FACTOR ENGINE
# =========================================================
def compute_time_factor(time_taken):

    time_taken = float(time_taken)

    if time_taken <= 2:
        return 1.0
    elif time_taken <= 4:
        return 0.8
    elif time_taken <= 6:
        return 0.6
    elif time_taken <= 10:
        return 0.4
    else:
        return 0.3


# =========================================================
# 8. 🧠 MASTER AI PROCESSOR
# =========================================================
def process_answer(state, category, score, time_taken, correct, last_diff):

    state.setdefault("skills", {})
    state.setdefault("streak", {})
    state.setdefault("history", {})

    state["skills"].setdefault(category, 0.5)
    state["streak"].setdefault(category, 0)

    # ---------------- STREAK ----------------
    state["streak"][category] = update_streak(
        state["streak"][category],
        correct
    )

    # ---------------- TIME FACTOR ----------------
    time_factor = compute_time_factor(time_taken)

    # ---------------- SKILL UPDATE ----------------
    old_skill = state["skills"][category]

    new_skill = estimate_skill(old_skill, score, time_factor)

    state["skills"][category] = new_skill

    # ---------------- DIFFICULTY ----------------
    next_diff = decide_difficulty(
        new_skill,
        state["streak"][category],
        last_diff
    )

    # ---------------- COGNITIVE LOAD ----------------
    load = cognitive_load(time_taken, score, state["streak"][category])

    # ---------------- EXPLANATION ----------------
    explanation = generate_explanation(
        category,
        last_diff,
        next_diff,
        score,
        time_taken
    )

    # ---------------- HISTORY ----------------
    state["history"].setdefault(category, [])
    state["history"][category].append({
        "score": round(score, 3),
        "skill": round(new_skill, 3),
        "time": float(time_taken),
        "difficulty": next_diff
    })

    # ---------------- FINAL OUTPUT ----------------
    return {
        "updated_state": state,
        "next_difficulty": next_diff,
        "explanation": explanation,
        "cognitive_load": load,
        "skill": round(new_skill, 3),
        "streak": state["streak"][category],
        "time_factor": time_factor
    }


# =========================================================
# 9. 🧩 DASHBOARD SUMMARY (FIXED SAFE VERSION)
# =========================================================
def get_skill_summary(state):

    skills = state.get("skills", {})

    if not skills:
        return {
            "categories": {},
            "avg_skill": 0
        }

    avg = sum(skills.values()) / max(len(skills), 1)

    return {
        "categories": {k: round(v, 3) for k, v in skills.items()},
        "avg_skill": round(avg, 3)
    }