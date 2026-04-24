let questions = JSON.parse(localStorage.getItem("questions")) || [];
let answers = [];

let totalAnswered = 0;
let streak = 0;

let startTime = 0;
let timerInterval = null;
let isLoadingNext = false;

let skillState = {
    AI_Int: 0,
    Coding_Int: 0,
    Design_Int: 0,
    Logical_Reasoning: 0,
    Analytical_Reasoning: 0,
    Verbal_Reasoning: 0,
    Math_Reasoning: 0,
    Coding_Skill: 0,
    Web_Arch: 0,
    Data_Mining: 0,
    Crypto_Focus: 0,
    Cloud_Ops: 0,
    Low_Level: 0,
    DB_Design: 0,
    System_Opt: 0,
    Risk_Eval: 0,
    User_Empathy: 0
};

let skillCounts = Object.fromEntries(
    Object.keys(skillState).map(key => [key, 0])
);

let confidence = 50;

window.onload = () => {
    localStorage.removeItem("questions");
    startQuiz();
};

function updateConfidence(val) {
    confidence = Number(val);
    const el = document.getElementById("confValue");
    if (el) el.innerText = confidence + "%";
}

function startQuiz() {
    fetch("/start", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            if (!Array.isArray(data) || data.length === 0) {
                document.getElementById("quiz").innerHTML = "No questions";
                return;
            }

            questions = data;
            localStorage.setItem("questions", JSON.stringify(data));
            showQuestion();
        })
        .catch(err => {
            console.error("Start error:", err);
        });
}

function showQuestion() {
    const q = questions[0];
    if (!q) return;

    const div = document.getElementById("quiz");

    div.innerHTML = `
        <h3>Question ${totalAnswered + 1}</h3>
        <p><b>${q["Question Text"]}</b></p>

        ${createOption(q, "Option1", "Weight1")}
        ${createOption(q, "Option2", "Weight2")}
        ${createOption(q, "Option3", "Weight3")}
    `;

    const cur = document.getElementById("currentCategory");
    const nxt = document.getElementById("nextCategory");

    if (cur) cur.innerText = "Current: " + (q.Category || "-");
    if (nxt) nxt.innerText = "Next: " + (questions[1]?.Category || "End");

    startTime = Date.now();
    clearInterval(timerInterval);

    timerInterval = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        const timer = document.getElementById("timer");
        if (timer) timer.innerText = "Time: " + elapsed.toFixed(1) + "s";
    }, 100);

    updateDifficultyUI(q);
    renderSkills();
}

function createOption(q, optKey, weightKey) {
    const option = q?.[optKey];
    const weight = Number(q?.[weightKey] || 0);

    if (!option || option === "undefined") return "";

    return `
        <button class="opt"
            onclick="selectOption(this, ${weight}, ${q.Weight1}, ${q.Weight2}, ${q.Weight3})">
            ${option}
        </button>
    `;
}

function selectOption(btn, weight, w1, w2, w3) {
    if (isLoadingNext) return;
    isLoadingNext = true;

    clearInterval(timerInterval);

    const timeTaken = (Date.now() - startTime) / 1000;
    document.querySelectorAll(".opt").forEach(button => {
        button.disabled = true;
    });

    btn.style.background = "#00c853";
    btn.style.color = "white";

    const allWeights = [w1, w2, w3];
    const maxWeight = Math.max(...allWeights);

    if (weight >= maxWeight) streak++;
    else streak = 0;

    const streakLabel = document.getElementById("streak");
    if (streakLabel) streakLabel.innerText = "Streak: " + streak;

    const currentQuestion = questions[0] || {};

    answers.push({
        question_id: currentQuestion["Question ID"] || "",
        question_text: currentQuestion["Question Text"] || "",
        category: currentQuestion.Category || "",
        difficulty: currentQuestion.Difficulty || "Medium",
        selected_option: btn.innerText.trim(),
        weight: weight,
        all_weights: allWeights,
        time: timeTaken,
        confidence: confidence,
        streak: streak
    });

    totalAnswered++;
    updateProgressBar();
    setTimeout(goNext, 200);
}

function goNext() {
    fetch("/next", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([answers[answers.length - 1]])
    })
        .then(res => res.json())
        .then(data => {
            isLoadingNext = false;

            if (!Array.isArray(data) || data.length === 0) {
                document.getElementById("quiz").innerHTML =
                    "<h3>Quiz Completed</h3>";
                return;
            }

            questions = data;
            localStorage.setItem("questions", JSON.stringify(data));

            showQuestion();
            updateSkill();
            showAIHint();
        })
        .catch(err => {
            console.error("Next error:", err);
            isLoadingNext = false;
        });
}

function updateSkill() {
    const ans = answers[answers.length - 1];
    if (!ans) return;

    const speed = ans.time < 2 ? 1 : ans.time < 5 ? 0.75 : 0.5;
    const maxWeight = Math.max(...(ans.all_weights || [ans.weight || 0, 1]));
    const normalized = maxWeight > 0 ? (ans.weight / maxWeight) : 0;
    const score = normalized * speed * (ans.confidence / 100);
    const cat = ans.category;

    if (skillState.hasOwnProperty(cat)) {
        skillCounts[cat] = (skillCounts[cat] || 0) + 1;
        skillState[cat] =
            ((skillState[cat] * (skillCounts[cat] - 1)) + score) /
            skillCounts[cat];
    }

    renderSkills();
}

function renderSkills() {
    const box = document.getElementById("skillsBox");
    if (!box) return;

    box.innerHTML = "";

    Object.keys(skillState).forEach(key => {
        box.innerHTML += `<div>${key}: ${skillState[key].toFixed(2)}</div>`;
    });
}

function showAIHint() {
    const ans = answers[answers.length - 1];
    if (!ans) return;

    let hint =
        ans.time < 2 ? "Fast" :
        ans.time > 5 ? "Slow" :
        "Normal";

    if (ans.confidence > 80 && ans.weight < Math.max(...ans.all_weights)) {
        hint += " | Overconfidence detected";
    }

    const hintBox = document.getElementById("aiHint");
    if (hintBox) hintBox.innerText = hint;
}

function updateDifficultyUI(q) {
    const el = document.getElementById("difficulty");
    if (el) el.innerText = "Difficulty: " + (q.Difficulty || "Medium");
}

function updateProgressBar() {
    const percent = (totalAnswered / 17) * 100;
    const bar = document.getElementById("progressBar");
    if (bar) bar.style.width = Math.min(percent, 100) + "%";
}

function stopQuiz() {
    if (totalAnswered < 17) {
        alert("Answer at least 17 questions!");
        return;
    }

    localStorage.clear();
    window.location = "/quiz-result";
}
