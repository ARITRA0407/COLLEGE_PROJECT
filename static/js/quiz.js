let questions = JSON.parse(localStorage.getItem("questions")) || [];
let answers = [];

let totalAnswered = 0;
let streak = 0;

let startTime = 0;
let timerInterval = null;

// 🔥 prevent double click / stuck state
let isLoadingNext = false;

// ===============================
// ✅ FULL 17-CATEGORY SKILL STATE
// ===============================
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

let confidence = 50;

// ---------------- LOAD ---------------- //
window.onload = () => {
    localStorage.removeItem("questions");
    startQuiz();
};

// ---------------- CONFIDENCE ---------------- //
function updateConfidence(val) {
    confidence = Number(val);
    const el = document.getElementById("confValue");
    if (el) el.innerText = confidence + "%";
}

// ---------------- START ---------------- //
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

// ---------------- SHOW QUESTION ---------------- //
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

    // safe UI updates
    const cur = document.getElementById("currentCategory");
    const nxt = document.getElementById("nextCategory");

    if (cur) cur.innerText = "Current: " + (q.Category || "-");
    if (nxt) nxt.innerText = "Next: " + (questions[1]?.Category || "End");

    startTime = Date.now();
    clearInterval(timerInterval);

    timerInterval = setInterval(() => {
        let t = (Date.now() - startTime) / 1000;
        const timer = document.getElementById("timer");
        if (timer) timer.innerText = "⏱ Time: " + t.toFixed(1) + "s";
    }, 100);

    updateDifficultyUI(q);
    renderSkills();
}

// ---------------- OPTIONS ---------------- //
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

// ---------------- SELECT ---------------- //
function selectOption(btn, weight, w1, w2, w3) {

    if (isLoadingNext) return; // 🔥 block spam clicks
    isLoadingNext = true;

    clearInterval(timerInterval);

    const timeTaken = (Date.now() - startTime) / 1000;

    document.querySelectorAll(".opt").forEach(b => b.disabled = true);

    btn.style.background = "#00c853";
    btn.style.color = "white";

    let all_w = [w1, w2, w3];
    let maxW = Math.max(...all_w);

    if (weight >= maxW) streak++;
    else streak = 0;

    const st = document.getElementById("streak");
    if (st) st.innerText = "🔥 Streak: " + streak;

    answers.push({
        category: questions[0]?.Category || "",
        weight: weight,
        all_weights: all_w,
        time: timeTaken,
        confidence: confidence,
        streak: streak
    });

    totalAnswered++;
    updateProgressBar();

    // 🔥 ensure transition delay + safety
    setTimeout(goNext, 200);
}

// ---------------- NEXT ---------------- //
function goNext() {

    fetch("/next", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([answers[answers.length - 1]])
    })
    .then(res => res.json())
    .then(data => {

        isLoadingNext = false; // 🔥 unlock

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
        isLoadingNext = false; // 🔥 always unlock even on error
    });
}

// ---------------- SKILL UPDATE ---------------- //
function updateSkill() {

    const ans = answers[answers.length - 1];
    if (!ans) return;

    const speed = ans.time < 2 ? 1 : ans.time < 5 ? 0.7 : 0.4;
    const score = ans.weight * speed * (confidence / 100);

    const cat = ans.category;

    if (skillState.hasOwnProperty(cat)) {
        skillState[cat] += score;
    }

    renderSkills();
}

// ---------------- RENDER SKILLS ---------------- //
function renderSkills() {

    const box = document.getElementById("skillsBox");
    if (!box) return;

    box.innerHTML = "";

    Object.keys(skillState).forEach(k => {
        box.innerHTML += `<div>${k}: ${skillState[k].toFixed(2)}</div>`;
    });
}

// ---------------- HINT ---------------- //
function showAIHint() {

    const ans = answers[answers.length - 1];
    if (!ans) return;

    let hint =
        ans.time < 2 ? "⚡ Fast" :
        ans.time > 5 ? "🐢 Slow" :
        "⚖ Normal";

    if (ans.confidence > 80 &&
        ans.weight < Math.max(...ans.all_weights)) {
        hint += " | ⚠ Overconfidence";
    }

    const hintBox = document.getElementById("aiHint");
    if (hintBox) hintBox.innerText = hint;
}

// ---------------- UI ---------------- //
function updateDifficultyUI(q) {
    const el = document.getElementById("difficulty");
    if (el) el.innerText = "Difficulty: " + (q.Difficulty || "Medium");
}

// ---------------- PROGRESS ---------------- //
function updateProgressBar() {

    let percent = (totalAnswered / 17) * 100;

    const bar = document.getElementById("progressBar");
    if (bar) bar.style.width = Math.min(percent, 100) + "%";
}

// ---------------- STOP ---------------- //
function stopQuiz() {

    if (totalAnswered < 17) {
        alert("Answer at least 17 questions!");
        return;
    }

    localStorage.clear();
    window.location = "/quiz-result";
}