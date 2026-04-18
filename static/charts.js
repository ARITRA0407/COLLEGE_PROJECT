// ===============================
// 📊 SAFE ADAPTIVE CHART SYSTEM (FIXED)
// Works with NEW quiz.html + quiz.js
// ===============================

let skillChart = null;
let progressChart = null;

// ===============================
// 🧠 AUTO BUILD SKILL DATA FROM WINDOW STATE
// ===============================
function getSkillData() {

    if (!window.answers) return null;

    const skills = {};

    window.answers.forEach(a => {
        const cat = a.category || "Unknown";
        const score = a.weight || 0;

        if (!skills[cat]) skills[cat] = 0;
        skills[cat] += score;
    });

    return {
        labels: Object.keys(skills),
        values: Object.values(skills)
    };
}


// ===============================
// 📊 SKILL BAR CHART
// ===============================
function renderSkillChart() {

    const canvas = document.getElementById("skillChart");
    if (!canvas || !window.Chart) return;

    const ctx = canvas.getContext("2d");

    const data = getSkillData();
    if (!data) return;

    if (skillChart) skillChart.destroy();

    skillChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: data.labels,
            datasets: [{
                label: "Skill Score",
                data: data.values,
                backgroundColor: "#00ffcc"
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: "Skill Performance"
                }
            }
        }
    });
}


// ===============================
// 📈 PROGRESS CHART (QUESTIONS ATTEMPTED)
// ===============================
function renderProgressChart() {

    const canvas = document.getElementById("progressChart");
    if (!canvas || !window.Chart) return;

    const ctx = canvas.getContext("2d");

    const attempted = window.answers ? window.answers.length : 0;

    if (progressChart) progressChart.destroy();

    progressChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Answered", "Remaining"],
            datasets: [{
                data: [attempted, Math.max(0, 17 - attempted)],
                backgroundColor: ["#36A2EB", "#e0e0e0"]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: "Quiz Progress"
                }
            }
        }
    });
}


// ===============================
// 🔄 MASTER UPDATE
// ===============================
function updateCharts() {
    renderSkillChart();
    renderProgressChart();
}


// ===============================
// 🚀 AUTO INIT
// ===============================
document.addEventListener("DOMContentLoaded", () => {

    const hasCanvas =
        document.getElementById("skillChart") ||
        document.getElementById("progressChart");

    if (!hasCanvas) return;

    if (window.__chartsLoaded) return;
    window.__chartsLoaded = true;

    setTimeout(updateCharts, 500);
});


// ===============================
// 🔁 LIVE UPDATE HOOK (IMPORTANT)
// call this from quiz.js after every answer
// ===============================
window.refreshCharts = function () {
    updateCharts();
};