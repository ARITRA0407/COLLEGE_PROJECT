const CSV_BASE = '/csv/';
const YEARS = [2021, 2022, 2023, 2024, 2025];
const RANK_FILES = YEARS.map(y => `rank_${y}.csv`);
const COLLEGE_FILE = 'college.csv';
const PLACEMENT_FILE = 'placement.csv';
const REVIEWS_FILE = 'reviews.csv';

function parseCSV(text) {
  const lines = text.split(/\r?\n/);
  const headers = splitCSVLine(lines[0]);
  const rows = [];
  let buf = '';
  for (let i = 1; i < lines.length; i++) {
    buf += lines[i];
    const quotes = (buf.match(/"/g) || []).length;
    if (quotes % 2 !== 0) {
      buf += '\n';
      continue;
    }
    const cols = splitCSVLine(buf);
    if (cols.length === headers.length) {
      const obj = {};
      headers.forEach((h, j) => obj[h.trim()] = (cols[j] || '').trim());
      rows.push(obj);
    }
    buf = '';
  }
  return rows;
}

function splitCSVLine(line) {
  const out = [];
  let cur = '';
  let inQ = false;
  for (const ch of line) {
    if (ch === '"') {
      inQ = !inQ;
      continue;
    }
    if (ch === ',' && !inQ) {
      out.push(cur);
      cur = '';
      continue;
    }
    cur += ch;
  }
  out.push(cur);
  return out;
}

function getVal(row, names) {
  for (const n of names) {
    if (n in row) return row[n];
    const lo = n.toLowerCase();
    for (const k of Object.keys(row))
      if (k.toLowerCase() === lo) return row[k];
  }
  return '';
}

function norm(s) {
  return (s == null ? '' : String(s)).trim().replace(/\s+/g, ' ').toLowerCase();
}

function findBestRow(rows, institute, program = '') {
  const instituteKey = norm(institute);
  const programKey = norm(program);

  return rows.find(row => {
    const rowInstitute = norm(getVal(row, ['Institute', 'institute', 'College', 'college', 'college_name', 'College_name', 'NAME', 'name']));
    const rowProgram = norm(getVal(row, ['Program', 'program', 'Course', 'course']));
    if (!rowInstitute) return false;

    const instituteMatch =
      rowInstitute === instituteKey ||
      rowInstitute.includes(instituteKey) ||
      instituteKey.includes(rowInstitute);

    if (!instituteMatch) return false;
    if (!programKey) return true;
    if (!rowProgram) return true;

    return (
      rowProgram === programKey ||
      rowProgram.includes(programKey) ||
      programKey.includes(rowProgram)
    );
  }) || {};
}

const rankData = {};
let collegeData = [],
  placementData = [],
  reviewsData = [];

const institutePrograms = {};

async function loadAll() {
  await Promise.all(RANK_FILES.map(async (f, idx) => {
    try {
      const txt = await (await fetch(CSV_BASE + f)).text();
      rankData[YEARS[idx]] = parseCSV(txt);
    } catch (e) {
      console.warn('rank load failed', f, e);
      rankData[YEARS[idx]] = [];
    }
  }));

  for (const y of YEARS) {
    for (const row of rankData[y]) {
      const inst = getVal(row, ['Institute', 'institute', 'College', 'college', 'INSTITUTE']);
      const prog = getVal(row, ['Program', 'program', 'Course', 'course']);
      if (!inst || !prog) continue;
      if (!institutePrograms[inst]) institutePrograms[inst] = new Set();
      institutePrograms[inst].add(prog);
    }
  }

  try {
    collegeData = parseCSV(await (await fetch(CSV_BASE + COLLEGE_FILE)).text());
  } catch (e) {
    collegeData = [];
  }
  try {
    placementData = parseCSV(await (await fetch(CSV_BASE + PLACEMENT_FILE)).text());
  } catch (e) {
    placementData = [];
  }
  try {
    reviewsData = parseCSV(await (await fetch(CSV_BASE + REVIEWS_FILE)).text());
  } catch (e) {
    reviewsData = [];
  }

  populateInstituteSelects();
}

function populateInstituteSelects() {
  const instList = Object.keys(institutePrograms).sort((a, b) => a.localeCompare(b));

  ['A', 'B'].forEach(side => {
    const el = document.getElementById('institute' + side);
    while (el.firstChild) el.removeChild(el.firstChild);
    const def = document.createElement('option');
    def.value = '';
    def.textContent = 'Select Institute';
    el.appendChild(def);
    instList.forEach(inst => {
      const opt = document.createElement('option');
      opt.value = inst;
      opt.textContent = inst;
      el.appendChild(opt);
    });
    el.disabled = false;
    el.addEventListener('change', () => onInstituteChange(side));
  });

  document.getElementById('compareBtn').addEventListener('click', doCompare);
}

function onInstituteChange(side) {
  const inst = document.getElementById('institute' + side).value;
  const progSelect = document.getElementById('program' + side);

  while (progSelect.firstChild) progSelect.removeChild(progSelect.firstChild);
  const def = document.createElement('option');
  def.value = '';
  def.textContent = 'Select Program';
  progSelect.appendChild(def);

  if (inst && institutePrograms[inst]) {
    const progs = Array.from(institutePrograms[inst]).sort((a, b) => a.localeCompare(b));
    progs.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      progSelect.appendChild(opt);
    });
    progSelect.disabled = false;
  } else {
    progSelect.disabled = true;
  }

  progSelect.value = '';
  progSelect.addEventListener('change', checkEnableCompare);
  checkEnableCompare();
}

function checkEnableCompare() {
  const ok = document.getElementById('instituteA').value &&
    document.getElementById('programA').value &&
    document.getElementById('instituteB').value &&
    document.getElementById('programB').value;
  const btn = document.getElementById('compareBtn');
  btn.disabled = !ok;
}

function buildPanelHTML(side) {
  return `
    <div class="visuals">
      <img id="photo${side}" class="college-img" src="" alt="college photo" style="display:none"/>
    </div>
    <div class="item-row">
      <div class="label">Location</div>
      <div class="value">
        <div id="locText${side}">—</div>
        <div class="map-frame" id="map${side}"></div>
      </div>
    </div>
    <div class="item-row">
      <div class="label">Website</div>
      <div class="value"><a id="website${side}" href="#" target="_blank">—</a></div>
    </div>
    <div class="chart-card">
      <canvas id="chart${side}" height="120"></canvas>
    </div>
    <div class="stat-grid">
      <div class="stat"><div class="title">Average CTC</div><div class="num" id="avgCtc${side}">—</div></div>
      <div class="stat"><div class="title">Median CTC</div><div class="num" id="medCtc${side}">—</div></div>
      <div class="stat"><div class="title">Highest CTC</div><div class="num" id="highCtc${side}">—</div></div>
      <div class="stat"><div class="title">Institution Rank</div><div class="num" id="instRank${side}">—</div></div>
    </div>
    <div>
      <div class="title" style="color:var(--muted);margin-top:6px">Placement Rating</div>
      <div class="stars" id="stars${side}">—</div>
    </div>
    <div class="item-row" style="margin-top:8px"><div class="label">Sentiment Score</div><div class="value" id="sentiment${side}">—</div></div>
    <div class="item-row"><div class="label">Mess Score</div><div class="value" id="mess${side}">—</div></div>
    <div class="item-row"><div class="label">Professor Score</div><div class="value" id="prof${side}">—</div></div>
    <div class="item-row"><div class="label">Campus Score</div><div class="value" id="campus${side}">—</div></div>
    <div class="item-row"><div class="label">Infrastructure Score</div><div class="value" id="infra${side}">—</div></div>
    <div class="item-row"><div class="label">Overall Aspect Score</div><div class="value" id="overall${side}">—</div></div>
  `;
}

const charts = {
  A: null,
  B: null
};

function doCompare() {
  ['A', 'B'].forEach(side => {
    const body = document.getElementById('body' + side);
    body.innerHTML = buildPanelHTML(side);
    body.classList.add('active');
  });

  const instA = document.getElementById('instituteA').value;
  const progA = document.getElementById('programA').value;
  const instB = document.getElementById('instituteB').value;
  const progB = document.getElementById('programB').value;

  renderBasicInfo('A', instA);
  renderBasicInfo('B', instB);
  renderPlacementAndReview('A', instA, progA);
  renderPlacementAndReview('B', instB, progB);
  updateChart('A', instA, progA);
  updateChart('B', instB, progB);

  notifyHeightSoon();
}

function renderBasicInfo(side, institute) {
  let row = findBestRow(collegeData, institute);

  const photo = getVal(row, ['photo', 'image', 'Image', 'picture', 'photo_url']);
  const website = getVal(row, ['Website', 'website', 'url']);
  const location = getVal(row, ['Location', 'location', 'latlong', 'lat_lon', 'latlon']);

  const photoEl = document.getElementById('photo' + side);
  if (photo) {
    photoEl.onload = notifyHeightSoon;
    photoEl.onerror = notifyHeightSoon;
    photoEl.src = photo.trim();
    photoEl.style.display = 'block';
  }

  const websiteEl = document.getElementById('website' + side);
  websiteEl.href = website || '#';
  websiteEl.textContent = website || '—';

  const locText = document.getElementById('locText' + side);
  const mapDiv = document.getElementById('map' + side);

  if (location) {
    locText.textContent = location;
    const parts = location.split(/[;, ]+/).map(s => s.trim()).filter(Boolean);
    let lat = null,
      lng = null;
    for (const p of parts) {
      if (/^-?\d+(\.\d+)?$/.test(p)) {
        if (lat === null) lat = parseFloat(p);
        else if (lng === null) lng = parseFloat(p);
      }
    }
    if (lat !== null && lng !== null) {
      mapDiv.innerHTML = `<iframe width="100%" height="100%" frameborder="0" scrolling="no"
        src="https://www.openstreetmap.org/export/embed.html?bbox=${lng-0.02}%2C${lat-0.02}%2C${lng+0.02}%2C${lat+0.02}&layer=mapnik&marker=${lat}%2C${lng}"></iframe>`;
    } else {
      mapDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted)">Location not parseable</div>';
    }
  } else {
    locText.textContent = '—';
    mapDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted)">No location data</div>';
  }
  notifyHeightSoon();
}

function renderPlacementAndReview(side, institute, program) {
  const p = findBestRow(placementData, institute, program);
  const r = findBestRow(reviewsData, institute);

  setText('avgCtc' + side, getVal(p, ['average_ctc', 'avg_ctc', 'Average_CTC']));
  setText('medCtc' + side, getVal(p, ['median_ctc', 'median', 'Median_CTC']));
  setText('highCtc' + side, getVal(p, ['highest_ctc', 'highest']));
  setText('instRank' + side, getVal(p, ['inst_rank', 'institution_rank', 'rank']));

  const rawRating = getVal(p, ['placement_rating', 'placement_rating_score', 'rating']);
  document.getElementById('stars' + side).innerHTML = renderStars(parseFloat((rawRating + '').replace(/[^0-9.\-]/g, '')));

  setText('sentiment' + side, getVal(r, ['sentiment_score', 'sentiment']));
  setText('mess' + side, getVal(r, ['mess_score', 'mess']));
  setText('prof' + side, getVal(r, ['professor_score', 'professor']));
  setText('campus' + side, getVal(r, ['campus_score', 'campus']));
  setText('infra' + side, getVal(r, ['infrastructure_score', 'infrastructure']));
  setText('overall' + side, getVal(r, ['overall_aspect_score', 'overall']));

  compareNumericPairs();
  notifyHeightSoon();
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = (val == null || val === '') ? '—' : val;
}

function renderStars(numeric) {
  if (isNaN(numeric)) return '—';
  if (numeric > 5) numeric = Math.max(0, Math.min(100, numeric)) / 100 * 5;
  else numeric = Math.max(0, Math.min(5, numeric));
  let out = '';
  for (let i = 0; i < 5; i++) {
    const pct = Math.round(Math.max(0, Math.min(1, numeric - i)) * 100);
    out += `<span class="star"><span class="star-empty">★</span><span class="star-fill" style="width:${pct}%"><span>★</span></span></span>`;
  }
  return out + ` <small style="color:var(--muted);margin-left:6px">${numeric.toFixed(2)}</small>`;
}

function compareNumericPairs() {
  const keys = ['avgCtc', 'medCtc', 'highCtc', 'instRank', 'sentiment', 'mess', 'prof', 'campus', 'infra', 'overall'];
  for (const k of keys) {
    const aEl = document.getElementById(k + 'A');
    const bEl = document.getElementById(k + 'B');
    if (!aEl || !bEl) continue;
    const aVal = parseFloat((aEl.textContent || '').replace(/[^0-9.\-]/g, ''));
    const bVal = parseFloat((bEl.textContent || '').replace(/[^0-9.\-]/g, ''));
    if (!isNaN(aVal) && !isNaN(bVal)) {
      const inverse = k === 'instRank';
      if (aVal === bVal) {
        aEl.className = 'num';
        bEl.className = 'num';
      } else if ((aVal > bVal) ^ inverse) {
        aEl.className = 'num higher';
        bEl.className = 'num lower';
      } else {
        aEl.className = 'num lower';
        bEl.className = 'num higher';
      }
    } else {
      if (aEl) aEl.className = 'num';
      if (bEl) bEl.className = 'num';
    }
  }
}

function updateChart(side, institute, program) {
  const opening = [],
    closing = [];
  const instituteKey = norm(institute);
  const programKey = norm(program);
  for (const y of YEARS) {
    const rows = (rankData[y] || []).filter(r =>
      norm(getVal(r, ['Institute', 'institute', 'College', 'college', 'NAME'])) === instituteKey &&
      norm(getVal(r, ['Program', 'program', 'Course', 'course'])) === programKey
    );
    if (!rows.length) {
      opening.push(null);
      closing.push(null);
      continue;
    }
    const oVals = rows.map(r => parseFloat(getVal(r, ['Opening Rank', 'opening_rank', 'OR', 'open_rank']))).filter(x => !isNaN(x));
    const cVals = rows.map(r => parseFloat(getVal(r, ['Closing Rank', 'closing_rank', 'CR', 'close_rank']))).filter(x => !isNaN(x));
    opening.push(oVals.length ? oVals.reduce((a, b) => a + b, 0) / oVals.length : null);
    closing.push(cVals.length ? cVals.reduce((a, b) => a + b, 0) / cVals.length : null);
  }
  const ctx = document.getElementById('chart' + side).getContext('2d');
  if (charts[side]) charts[side].destroy();
  charts[side] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: YEARS.map(String),
      datasets: [{
        label: 'Opening Rank',
        data: opening,
        fill: false,
        tension: 0.2,
        borderWidth: 2
      }, {
        label: 'Closing Rank',
        data: closing,
        fill: false,
        tension: 0.2,
        borderWidth: 2
      }]
    },
    options: {
      animation: {
        onComplete: notifyHeightSoon
      },
      plugins: {
        title: {
          display: true,
          text: 'Opening / Closing Rank by Year'
        }
      },
      interaction: {
        mode: 'index'
      },
      scales: {
        y: {
          title: {
            display: true,
            text: 'Rank'
          },
          beginAtZero: false
        },
        x: {
          title: {
            display: true,
            text: 'Year'
          }
        }
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  loadAll().catch(e => console.warn('loadAll failed', e));
  setTimeout(notifyHeight, 200);
  window.addEventListener('load', notifyHeightSoon);
  if ('ResizeObserver' in window) {
    const resizeObserver = new ResizeObserver(() => notifyHeightSoon());
    resizeObserver.observe(document.body);
  }
});

function notifyHeight() {
  const h = Math.max(
    document.documentElement.scrollHeight,
    document.body.scrollHeight
  );
  try {
    window.parent.postMessage({
      type: 'iframeHeight',
      height: h
    }, '*');
  } catch (e) {}
}

function notifyHeightSoon() {
  window.requestAnimationFrame(() => {
    setTimeout(notifyHeight, 60);
  });
}
