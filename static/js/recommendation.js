/* Utilities */
function safeUrl(url) {
  if (!url || typeof url !== "string") return "";
  url = url.trim();
  if (
    url.startsWith("data:") ||
    url.startsWith("http://") ||
    url.startsWith("https://") ||
    url.startsWith("blob:")
  )
    return url;
  if (url.startsWith("/") || url.startsWith("./") || url.startsWith("../"))
    return url;
  if (url.match(/^[\w\-\.]+\.[a-z]{2,}/i)) return "https://" + url;
  return "";
}
function esc(s) {
  return typeof s === "string"
    ? s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    : s === null || s === undefined
    ? ""
    : s;
}
function showMessage(type, text) {
  const el = document.getElementById("results-message");
  el.style.display = "block";
  el.className = "message " + type;
  el.textContent = text;
  document.getElementById("results-list").innerHTML = "";
}
function clearMessage() {
  const el = document.getElementById("results-message");
  el.style.display = "none";
  el.className = "message";
  el.textContent = "";
}

/* Metadata load */
async function loadMetadata() {
  try {
    const res = await fetch("/metadata");
    if (!res.ok) throw new Error("Failed to load metadata");
    const meta = await res.json();
    const programEl = document.getElementById("program");
    const streamEl = document.getElementById("stream");
    const quotaEl = document.getElementById("quota");
    const categoryEl = document.getElementById("category");
    const locationEl = document.getElementById("location");

    programEl.innerHTML =
      '<option value="" disabled selected>Select Program</option>';
    (meta.programs || []).forEach((p) => {
      const o = document.createElement("option");
      o.value = p;
      o.textContent = p;
      programEl.appendChild(o);
    });

    function fill(el, arr) {
      el.innerHTML = '<option value="">Any</option>';
      (arr || []).forEach((v) => {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = v;
        el.appendChild(o);
      });
    }
    fill(streamEl, meta.streams);
    fill(quotaEl, meta.quotas);
    fill(categoryEl, meta.categories);
    fill(locationEl, meta.locations);

    if (!meta.programs || meta.programs.length === 0)
      showMessage("error", "No programs found — check CSV.");
  } catch (err) {
    console.error(err);
    showMessage("error", "Failed to load metadata from server.");
  }
}

/* Global store */
let LAST_RESULTS = [];

/* Maps helpers */
function parseLatLonFromLocation(locText) {
  if (!locText || typeof locText !== "string") return null;
  const regex = /(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)/;
  const m = locText.match(regex);
  if (m) return { lat: m[1], lon: m[2] };
  return null;
}
function mapsEmbedUrlForItem(item) {
  const loc =
    item.Location ||
    item["Location"] ||
    item.District ||
    item["District"] ||
    "";
  const coords = parseLatLonFromLocation(loc);
  if (coords) {
    return `https://www.google.com/maps?q=${coords.lat},${coords.lon}&output=embed`;
  } else if (loc && loc.trim() !== "") {
    return `https://www.google.com/maps?q=${encodeURIComponent(
      loc
    )}&output=embed`;
  } else {
    const name = item.Institute || item["Institute"] || "";
    return `https://www.google.com/maps?q=${encodeURIComponent(
      name
    )}&output=embed`;
  }
}

/* Card renderer */
function renderCard(item) {
  const logoUrl = safeUrl(item.logo_image || item["logo_image"] || "");
  const picUrl = safeUrl(item.Picture || item["Picture"] || "");
  const website = item.Website || item["Website"] || "";
  const websiteSafe =
    website && (website.startsWith("http://") || website.startsWith("https://"))
      ? website
      : website
      ? "https://" + website
      : "";

  // left
  const left = document.createElement("div");
  left.className = "card-left";
  const logoRow = document.createElement("div");
  logoRow.className = "logo-row";
  const logoImg = document.createElement("img");
  logoImg.className = "logo-img";
  logoImg.alt = "logo";
  logoImg.src = logoUrl || "";
  logoImg.onerror = function () {
    this.style.display = "none";
  };
  logoRow.appendChild(logoImg);
  const titleWrap = document.createElement("div");
  const instName = esc(item.Institute || item["Institute"] || "");
  const nameLink = document.createElement("a");
  nameLink.className = "inst-link";
  nameLink.href = websiteSafe || "#";
  nameLink.target = "_blank";
  nameLink.rel = "noopener noreferrer";
  nameLink.innerHTML = instName || "(Unknown)";
  titleWrap.appendChild(nameLink);
  logoRow.appendChild(titleWrap);
  left.appendChild(logoRow);

  if (picUrl) {
    const pic = document.createElement("img");
    pic.className = "picture";
    pic.src = picUrl;
    pic.alt = instName + " picture";
    pic.onerror = function () {
      this.style.display = "none";
    };
    left.appendChild(pic);
  } else {
    const empty = document.createElement("div");
    empty.style.height = "140px";
    left.appendChild(empty);
  }

  // embedded map under picture
  const mapUrl = mapsEmbedUrlForItem(item);
  const iframe = document.createElement("iframe");
  iframe.className = "map-frame";
  iframe.src = mapUrl;
  iframe.loading = "lazy";
  iframe.referrerPolicy = "no-referrer-when-downgrade";
  left.appendChild(iframe);

  // right
  const right = document.createElement("div");
  right.className = "card-right";
  const highlight = document.createElement("div");
  highlight.className = "highlight-row";
  const rankPill = document.createElement("div");
  rankPill.className = "rank-pill";
  const predictedRank =
    item["Closing Rank"] ??
    item["Predicted Closing Rank"] ??
    item["PredictedClosingRank"] ??
    "-";
  rankPill.innerHTML = `Closing Rank: <strong>${predictedRank}</strong>`;
  highlight.appendChild(rankPill);

  const openPill = document.createElement("div");
  openPill.className = "rank-pill";
  openPill.innerHTML = `Opening: <strong>${
    item["Opening Rank"] || item.OpeningRank || "-"
  }</strong>`;
  highlight.appendChild(openPill);

  const ctcSpan = document.createElement("div");
  ctcSpan.className = "ctc";
  ctcSpan.style.marginLeft = "auto";
  const avgctc =
    item.average_ctc ?? item["average_ctc"] ?? item["Max Average CTC"] ?? "";
  ctcSpan.innerHTML = `Avg CTC: <strong>${
    avgctc !== "" && !isNaN(parseFloat(avgctc))
      ? parseFloat(avgctc).toFixed(1) + " L"
      : "-"
  }</strong>`;
  highlight.appendChild(ctcSpan);
  right.appendChild(highlight);

  // details grid
  const grid = document.createElement("div");
  grid.className = "details-grid";
  function addDetail(k, v) {
    const d = document.createElement("div");
    d.className = "detail";
    const kk = document.createElement("div");
    kk.className = "k";
    kk.textContent = k;
    const vv = document.createElement("div");
    vv.className = "v";
    vv.innerHTML = v;
    d.appendChild(kk);
    d.appendChild(vv);
    grid.appendChild(d);
  }

  addDetail("Program", esc(item.Program || item["Program"] || "-"));
  addDetail("Stream", esc(item.Stream || item["Stream"] || "-"));
  addDetail("Seat Type", esc(item["Seat Type"] || item["SeatType"] || "-"));
  addDetail("Quota", esc(item.Quota || item["Quota"] || "-"));
  addDetail("Category", esc(item.Category || item["Category"] || "-"));
  addDetail("District", esc(item.District || item["District"] || "-"));
  addDetail(
    "Top Recruiter",
    esc(
      item["top recruiter"] ||
        item.top_recruiter ||
        item["top_recruiters"] ||
        "-"
    )
  );
  addDetail("Top Job Title", esc(item["job_title"] || item.job_title || "-"));
  addDetail(
    "Placement Rank",
    esc(item.institute_rank || item["institute_rank"] || "-")
  );
  addDetail("Median CTC (L)", item.median_ctc || item["median_ctc"] || "-");
  addDetail("Highest CTC (L)", item.highest_ctc || item["highest_ctc"] || "-");

  const fmtScore = (v) => {
    const n = parseFloat(v);
    return !isNaN(n) ? (Math.round(n * 10) / 10).toString() : "-";
  };
  addDetail("Rating", fmtScore(item.rating || item["rating"]));
  addDetail(
    "Placement Score",
    fmtScore(
      item.placement_score ||
        item["placement_score"] ||
        item.placements_score ||
        item["placements_score"]
    )
  );
  addDetail(
    "Overall Score",
    fmtScore(item.overall_aspect_score || item["overall_aspect_score"])
  );
  addDetail("Mess Score", fmtScore(item.mess_score || item["mess_score"]));
  addDetail(
    "Professor Score",
    fmtScore(item.professor_score || item["professor_score"])
  );
  addDetail(
    "Campus Score",
    fmtScore(item.campus_score || item["campus_score"])
  );
  addDetail(
    "Infrastructure Score",
    fmtScore(item.infrastructure_score || item["infrastructure_score"])
  );

  right.appendChild(grid);

  // website small
  const siteWrap = document.createElement("div");
  siteWrap.style.marginTop = "8px";
  const site = websiteSafe || item.Website || item["Website"] || "";
  if (site) {
    const a = document.createElement("a");
    a.href = site;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.className = "small-link";
    a.textContent = site;
    siteWrap.appendChild(a);
  }
  right.appendChild(siteWrap);

  const card = document.createElement("div");
  card.className = "college-card";
  card.appendChild(left);
  card.appendChild(right);
  return card;
}

/* render list */
function renderResultsList(data) {
  const list = document.getElementById("results-list");
  list.innerHTML = "";
  if (!Array.isArray(data) || data.length === 0) {
    list.innerHTML =
      '<p style="color:var(--muted);">No recommendations to show.</p>';
    return;
  }
  data.forEach((item) => {
    const card = renderCard(item);
    list.appendChild(card);
  });
}

/* client-side sort (single metric, default order per metric) */
function applyClientSort(results) {
  let arr = (results || []).slice();
  const metric =
    document.getElementById("client_sort_metric").value ||
    "Predicted Closing Rank";

  // Determine sort direction default:
  // For ranks: ascending (lower better). For scores/ctc: descending.
  const rankMetrics = [
    "Predicted Closing Rank",
    "Closing Rank",
    "PredictedClosingRank",
  ];
  const descendingByDefault = [
    "Max Average CTC",
    "placement_score",
    "overall_aspect_score",
    "professor_score",
    "mess_score",
  ];
  const orderDesc = descendingByDefault.includes(metric)
    ? true
    : !rankMetrics.includes(metric);

  arr.sort((a, b) => {
    const getVal = (obj, key) => {
      if (!obj) return null;
      if (
        key === "Predicted Closing Rank" ||
        key === "PredictedClosingRank" ||
        key === "Closing Rank"
      ) {
        return parseFloat(
          obj["Predicted Closing Rank"] ??
            obj["Closing Rank"] ??
            obj["PredictedClosingRank"] ??
            Infinity
        );
      }
      const val =
        obj[key] ??
        obj[key.replaceAll(" ", "_")] ??
        obj[key.toLowerCase()] ??
        obj[key.toUpperCase()];
      if (val === undefined || val === null || val === "") return null;
      const n = parseFloat(val);
      return isNaN(n) ? ("" + val).toLowerCase() : n;
    };
    const va = getVal(a, metric);
    const vb = getVal(b, metric);
    if (va === null && vb === null) return 0;
    if (va === null) return 1;
    if (vb === null) return -1;
    if (typeof va === "number" && typeof vb === "number") {
      return orderDesc ? vb - va : va - vb;
    } else {
      if (va < vb) return orderDesc ? 1 : -1;
      if (va > vb) return orderDesc ? -1 : 1;
      return 0;
    }
  });

  return arr;
}

/* form submit */
document
  .getElementById("recommendation-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();
    clearMessage();
    const f = e.target;
    const payload = {
      rank: f.rank.value,
      program: f.program.value,
      stream: f.stream.value,
      quota: f.quota.value,
      category: f.category.value,
      location: f.location.value,
      target_year: 2026,
      top_n: 50,
    };
    if (!payload.rank || !payload.program) {
      showMessage("error", "Please provide Rank and Program.");
      return;
    }
    showMessage("info", "Fetching recommendations from server...");
    try {
      const res = await fetch("/recommend_colleges", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await res.json();
      if (!res.ok || result.status !== "success") {
        const cls = result.status === "warning" ? "warning" : "error";
        showMessage(cls, result.message || "No results.");
        if (result.data && Array.isArray(result.data) && result.data.length) {
          LAST_RESULTS = result.data.slice();
          const displayed = applyClientSort(LAST_RESULTS);
          renderResultsList(displayed);
        }
        return;
      }
      clearMessage();
      LAST_RESULTS = result.data || [];
      const displayed = applyClientSort(LAST_RESULTS);
      renderResultsList(displayed);
    } catch (err) {
      console.error(err);
      showMessage("error", "Error fetching recommendations. See server logs.");
    }
  });

/* client-side controls */
document.getElementById("apply_client_sort").addEventListener("click", (e) => {
  e.preventDefault();
  const displayed = applyClientSort(LAST_RESULTS);
  renderResultsList(displayed);
});
document.getElementById("reset_client_sort").addEventListener("click", (e) => {
  e.preventDefault();
  document.getElementById("client_sort_metric").value =
    "Predicted Closing Rank";
  renderResultsList(LAST_RESULTS);
});

/* initial load */
loadMetadata();
