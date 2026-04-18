
(function () {
  const select = document.getElementById("college-select");
  const btn = document.getElementById("load-btn");
  const selName = document.getElementById("selected-name");

  const ids = {
    title: document.getElementById("inst-title"),
    district: document.getElementById("inst-district"),
    website: document.getElementById("inst-website"),
    logoWrap: document.getElementById("logo-wrap"),
    hero: document.getElementById("hero-picture"),
    coordLine: document.getElementById("coord-line"),
    numPrograms: document.getElementById("num-programs"),
    programList: document.getElementById("program-list"),
    topRecruiters: document.getElementById("top-recruiters"),
    keyProfiles: document.getElementById("key-profiles"),
    placementAvg: document.getElementById("placement-avg"),
    placementMed: document.getElementById("placement-med"),
    placementHigh: document.getElementById("placement-high"),
    placementRating: document.getElementById("placement-rating"),
    reviewsList: document.getElementById("reviews-list"),
    sentimentScore: document.getElementById("sentiment-score"),
    messScore: document.getElementById("mess-score"),
    professorScore: document.getElementById("professor-score"),
    campusScore: document.getElementById("campus-score"),
    infraScore: document.getElementById("infrastructure-score"),
    overallAspect: document.getElementById("overall-aspect-score"),
    linksBody: document.getElementById("links-body"),
  };

  // small Leaflet map
  let map = null;
  function initMap(lat, lon) {
    try {
      if (!map) {
        map = L.map("mini-map", {
          zoomControl: false,
          attributionControl: false,
        }).setView([lat, lon], 12);
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          maxZoom: 19,
        }).addTo(map);
        L.marker([lat, lon]).addTo(map);
      } else {
        map.setView([lat, lon], 12);
        L.marker([lat, lon]).addTo(map);
      }
    } catch (e) {
      const el = document.getElementById("mini-map");
      el.innerHTML =
        '<div style="padding:12px;font-size:13px;color:#6b7280">Map unavailable</div>';
    }
  }

  function enaSelect(items) {
    select.innerHTML = "";
    const start = document.createElement("option");
    start.value = "";
    start.textContent = "Choose college / institute";
    select.appendChild(start);
    items.forEach((name) => {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      select.appendChild(o);
    });
  }

  async function loadCollegeList() {
    try {
      const res = await fetch("/explore/api/colleges");
      if (!res.ok) throw new Error("Failed to load");
      const json = await res.json();
      enaSelect(json.colleges || []);
    } catch (e) {
      select.innerHTML = '<option value="">Could not load colleges</option>';
      console.error(e);
    }
  }

  function clearDetails() {
    ids.title.textContent = "—";
    ids.district.textContent = "District —";
    ids.website.textContent = "Website —";
    ids.logoWrap.innerHTML = "";
    ids.hero.innerHTML = "";
    ids.coordLine.textContent = "Lat / Lon —";
    ids.numPrograms.textContent = "0";
    ids.programList.innerHTML = "";
    ids.topRecruiters.innerHTML = "—";
    ids.keyProfiles.innerHTML = "—";
    ids.placementAvg.textContent = "—";
    ids.placementMed.textContent = "—";
    ids.placementHigh.textContent = "—";
    ids.placementRating.textContent = "—";
    ids.reviewsList.innerHTML = '<div class="muted">No reviews loaded.</div>';
    ids.sentimentScore.textContent = "—";
    ids.messScore.textContent = "—";
    ids.professorScore.textContent = "—";
    ids.campusScore.textContent = "—";
    ids.infraScore.textContent = "—";
    ids.overallAspect.textContent = "—";
    ids.linksBody.innerHTML = "";
  }

  function renderChipList(containerEl, items) {
    containerEl.innerHTML = "";
    if (!items || !items.length) {
      containerEl.textContent = "—";
      return;
    }
    items.forEach((it) => {
      const c = document.createElement("span");
      c.className = "chip";
      c.textContent = it;
      containerEl.appendChild(c);
    });
  }

  async function loadDetails(name) {
    if (!name) return;
    clearDetails();
    selName.textContent = name;
    try {
      const q = encodeURIComponent(name);
      const [dRes, rRes, pRes] = await Promise.all([
        fetch(`/explore/api/college?name=${q}`),
        fetch(`/explore/api/reviews?name=${q}`),
        fetch(`/explore/api/placement?name=${q}`),
      ]);
      if (!dRes.ok) throw new Error("details failed");
      const d = await dRes.json();
      const reviews = rRes.ok ? await rRes.json() : { reviews: [] };
      const placement = pRes.ok ? await pRes.json() : {};

      // populate UI
      ids.title.textContent = d.institute_name || name;
      ids.district.textContent = "District: " + (d.district || "—");
      if (d.website) {
        ids.linksBody.innerHTML = `<a href="${d.website}" target="_blank" rel="noopener">${d.website}</a>`;
        ids.website.innerHTML = `<a href="${d.website}" target="_blank" rel="noopener" style="color:var(--accent)">${d.website}</a>`;
      } else {
        ids.linksBody.textContent = "—";
      }

      // images: logo and picture (if present)
      if (d.logo_image) {
        ids.logoWrap.innerHTML = `<img alt="logo" src="${d.logo_image}" onerror="this.style.display='none'"/>`;
      }
      if (d.picture) {
        ids.hero.innerHTML = `<img alt="picture" src="${d.picture}" onerror="this.style.backgroundColor='#e6edf3'"/>`;
      } else {
        ids.hero.innerHTML = `<div style="padding:32px;color:var(--muted)">No picture available</div>`;
      }

      // coords
      if (d.latitude && d.longitude) {
        ids.coordLine.textContent = `Lat: ${d.latitude.toFixed(
          6
        )} • Lon: ${d.longitude.toFixed(6)}`;
        initMap(d.latitude, d.longitude);
      } else {
        ids.coordLine.textContent = "Lat / Lon not available";
        document.getElementById("mini-map").innerHTML =
          '<div style="padding:12px;font-size:13px;color:#6b7280">Coordinates not available</div>';
      }

      // programs, top recruiters, key profiles (from placement.csv aggregation)
      const programs =
        d.programs && d.programs.length ? d.programs : placement.programs || [];
      ids.numPrograms.textContent =
        d.num_programs !== undefined && d.num_programs !== null
          ? d.num_programs
          : placement.num_programs || programs.length || 0;
      ids.programList.innerHTML = "";
      if (programs && programs.length) {
        programs.forEach((p) => {
          const li = document.createElement("li");
          li.textContent = p;
          ids.programList.appendChild(li);
        });
      } else {
        ids.programList.innerHTML =
          '<li style="opacity:.6">No program names available</li>';
      }

      const recruiters =
        d.top_recruiters && d.top_recruiters.length
          ? d.top_recruiters
          : placement.top_recruiters || [];
      const profiles =
        d.key_profiles && d.key_profiles.length
          ? d.key_profiles
          : placement.job_profiles || [];
      renderChipList(document.getElementById("top-recruiters"), recruiters);
      renderChipList(document.getElementById("key-profiles"), profiles);

      // scores (aggregated from reviews.csv)
      ids.sentimentScore.textContent = d.sentiment_score || "—";
      ids.messScore.textContent = d.mess_score || "—";
      ids.professorScore.textContent = d.professor_score || "—";
      ids.campusScore.textContent = d.campus_score || "—";
      ids.infraScore.textContent = d.infrastructure_score || "—";
      ids.overallAspect.textContent = d.overall_aspect_score || "—";

      // placement (aggregated numbers)
      ids.placementAvg.textContent =
        placement.avg_ctc || d.placement_summary?.avg_ctc || "—";
      ids.placementMed.textContent =
        placement.median_ctc || d.placement_summary?.median_ctc || "—";
      ids.placementHigh.textContent =
        placement.highest_ctc || d.placement_summary?.highest_ctc || "—";
      ids.placementRating.textContent =
        placement.placement_rating ||
        d.placement_summary?.placement_rating ||
        "—";

      // reviews
      const rlist = (reviews.reviews || []).slice(0, 50);
      if (rlist.length === 0) {
        ids.reviewsList.innerHTML =
          '<div class="muted">No reviews in reviews.csv for this institute.</div>';
      } else {
        ids.reviewsList.innerHTML = "";
        rlist.forEach((rv) => {
          const item = document.createElement("div");
          item.className = "review-item";
          const meta = document.createElement("div");
          meta.className = "review-meta";
          const dstr = rv.date ? ` • ${rv.date}` : "";
          const ratingText = rv.rating ? ` • ${rv.rating}` : "";
          meta.innerHTML = `<strong>${
            rv.source || "Unknown"
          }</strong>${dstr}${ratingText}`;
          const txt = document.createElement("div");
          txt.textContent = rv.review_text || "";
          item.appendChild(meta);
          item.appendChild(txt);
          ids.reviewsList.appendChild(item);
        });
      }
    } catch (e) {
      console.error("error loading details", e);
      ids.reviewsList.innerHTML =
        '<div class="muted">Could not load data for this institute.</div>';
    }
  }

  // bind events
  btn.addEventListener("click", () => {
    const name = select.value;
    if (!name) return alert("Pick an institute first");
    loadDetails(name);
  });

  select.addEventListener("change", (ev) => {
    const v = ev.target.value || "";
    selName.textContent = v || "No institute selected";
  });

  select.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") btn.click();
  });

  // initial load
  loadCollegeList();
})();
