// -------- CONFIG --------
    const CSV_BASE = '/csv/'; // adjust if your CSVs are at '/static/csv/' etc.
    const YEARS = [2021,2022,2023,2024,2025];
    const RANK_FILES = YEARS.map(y => `rank_${y}.csv`);
    const COLLEGE_FILE = 'college.csv';
    const PLACEMENT_FILE = 'placement.csv';
    const REVIEWS_FILE = 'reviews.csv';

// minimal CSV parser (header aware) - UPDATED TO HANDLE MULTI-LINE FIELDS
function parseCSV(text){
  // Use a simpler approach: extract column values directly from the text,
  // respecting quoted newlines and commas.
  
  const headers = splitCSVLine(text.split(/\r?\n/)[0]); // Only split the first line for headers
  const outRows = [];
  
  let line = '';
  let inQ = false;
  let lines = text.split(/\r?\n/);
  
  // Skip the header line
  for(let i=1; i<lines.length; i++){
      line += lines[i];
      // Check if this line closes a quote started earlier
      let openQuotes = (line.match(/"/g) || []).length;
      if (openQuotes % 2 !== 0) {
          // If the quote count is odd, we're inside a quoted field, so keep appending
          line += '\n'; // Preserve the newline for the content
          continue;
      }
      
      // Found a complete record line
      const cols = splitCSVLine(line);
      if(cols.length === headers.length){
        const obj = {};
        for(let j=0; j<headers.length; j++) obj[headers[j].trim()] = (cols[j] !== undefined) ? cols[j].trim() : '';
        outRows.push(obj);
      }
      line = ''; // Reset for the next record
  }
  
  return outRows;
}

    // simple CSV line splitter (handles quoted commas)
    function splitCSVLine(line){
      const out = [];
      let cur = '';
      let inQ = false;
      for(let i=0;i<line.length;i++){
        const ch = line[i];
        if(ch === '"') { inQ = !inQ; continue; }
        if(ch === ',' && !inQ){ out.push(cur); cur=''; continue; }
        cur += ch;
      }
      out.push(cur);
      return out.map(s=>s===undefined? '': s);
    }

    // tolerant get: tries to read many possible column name variants
    function getVal(row, names){
      for(const n of names){
        if(n in row) return row[n];
        const lower = n.toLowerCase();
        for(const k of Object.keys(row)){
          if(k.toLowerCase() === lower) return row[k];
        }
      }
      return '';
    }

    // normalize names for tolerant matching
    function normalizeName(s){
      if(s === undefined || s === null) return '';
      return (''+s).trim().replace(/\s+/g,' ').toLowerCase();
    }

    // global data containers
    const rankData = {}; // year -> rows
    let collegeData = [];
    let placementData = [];
    let reviewsData = [];

    // fetch multiple CSVs
    async function loadAll(){
      // load rank files
      await Promise.all(RANK_FILES.map(async (f,idx) => {
        try{
          const res = await fetch(CSV_BASE + f);
          const txt = await res.text();
          rankData[YEARS[idx]] = parseCSV(txt);
        }catch(e){
          console.warn('Could not load', f, e);
          rankData[YEARS[idx]] = [];
        }
      }));

      // other files
      try{ collegeData = parseCSV(await (await fetch(CSV_BASE + COLLEGE_FILE)).text()); }catch(e){ console.warn('college load failed', e); collegeData=[]; }
      try{ placementData = parseCSV(await (await fetch(CSV_BASE + PLACEMENT_FILE)).text()); }catch(e){ console.warn('placement load failed', e); placementData=[]; }
      try{ reviewsData = parseCSV(await (await fetch(CSV_BASE + REVIEWS_FILE)).text()); }catch(e){ console.warn('reviews load failed', e); reviewsData=[]; }

      populateInstituteProgramLists();
    }

    // combine rankData and derive unique institutes/programs
    function populateInstituteProgramLists(){
      const institutes = new Set();
      const programs = new Set();
      for(const y of YEARS){
        for(const row of rankData[y]){
          const inst = getVal(row,['Institute','institute','college','College','INSTITUTE']);
          const prog = getVal(row,['Program','program','course','Course']);
          if(inst) institutes.add(inst);
          if(prog) programs.add(prog);
        }
      }
      const instList = Array.from(institutes).sort((a,b)=>a.localeCompare(b));
      const progList = Array.from(programs).sort((a,b)=>a.localeCompare(b));

      // helper to fill select, clearing old options and adding a default first
      function fillSelect(elId, items, defaultLabel){
        const el = document.getElementById(elId);
        if(!el) return;
        // clear existing options
        while(el.firstChild) el.removeChild(el.firstChild);
        // add placeholder
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = defaultLabel;
        el.appendChild(placeholder);
        // add items
        for(const it of items){
          const opt = document.createElement('option');
          opt.value = it;
          opt.textContent = it;
          el.appendChild(opt);
        }
        // ensure select is enabled and can be interacted with
        el.disabled = false;
        el.style.pointerEvents = 'auto';
      }

      fillSelect('instituteA', instList, 'Select Institute');
      fillSelect('instituteB', instList, 'Select Institute');
      fillSelect('programA', progList, 'Select Program');
      fillSelect('programB', progList, 'Select Program');

      // when user changes selection, check enable
      const selectors = ['instituteA','programA','instituteB','programB'];
      for(const id of selectors){
        const el = document.getElementById(id);
        if(el) el.addEventListener('change', checkEnableCompare);
      }

      document.getElementById('compareBtn').addEventListener('click', doCompare);
    }

    function checkEnableCompare(){
      const iA = document.getElementById('instituteA').value;
      const pA = document.getElementById('programA').value;
      const iB = document.getElementById('instituteB').value;
      const pB = document.getElementById('programB').value;
      const btn = document.getElementById('compareBtn');
      if(!btn) return;
      if(iA && pA && iB && pB){
        btn.disabled = false; // ensure property is updated
        btn.classList.add('enabled');
      } else {
        btn.disabled = true;
        btn.classList.remove('enabled');
      }
    }

    // perform comparison and render contents
    function doCompare(){
      const instA = document.getElementById('instituteA').value;
      const progA = document.getElementById('programA').value;
      const instB = document.getElementById('instituteB').value;
      const progB = document.getElementById('programB').value;

      renderBasicInfo('A', instA);
      renderBasicInfo('B', instB);

      renderPlacementAndReview('A', instA);
      renderPlacementAndReview('B', instB);

      // initial chart render (averages across all available combinations)
      updateChartForSide('A', progA, instA);
      updateChartForSide('B', progB, instB);
    }

    // RENDER BASIC INFO — updated to match college row by normalized names, logo removed
    function renderBasicInfo(side, institute){
      // find college row by tolerant name matching
      const normalizedTarget = normalizeName(institute);
      let row = null;
      for(const r of collegeData){
        // check multiple candidate name columns often used in college.csv
        const cand = getVal(r, ['college_name','College_name','college','College','Institute','institute','name','Name','institution','Institution','institute_name']);
        if(normalizeName(cand) === normalizedTarget){ row = r; break; }
      }
      // if not found by those, also attempt fuzzy fallback: compare startWith or contains
      if(!row && normalizedTarget){
        for(const r of collegeData){
          const cand = getVal(r, ['college_name','College_name','college','College','Institute','institute','name','Name','institution','Institution','institute_name']);
          const normCand = normalizeName(cand);
          if(normCand && (normCand === normalizedTarget || normCand.indexOf(normalizedTarget) !== -1 || normalizedTarget.indexOf(normCand) !== -1)){
            row = r;
            break;
          }
        }
      }
      row = row || {};

      const photoEl = document.getElementById('photo'+side);
      const websiteEl = document.getElementById('website'+side);
      const locText = document.getElementById('locText'+side);
      const mapDiv = document.getElementById('map'+side);

      const photo = getVal(row,['photo','image','Image','picture','photo_url','photo-url']);
      const website = getVal(row,['Website','website','url']);
      const location = getVal(row,['Location','location','latlong','latitude_longitude','lat_lon','latlon']);

      if(photo){ 
        const resolvedPhoto = (photo+'').trim();
        photoEl.src = resolvedPhoto || photo; 
        photoEl.style.display='block'; 
      }
      else { photoEl.src=''; photoEl.style.display='none'; }

      websiteEl.href = website || '#'; websiteEl.textContent = website || '—';

      if(location){
        locText.textContent = location;
        // try parse lat,long
        const parts = location.split(/[;, ]+/).map(s=>s.trim()).filter(s=>s!=='');
        let lat=null,lng=null;
        for(const p of parts){
          const m = p.match(/^-?\d+(?:\.\d+)?$/);
          if(m){ if(lat===null) lat=parseFloat(p); else if(lng===null) lng=parseFloat(p); }
        }
        if(lat!==null && lng!==null){
          // embed openstreetmap viewer
          mapDiv.innerHTML = `<iframe width="100%" height="100%" frameborder="0" scrolling="no" marginheight="0" marginwidth="0"
            src="https://www.openstreetmap.org/export/embed.html?bbox=${lng-0.02}%2C${lat-0.02}%2C${lng+0.02}%2C${lat+0.02}&layer=mapnik&marker=${lat}%2C${lng}"></iframe>`;
        } else {
          mapDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted)">Location not parseable</div>';
        }
      } else {
        locText.textContent = '—'; mapDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted)">No location</div>';
      }
    }

    function renderPlacementAndReview(side, institute){
      // placementData likely contains institute and avg_ctc etc
      const p = placementData.find(r=>{
        const name = getVal(r,['Institute','institute','college','College','NAME']);
        return name === institute;
      }) || {};

      // reviews: match using college_name primarily
      const r = reviewsData.find(rw=>{
        const name = getVal(rw,['college_name','College_name','college','Institute','institute','College','NAME']);
        return name === institute;
      }) || {};

      // placement
      setText('avgCtc'+side, getVal(p,['average_ctc','avg_ctc','Average_CTC','avgctc']) || '—');
      setText('medCtc'+side, getVal(p,['median_ctc','median','Median_CTC']) || '—');
      setText('highCtc'+side, getVal(p,['highest_ctc','highest','highest_ctc']) || '—');
      setText('instRank'+side, getVal(p,['inst_rank','institution_rank','rank']) || '—');

      // placement rating display as stars (rating may be 0-5 or 0-100)
      const rawRating = getVal(p,['placement_rating','placement_rating_score','rating','placement_rating_percent']) || '';
      const numericRating = parseFloat((rawRating+'').replace(/[^0-9.\-]/g,'')); // try to coerce
      document.getElementById('stars'+side).innerHTML = renderStars(numericRating);

      // reviews (robust names). note we look for college_name in reviews as requested.
      setText('sentiment'+side, getVal(r,['sentiment_score','sentiment','sentiment_score_percent']) || '—');
      setText('mess'+side, getVal(r,['mess_score','mess','mess_score_percent']) || '—');
      setText('prof'+side, getVal(r,['professor_score','professor','professor_score']) || '—');
      setText('campus'+side, getVal(r,['campus_score','campus','campus_score']) || '—');
      setText('infra'+side, getVal(r,['infrastructure_score','infrastructure','infrastructure_score']) || '—');
      setText('overall'+side, getVal(r,['overall_aspect_score','overall','overall_score']) || '—');

      compareNumericPairs();
    }

    function setText(id, val){ document.getElementById(id).textContent = (val===undefined || val===null || val==='') ? '—' : val; }

    // renderStars now supports 0-5 or 0-100 scale and partial-star visuals
    function renderStars(raw){
      if(raw === '' || raw === null || raw === undefined || isNaN(Number(raw))) return '—';
      let numeric = Number(raw);
      // interpret value: if > 5 and looks like percent (0-100), convert to 0-5
      if(numeric > 5){
        // clamp 0..100 then convert
        numeric = Math.max(0, Math.min(100, numeric))/100 * 5;
      } else {
        numeric = Math.max(0, Math.min(5, numeric));
      }
      // numeric between 0..5, possibly fractional
      let out = '';
      for(let i=0;i<5;i++){
        const fill = Math.max(0, Math.min(1, numeric - i)); // 0..1
        const pct = Math.round(fill * 100);
        out += `<span class="star" title="${numeric.toFixed(2)}">
                  <span class="star-empty">★</span>
                  <span class="star-fill" style="width:${pct}%;"><span style="display:inline-block;">★</span></span>
                </span>`;
      }
      out += ` <small style="color:var(--muted); margin-left:6px;">${numeric.toFixed(2)}</small>`;
      return out;
    }

    function compareNumericPairs(){
      // pairs to compare: avgCtc, medCtc, highCtc, instRank, sentiment, mess, prof, campus, infra, overall
      const keys = ['avgCtc','medCtc','highCtc','instRank','sentiment','mess','prof','campus','infra','overall'];
      for(const k of keys){
        const aEl = document.getElementById(k+'A');
        const bEl = document.getElementById(k+'B');
        if(!aEl || !bEl) continue;
        const aVal = parseFloat((aEl && aEl.textContent||'').replace(/[^0-9\.\-]/g,''));
        const bVal = parseFloat((bEl && bEl.textContent||'').replace(/[^0-9\.\-]/g,''));
        if(!isNaN(aVal) && !isNaN(bVal)){
          // for instRank lower is better (rank 1 is top), for others higher is better.
          const inverse = (k==='instRank');
          if(aVal===bVal){ aEl.className='num'; bEl.className='num'; }
          else if((aVal > bVal) ^ inverse){ aEl.className='num higher'; bEl.className='num lower'; }
          else { aEl.className='num lower'; bEl.className='num higher'; }
        } else {
          if(aEl) aEl.className='num'; if(bEl) bEl.className='num';
        }
      }
    }

    // ---- Charting for a side ----
    const charts = {A:null,B:null};

    // Simplified: no filters — plot average OR and CR per year across all rows matching institute+program
    function updateChartForSide(side, programSelected=null, instituteSelected=null){
      if(!programSelected) programSelected = (side==='A')? document.getElementById('programA').value : document.getElementById('programB').value;
      if(!instituteSelected) instituteSelected = (side==='A')? document.getElementById('instituteA').value : document.getElementById('instituteB').value;

      const years = YEARS.slice();
      const opening = [], closing = [];

      for(const y of years){
        const rows = rankData[y] || [];
        // select all rows that match institute+program (no other filtering)
        const rowsToUse = rows.filter(r=>{
          const inst = getVal(r,['Institute','institute','college','College','NAME']);
          const prog = getVal(r,['Program','program','course','Course']);
          return (inst === instituteSelected && prog === programSelected);
        });

        if(rowsToUse.length === 0){
          opening.push(null); closing.push(null);
        } else {
          const oVals = rowsToUse.map(m => parseFloat(getVal(m,['Opening Rank','opening_rank','OR','open_rank']))).filter(x=>!isNaN(x));
          const cVals = rowsToUse.map(m => parseFloat(getVal(m,['Closing Rank','closing_rank','CR','close_rank']))).filter(x=>!isNaN(x));
          const o = oVals.length? (oVals.reduce((a,b)=>a+b,0)/oVals.length): null;
          const c = cVals.length? (cVals.reduce((a,b)=>a+b,0)/cVals.length): null;
          opening.push(o); closing.push(c);
        }
      }

      const chartTitle = `OR/CR (averaged across available data)`;

      const ctx = document.getElementById('chart'+side).getContext('2d');
      if(charts[side]) charts[side].destroy();
      charts[side] = new Chart(ctx, {
        type:'line',
        data:{ labels: YEARS.map(String), datasets:[
          {label:'Opening Rank', data: opening, fill:false, tension:0.2, borderWidth:2},
          {label:'Closing Rank', data: closing, fill:false, tension:0.2, borderWidth:2}
        ]},
        options:{
          plugins: {
            title: { display: true, text: chartTitle }
          },
          interaction:{mode:'index'},
          scales:{ y:{title:{display:true,text:'OR / CR'}, beginAtZero:false}, x:{title:{display:true,text:'Year'}}}
        }
      });
    }

    // ---- startup ----
    document.addEventListener('DOMContentLoaded', () => {
      // call loadAll once DOM is ready
      loadAll().catch(e => console.warn('loadAll failed', e));

      // ensure selects trigger enable-check
      const extraIds = ['instituteA','programA','instituteB','programB'];
      extraIds.forEach(id => {
        const el = document.getElementById(id);
        if(el){
          el.addEventListener('change', checkEnableCompare);
          el.addEventListener('input', checkEnableCompare);
        }
      });

      // ensure compare button is visible
      const compareBtn = document.getElementById('compareBtn');
      function forceShowCompareButton(){
        if(!compareBtn) return false;
        try{
          compareBtn.style.display = 'inline-block';
          compareBtn.style.opacity = '1';
          compareBtn.style.visibility = 'visible';
          compareBtn.style.pointerEvents = 'auto';
          compareBtn.style.zIndex = '999999';
          compareBtn.style.position = 'absolute';
          compareBtn.style.right = '16px';
          compareBtn.style.top = '12px';
          const header = document.querySelector('#panelB .panel-header');
          if(header && compareBtn.parentElement !== header){
            header.appendChild(compareBtn);
          }
          const rect = compareBtn.getBoundingClientRect();
          const visible = (rect.width>0 && rect.height>0 && rect.top >= 0 && rect.left >= 0);
          console.log('[COMPARE DEBUG] compareBtn computed style:', window.getComputedStyle(compareBtn));
          console.log('[COMPARE DEBUG] compareBtn rect:', rect, 'visible:', visible);
          return visible;
        }catch(e){ console.warn('forceShowCompareButton error', e); return false; }
      }

      const visibleNow = forceShowCompareButton();
      setTimeout(() => {
        const ok = forceShowCompareButton();
        if(!ok){
          const dbg = document.getElementById('compare-debug-badge');
          if(dbg) dbg.style.display = 'block';
        } else {
          const dbg = document.getElementById('compare-debug-badge');
          if(dbg) dbg.style.display = 'none';
        }
      }, 200);

      if(compareBtn){
        compareBtn.addEventListener('click', (ev) => { ev.preventDefault(); if(!compareBtn.disabled) doCompare(); });
      }

      // initial enable-check
      setTimeout(() => { try { checkEnableCompare(); } catch(e){ /* silent */ } }, 100);
    });