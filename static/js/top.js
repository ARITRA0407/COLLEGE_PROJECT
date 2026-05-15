(function() {

  function makeCard(item) {
    const card = document.createElement('article');
    card.className = 'top-card';
    card.setAttribute('role', 'listitem');

    const head = document.createElement('div');
    head.className = 'card-head';
    const rank = document.createElement('div');
    rank.className = 'rank-badge';
    rank.textContent = item.rank ?? '';
    head.appendChild(rank);

    const titleWrap = document.createElement('div');
    titleWrap.style.flex = '1';
    const title = document.createElement('a');
    title.className = 'title';

    const site = (item.Website || item.website || '').toString().trim();
    title.href = site && site !== '' ? site : '#';
    title.target = '_blank';
    title.rel = 'noopener noreferrer';
    title.textContent = item.Institute || item.Institute || 'Unknown';
    titleWrap.appendChild(title);

    head.appendChild(titleWrap);
    card.appendChild(head);

    const imgWrap = document.createElement('div');
    imgWrap.className = 'card-img-wrap';
    const imgInner = document.createElement('div');
    imgInner.className = 'card-img-inner';
    const img = document.createElement('img');

    if (item.Picture) {
      img.src = item.Picture;
      img.alt = (item.Institute ? item.Institute + ' picture' : 'College picture');
    } else {
      img.alt = 'No image available';
      img.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200"><rect width="100%" height="100%" fill="#eef6ff"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#7a98b7" font-size="18">No Image</text></svg>'
      );
    }
    imgInner.appendChild(img);
    imgWrap.appendChild(imgInner);
    card.appendChild(imgWrap);

    const footer = document.createElement('div');
    footer.className = 'card-footer';
    const instituteSpan = document.createElement('div');
    instituteSpan.textContent = item.District ? (item.District) : '';
    const yearSpan = document.createElement('div');
    yearSpan.textContent = '';
    footer.appendChild(instituteSpan);
    footer.appendChild(yearSpan);
    card.appendChild(footer);

    return card;
  }

  async function loadAndStart() {
    const track = document.getElementById('top-track');
    track.innerHTML = '';
    let data = [];

    try {
      const res = await fetch('/top/data');
      if (!res.ok) throw new Error('Network response not OK');
      data = await res.json();
    } catch (err) {
      console.error('Failed to fetch /top/data:', err);
      track.innerHTML = '<div class="no-data">Could not load top colleges.</div>';
      return;
    }

    if (!Array.isArray(data) || data.length === 0) {
      track.innerHTML = '<div class="no-data">No data available</div>';
      return;
    }

    data = data.slice(0, 10);

    function appendList(list) {
      for (const item of list) {
        const card = makeCard(item);
        track.appendChild(card);
      }
    }

    appendList(data);
    appendList(data.map(d => Object.assign({}, d)));

    let animationId = null;
    let paused = false;
    let speedPxPerSec = 30;
    let offset = 0;
    const wrap = document.getElementById('top-scroller-wrap');

    wrap.addEventListener('mouseenter', () => paused = true);
    wrap.addEventListener('mouseleave', () => paused = false);

    function computeSizes() {
      const totalWidth = track.scrollWidth;
      const visibleWidth = wrap.clientWidth;
      const loopRange = totalWidth / 2;
      return {
        totalWidth,
        visibleWidth,
        loopRange
      };
    }

    let lastTime = performance.now();

    function step(now) {
      const dt = Math.min(100, now - lastTime) / 1000;
      lastTime = now;
      if (!paused) {
        const {
          loopRange
        } = computeSizes();

        offset += speedPxPerSec * dt;
        if (offset >= 0) {
          offset = -loopRange + (offset - 0);
        }
        track.style.transform = `translateX(${offset}px)`;
      }
      animationId = requestAnimationFrame(step);
    }

    const sizes = computeSizes();
    offset = -sizes.loopRange;
    track.style.transform = `translateX(${offset}px)`;

    setTimeout(() => {
      lastTime = performance.now();
      if (animationId) cancelAnimationFrame(animationId);
      animationId = requestAnimationFrame(step);
    }, 80);

    window.addEventListener('resize', () => {
      const s = computeSizes();

      const loopRange = s.loopRange;
      if (offset < -loopRange || offset >= 0) offset = -loopRange;
    });
  }

  document.addEventListener('DOMContentLoaded', loadAndStart);
})();
