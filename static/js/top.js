  (function () {
      // Helper to create a card element for a single college entry
      function makeCard(item) {
        const card = document.createElement('article');
        card.className = 'top-card';
        card.setAttribute('role', 'listitem');

        // Head: rank badge + name link
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
        // Prefer Website (capitalized) returned by backend; fall back to lowercase or '#'
        const site = (item.Website || item.website || '').toString().trim();
        title.href = site && site !== '' ? site : '#';
        title.target = '_blank';
        title.rel = 'noopener noreferrer';
        title.textContent = item.Institute || item.Institute || 'Unknown';
        titleWrap.appendChild(title);

        head.appendChild(titleWrap);
        card.appendChild(head);

        // Image with black frame and inner white canvas to standardize appearance
        const imgWrap = document.createElement('div');
        imgWrap.className = 'card-img-wrap';
        const imgInner = document.createElement('div');
        imgInner.className = 'card-img-inner';
        const img = document.createElement('img');
        // directly render image URL; if empty, use placeholder data-url (SVG)
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

        // Footer: optionally show institute district or something
        const footer = document.createElement('div');
        footer.className = 'card-footer';
        const instituteSpan = document.createElement('div');
        instituteSpan.textContent = item.District ? (item.District) : '';
        const yearSpan = document.createElement('div');
        yearSpan.textContent = ''; // reserved for future small info
        footer.appendChild(instituteSpan);
        footer.appendChild(yearSpan);
        card.appendChild(footer);

        return card;
      }

      // Fetch top data and render; duplicate list to create seamless circular scroll.
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

        // ensure at most 10 items (should already be 10)
        data = data.slice(0, 10);

        // Build cards and append them twice for seamless loop
        function appendList(list) {
          for (const item of list) {
            const card = makeCard(item);
            track.appendChild(card);
          }
        }

        appendList(data);
        appendList(data.map(d => Object.assign({}, d))); // duplicate

        // After DOM appended, compute animation
        // We will translate the track from -loopRange -> 0 so items move left->right visually.
        let animationId = null;
        let paused = false;
        let speedPxPerSec = 30; // px/sec — adjust for "slowly" movement
        let offset = 0; // in px; translation applied
        const wrap = document.getElementById('top-scroller-wrap');

        // Pause on hover
        wrap.addEventListener('mouseenter', () => paused = true);
        wrap.addEventListener('mouseleave', () => paused = false);

        // compute max translate range (we duplicated list so loop range = half content width)
        function computeSizes() {
          const totalWidth = track.scrollWidth;
          const visibleWidth = wrap.clientWidth;
          const loopRange = totalWidth / 2; // because we duplicated once
          return { totalWidth, visibleWidth, loopRange };
        }

        let lastTime = performance.now();

        function step(now) {
          const dt = Math.min(100, now - lastTime) / 1000; // seconds
          lastTime = now;
          if (!paused) {
            const { loopRange } = computeSizes();
            // move left-to-right: increase offset towards 0, then wrap to -loopRange
            offset += speedPxPerSec * dt;
            if (offset >= 0) {
              offset = -loopRange + (offset - 0);
            }
            track.style.transform = `translateX(${offset}px)`;
          }
          animationId = requestAnimationFrame(step);
        }

        // initialize offset at -loopRange (start left), so movement will go rightwards
        const sizes = computeSizes();
        offset = -sizes.loopRange;
        track.style.transform = `translateX(${offset}px)`;
        // small delay to ensure layout stable then start
        setTimeout(() => {
          lastTime = performance.now();
          if (animationId) cancelAnimationFrame(animationId);
          animationId = requestAnimationFrame(step);
        }, 80);

        // Responsiveness: recompute on resize and reset offset to avoid gap
        window.addEventListener('resize', () => {
          const s = computeSizes();
          // clamp offset into [-loopRange, 0)
          const loopRange = s.loopRange;
          if (offset < -loopRange || offset >= 0) offset = -loopRange;
        });
      }

      // Kickoff
      document.addEventListener('DOMContentLoaded', loadAndStart);
    })();