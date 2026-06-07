(function () {
  const doc = document;
  const root = doc.documentElement;
  root.classList.add("motion-enabled");
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hasGsap = () => typeof window.gsap !== "undefined";

  function normalize(value) {
    return (value || "").toString().trim().toLowerCase();
  }

  function buildJsonHeaders() {
    const headers = { "Content-Type": "application/json" };
    const operatorTokenMeta = doc.querySelector('meta[name="x-tnmi-operator-token"]');
    const operatorToken = operatorTokenMeta ? operatorTokenMeta.content : null;
    if (operatorToken) headers["X-TNMI-Operator-Token"] = operatorToken;
    return headers;
  }

  // ===== Initial reveal (everything except filter deck, which has its own choreography) =====
  function markReveal() {
    const targets = doc.querySelectorAll(
      ".briefing-section, .briefing-panel, .official-card, .notice-band"
    );
    let delay = 0;
    targets.forEach(function (el) {
      el.classList.add("reveal");
      el.setAttribute("data-reveal-delay", Math.min(5, delay));
      delay += 1;
      if (delay > 5) delay = 0;
    });
  }

  function animateIn() {
    if (prefersReducedMotion) {
      root.classList.add("is-ready");
      return;
    }
    requestAnimationFrame(function () {
      root.classList.add("is-ready");
    });
  }

  // Entrance is handled via CSS @keyframes — works without rAF / GSAP.

  // ===== Count-up numerals =====
  function easeOutQuart(t) {
    return 1 - Math.pow(1 - t, 4);
  }

  function animateCounter(el, target, duration) {
    if (prefersReducedMotion || target <= 0) {
      el.textContent = target.toLocaleString("en-IN");
      return;
    }
    const start = performance.now();
    function step(now) {
      const elapsed = now - start;
      const t = Math.min(1, elapsed / duration);
      const value = Math.round(target * easeOutQuart(t));
      el.textContent = value.toLocaleString("en-IN");
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function setupCounters() {
    const counters = doc.querySelectorAll("[data-counter]");
    if (!counters.length) return;
    const observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          const el = entry.target;
          const target = parseInt(el.getAttribute("data-counter"), 10) || 0;
          animateCounter(el, target, 1100);
          observer.unobserve(el);
        });
      },
      { threshold: 0.4 }
    );
    counters.forEach(function (el) {
      el.textContent = "0";
      observer.observe(el);
    });
  }

  // ===== Filter deck — click-to-filter the narrative grid =====
  function setupFilterDeck() {
    const cards = Array.from(doc.querySelectorAll(".fcard"));
    const records = Array.from(doc.querySelectorAll('[data-role="latest-record"]'));
    const stanceFilter = doc.getElementById("stance-filter");
    const visibleCount = doc.getElementById("visible-evidence-count");
    if (!cards.length || !records.length) return;

    const matchers = {
      all: () => true,
      positive: (r) => r.dataset.stance === "positive",
      negative: (r) => r.dataset.stance === "negative",
      mixed: (r) => r.dataset.stance === "mixed",
      people: (r) =>
        r.dataset.peopleIssue === "true" ||
        r.dataset.needsReview === "true",
    };

    let currentFilter = "all";
    let searchQuery = "";
    let sourceFilter = "";
    let dateWindowDays = 0; // 0 = all time
    let specificDate = ""; // "YYYY-MM-DD" or "" — exact-day match
    let themeFilterIds = null; // Set<string> of raw_item ids or null

    function isMatch(record) {
      const matcher = matchers[currentFilter] || matchers.all;
      if (!matcher(record)) return false;
      if (themeFilterIds && !themeFilterIds.has(record.dataset.rawId)) return false;
      if (sourceFilter && record.dataset.source !== sourceFilter) return false;
      if (specificDate) {
        const publishedRaw = record.dataset.publishedAt;
        if (!publishedRaw) return false;
        // Take just the YYYY-MM-DD prefix of the ISO timestamp.
        if (publishedRaw.slice(0, 10) !== specificDate) return false;
      } else if (dateWindowDays > 0) {
        const publishedRaw = record.dataset.publishedAt;
        if (!publishedRaw) return false;
        const published = Date.parse(publishedRaw);
        if (isNaN(published)) return false;
        const ageDays = (Date.now() - published) / (1000 * 60 * 60 * 24);
        if (ageDays > dateWindowDays) return false;
      }
      if (searchQuery) {
        const haystack = [
          record.textContent,
          record.dataset.source,
          record.dataset.stance,
          record.dataset.department,
          record.dataset.district,
        ]
          .filter(Boolean)
          .join(" ");
        if (!normalize(haystack).includes(searchQuery)) return false;
      }
      return true;
    }

    // Expose so other helpers (themes, date filter) can re-trigger.
    window.__tnmiApplyFilter = function (opts) {
      opts = opts || {};
      if (opts.themeIds !== undefined) {
        themeFilterIds = opts.themeIds && opts.themeIds.size ? opts.themeIds : null;
      }
      if (opts.dateDays !== undefined) {
        dateWindowDays = Number(opts.dateDays) || 0;
      }
      applyFilter(true);
    };

    function applyFilter(animate) {
      const toShow = [];
      const toHide = [];
      records.forEach(function (record) {
        if (isMatch(record)) toShow.push(record);
        else toHide.push(record);
      });

      if (visibleCount) {
        visibleCount.textContent =
          toShow.length === records.length
            ? records.length + " evidence records"
            : toShow.length + " of " + records.length + " evidence records";
      }

      // Track which cards are NEWLY visible (were hidden, are now shown) BEFORE
      // mutating any state. We use this to add a subtle entry animation.
      const justShown = toShow.filter(function (el) { return el.hidden; });

      // Authoritative state change is SYNCHRONOUS — no setTimeout, no race.
      // Cards that don't match are immediately removed from the layout via
      // [hidden]; cards that do match are immediately visible. Animation, if
      // available, is layered ON TOP for visual polish only.
      const visibleRawIds = new Set();
      records.forEach(function (r) {
        const visible = isMatch(r);
        r.hidden = !visible;
        if (visible && r.dataset.rawId) visibleRawIds.add(r.dataset.rawId);
        // Clear any leftover inline styles from a previous animation.
        r.style.removeProperty("transform");
        r.style.removeProperty("opacity");
      });

      // Mirror the visibility to the table-view rows so both views stay in sync.
      doc.querySelectorAll('[data-role="latest-row"]').forEach(function (row) {
        row.hidden = !visibleRawIds.has(row.dataset.rawId);
      });

      if (!animate || prefersReducedMotion || !hasGsap() || !justShown.length) {
        return;
      }

      // Subtle stagger fade for the cards that just appeared. No exit animation
      // — exits would compete with the synchronous [hidden] removal.
      gsap.fromTo(
        justShown,
        { opacity: 0, y: 8 },
        {
          opacity: 1,
          y: 0,
          duration: 0.32,
          ease: "power3.out",
          stagger: 0.025,
          clearProps: "transform,opacity",
        }
      );
    }

    function setActive(filterKey, animate) {
      currentFilter = filterKey;
      cards.forEach((card) => {
        const isActive = card.dataset.filter === filterKey;
        card.classList.toggle("is-active", isActive);
        card.setAttribute("aria-selected", isActive ? "true" : "false");
      });

      if (stanceFilter) {
        const dropdownValue =
          filterKey === "people" || filterKey === "all" ? "" : filterKey;
        if (stanceFilter.value !== dropdownValue) {
          stanceFilter.value = dropdownValue;
        }
      }

      applyFilter(animate);
    }

    cards.forEach((card) => {
      card.addEventListener("click", function () {
        const key = card.dataset.filter;
        if (!key || key === currentFilter) return;
        setActive(key, true);
      });
      card.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          const key = card.dataset.filter;
          if (key) setActive(key, true);
        }
      });
    });

    // Bridge to existing search/source filter UI so the deck stays in sync.
    const search = doc.getElementById("evidence-search");
    const sourceSelect = doc.getElementById("source-filter");

    if (sourceSelect) {
      const sources = Array.from(
        new Set(records.map((r) => r.dataset.source).filter(Boolean))
      ).sort();
      sources.forEach((source) => {
        const option = doc.createElement("option");
        option.value = source;
        option.textContent = source;
        sourceSelect.append(option);
      });
      sourceSelect.addEventListener("change", function () {
        sourceFilter = sourceSelect.value;
        applyFilter(true);
      });
    }

    if (search) {
      let timer;
      search.addEventListener("input", function () {
        clearTimeout(timer);
        timer = setTimeout(function () {
          searchQuery = normalize(search.value);
          applyFilter(true);
        }, 80);
      });
    }

    if (stanceFilter) {
      stanceFilter.addEventListener("change", function () {
        const dropdownValue = stanceFilter.value;
        const filterKey = dropdownValue || "all";
        setActive(filterKey, true);
      });
    }

    const dateFilter = doc.getElementById("date-filter");
    const dayPicker = doc.getElementById("specific-date-filter");
    const dayClear = doc.getElementById("specific-date-clear");

    function syncDayClearButton() {
      if (!dayClear) return;
      dayClear.hidden = !specificDate;
    }

    if (dateFilter) {
      dateFilter.addEventListener("change", function () {
        dateWindowDays = Number(dateFilter.value) || 0;
        // Range and specific-day are mutually exclusive — clear day if a range
        // is picked, otherwise the two filters fight each other.
        if (dateWindowDays > 0 && dayPicker) {
          dayPicker.value = "";
          specificDate = "";
          syncDayClearButton();
        }
        applyFilter(true);
      });
    }

    if (dayPicker) {
      dayPicker.addEventListener("change", function () {
        specificDate = dayPicker.value || "";
        // Clear the range dropdown so the day filter is the only date filter.
        if (specificDate && dateFilter && dateFilter.value !== "") {
          dateFilter.value = "";
          dateWindowDays = 0;
        }
        syncDayClearButton();
        applyFilter(true);
      });
    }

    if (dayClear) {
      dayClear.addEventListener("click", function () {
        specificDate = "";
        if (dayPicker) dayPicker.value = "";
        syncDayClearButton();
        applyFilter(true);
      });
    }

    // Initial state
    applyFilter(false);
  }

  // ===== Recurring Themes click → narrow narrative grid =====
  function setupThemesClickThrough() {
    const themeCards = Array.from(doc.querySelectorAll(".theme-card[data-theme-members]"));
    if (!themeCards.length || !window.__tnmiApplyFilter) return;

    let activeThemeId = null;

    function setActiveTheme(card) {
      const themeId = card ? card.dataset.themeId : null;
      themeCards.forEach(function (other) {
        const isActive = other === card;
        other.classList.toggle("is-active", isActive);
        other.setAttribute("aria-pressed", isActive ? "true" : "false");
      });
      activeThemeId = themeId;
      const ids = card
        ? new Set(card.dataset.themeMembers.split(",").filter(Boolean))
        : null;
      window.__tnmiApplyFilter({ themeIds: ids });
    }

    themeCards.forEach(function (card) {
      card.addEventListener("click", function () {
        if (card.dataset.themeId === activeThemeId) {
          // toggle off
          setActiveTheme(null);
        } else {
          setActiveTheme(card);
        }
      });
      card.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          card.click();
        }
      });
    });
  }

  // ===== Per-card actions (Mark Reviewed / Flag Escalation) =====
  function setupCardActions() {
    const buttons = Array.from(doc.querySelectorAll(".card-action"));
    if (!buttons.length) return;

    const operatorTokenMeta = doc.querySelector('meta[name="x-tnmi-operator-token"]');
    const operatorToken = operatorTokenMeta ? operatorTokenMeta.content : null;

    function statusFor(action) {
      return action === "review" ? "approved" : "escalated";
    }

    function noteFor(action) {
      return action === "review"
        ? "Marked reviewed from dashboard"
        : "Flagged for escalation from dashboard";
    }

    function flashCard(card, action) {
      if (!card) return;
      if (action === "review") card.classList.add("is-reviewed");
      if (action === "escalate") card.classList.add("is-escalated");
    }

    buttons.forEach(function (btn) {
      btn.addEventListener("click", async function () {
        const action = btn.dataset.action;
        const analysisId = btn.dataset.analysisId;
        if (!action || !analysisId) return;

        const card = btn.closest(".narrative-card");
        const originalHTML = btn.innerHTML;
        btn.disabled = true;
        btn.setAttribute("aria-disabled", "true");
        btn.innerHTML = '<i class="fa-light fa-circle-notch fa-spin" aria-hidden="true"></i><span>Saving</span>';

        try {
          const headers = { "Content-Type": "application/json" };
          if (operatorToken) headers["X-TNMI-Operator-Token"] = operatorToken;
          const response = await fetch("/review/decisions", {
            method: "POST",
            headers: headers,
            body: JSON.stringify({
              analysis_id: Number(analysisId),
              reviewer_name: "Operator",
              status: statusFor(action),
              note: noteFor(action),
            }),
          });
          if (!response.ok) throw new Error("HTTP " + response.status);
          btn.classList.add("is-saved");
          btn.innerHTML =
            action === "review"
              ? '<i class="fa-solid fa-circle-check" aria-hidden="true"></i><span>Reviewed</span>'
              : '<i class="fa-solid fa-flag" aria-hidden="true"></i><span>Escalated</span>';
          flashCard(card, action);
        } catch (err) {
          btn.innerHTML = '<i class="fa-light fa-triangle-exclamation" aria-hidden="true"></i><span>Failed</span>';
          setTimeout(function () {
            btn.disabled = false;
            btn.removeAttribute("aria-disabled");
            btn.innerHTML = originalHTML;
          }, 2200);
        }
      });
    });
  }

  // ===== Scroll-spy nav =====
  function setupScrollSpy() {
    const links = Array.from(doc.querySelectorAll(".nav-link[data-nav-target]"));
    if (!links.length) return;
    const sectionMap = new Map();
    links.forEach(function (link) {
      const id = link.getAttribute("data-nav-target");
      const section = doc.getElementById(id);
      if (section) sectionMap.set(id, { link: link, section: section });
    });
    if (!sectionMap.size) return;

    const observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          const id = entry.target.id;
          const item = sectionMap.get(id);
          if (!item) return;
          if (entry.isIntersecting) {
            links.forEach((l) => l.classList.remove("is-current"));
            item.link.classList.add("is-current");
          }
        });
      },
      { rootMargin: "-40% 0px -50% 0px", threshold: 0 }
    );
    sectionMap.forEach((item) => observer.observe(item.section));

    links.forEach(function (link) {
      link.addEventListener("click", function (event) {
        const id = link.getAttribute("data-nav-target");
        const target = doc.getElementById(id);
        if (!target) return;
        event.preventDefault();
        const navStrip = doc.querySelector(".nav-strip");
        const offsetHeight = navStrip ? navStrip.getBoundingClientRect().height + 16 : 16;
        const top = target.getBoundingClientRect().top + window.scrollY - offsetHeight;
        window.scrollTo({ top: top, behavior: prefersReducedMotion ? "auto" : "smooth" });
      });
    });
  }

  // ===== Source bar entrance =====
  function setupSourceBars() {
    const bars = Array.from(doc.querySelectorAll(".source-bar-fill"));
    if (!bars.length) return;
    bars.forEach(function (bar) {
      const target = bar.style.getPropertyValue("--bar-width") || "0%";
      bar.style.setProperty("--bar-width", "0%");
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          bar.style.setProperty("--bar-width", target);
        });
      });
    });
  }

  // ===== Ask AI grounded in stored newspaper evidence =====
  function setupChatbot() {
    const form = doc.getElementById("chat-form");
    const input = doc.getElementById("chat-question");
    const submit = doc.getElementById("chat-submit");
    const status = doc.getElementById("chat-status");
    const answerCard = doc.getElementById("chat-answer-card");
    const answerText = doc.getElementById("chat-answer-text");
    const answerBadge = doc.getElementById("chat-answer-badge");
    const evidenceList = doc.getElementById("chat-evidence-list");
    if (!form || !input || !submit || !status || !answerCard || !answerText || !answerBadge || !evidenceList) return;

    const icon = submit.querySelector("i");
    const label = submit.querySelector("span");
    const originalIcon = icon ? icon.className : "";
    const originalLabel = label ? label.textContent : "Ask AI";

    function setLoading(isLoading) {
      submit.disabled = isLoading;
      submit.classList.toggle("is-loading", isLoading);
      if (icon) {
        icon.className = isLoading
          ? "fa-light fa-circle-notch"
          : originalIcon;
      }
      if (label) label.textContent = isLoading ? "Reading evidence" : originalLabel;
    }

    function clearEvidence() {
      while (evidenceList.firstChild) evidenceList.removeChild(evidenceList.firstChild);
    }

    function renderEvidence(evidence) {
      clearEvidence();
      evidence.forEach(function (item, index) {
        const article = doc.createElement("article");
        article.className = "chat-evidence-item";

        const titleRow = doc.createElement("div");
        titleRow.className = "chat-evidence-title";

        const title = doc.createElement("span");
        title.textContent = (index + 1) + ". " + (item.title || "Untitled item");

        const meta = doc.createElement("span");
        meta.className = "chat-evidence-meta";
        meta.textContent = [item.source_name, item.stance].filter(Boolean).join(" · ");

        const snippet = doc.createElement("p");
        snippet.className = "chat-evidence-snippet";
        snippet.textContent = item.snippet || item.summary || "";

        const link = doc.createElement("a");
        link.className = "chat-evidence-link";
        link.href = item.source_url || "#";
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = "Open source";

        const linkIcon = doc.createElement("i");
        linkIcon.className = "fa-light fa-arrow-up-right-from-square";
        linkIcon.setAttribute("aria-hidden", "true");
        link.appendChild(linkIcon);

        titleRow.append(title, meta);
        article.append(titleRow, snippet, link);
        evidenceList.appendChild(article);
      });
    }

    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      const question = (input.value || "").trim();
      if (question.length < 3) {
        status.textContent = "Please ask a more specific question.";
        status.classList.add("is-error");
        return;
      }

      status.textContent = "Checking the stored newspaper evidence...";
      status.classList.remove("is-error");
      answerCard.hidden = true;
      clearEvidence();
      setLoading(true);

      try {
        const response = await fetch("/chat/ask", {
          method: "POST",
          headers: buildJsonHeaders(),
          body: JSON.stringify({ question: question, limit: 5 }),
        });
        if (!response.ok) throw new Error("HTTP " + response.status);

        const payload = await response.json();
        answerText.textContent = payload.answer || "No answer was returned.";
        answerBadge.textContent = payload.used_ai
          ? "AI answer from stored evidence"
          : "Stored evidence answer";
        renderEvidence(Array.isArray(payload.evidence) ? payload.evidence : []);
        answerCard.hidden = false;
        status.textContent = payload.evidence && payload.evidence.length
          ? "Answer prepared with " + payload.evidence.length + " source link" + (payload.evidence.length === 1 ? "." : "s.")
          : "No matching stored newspaper evidence was found.";
      } catch (_) {
        status.textContent = "Could not answer now. Please try again after checking the stored data.";
        status.classList.add("is-error");
      } finally {
        setLoading(false);
      }
    });
  }

  // ===== Per-card stance correction (operator override) =====
  function setupStanceCorrection() {
    const containers = Array.from(doc.querySelectorAll(".card-correct"));
    if (!containers.length) return;

    const operatorTokenMeta = doc.querySelector('meta[name="x-tnmi-operator-token"]');
    const operatorToken = operatorTokenMeta ? operatorTokenMeta.content : null;
    function authHeaders() {
      const headers = { "Content-Type": "application/json" };
      if (operatorToken) headers["X-TNMI-Operator-Token"] = operatorToken;
      return headers;
    }

    function closeAllMenus() {
      containers.forEach((c) => {
        const menu = c.querySelector(".card-correct-menu");
        const trigger = c.querySelector(".card-action--correct");
        if (menu) menu.hidden = true;
        if (trigger) trigger.setAttribute("aria-expanded", "false");
      });
    }

    doc.addEventListener("click", function (event) {
      if (!event.target.closest(".card-correct")) closeAllMenus();
    });

    containers.forEach(function (container) {
      const trigger = container.querySelector(".card-action--correct");
      const menu = container.querySelector(".card-correct-menu");
      if (!trigger || !menu) return;

      trigger.addEventListener("click", function (event) {
        event.stopPropagation();
        const isOpen = !menu.hidden;
        closeAllMenus();
        menu.hidden = isOpen;
        trigger.setAttribute("aria-expanded", String(!isOpen));
      });

      Array.from(menu.querySelectorAll(".card-correct-option")).forEach(function (option) {
        option.addEventListener("click", async function (event) {
          event.stopPropagation();
          const stance = option.dataset.stance;
          const analysisId = container.dataset.analysisId;
          if (!stance || !analysisId) return;

          const originalText = trigger.innerHTML;
          trigger.disabled = true;
          trigger.innerHTML = '<i class="fa-light fa-circle-notch fa-spin" aria-hidden="true"></i><span>Saving</span>';
          menu.hidden = true;

          try {
            const response = await fetch("/review/decisions", {
              method: "POST",
              headers: authHeaders(),
              body: JSON.stringify({
                analysis_id: Number(analysisId),
                reviewer_name: "Operator",
                status: "corrected",
                note: "Stance corrected from dashboard",
                corrected_stance: stance,
              }),
            });
            if (!response.ok) throw new Error("HTTP " + response.status);
            trigger.innerHTML = '<i class="fa-solid fa-circle-check" aria-hidden="true"></i><span>Fixed &mdash; reload</span>';
            // Reload after a short delay so the dashboard rebuilds with
            // the corrected stance applied at the source-of-truth layer.
            setTimeout(() => window.location.reload(), 700);
          } catch (err) {
            trigger.innerHTML = '<i class="fa-light fa-triangle-exclamation" aria-hidden="true"></i><span>Failed</span>';
            setTimeout(() => {
              trigger.disabled = false;
              trigger.innerHTML = originalText;
            }, 2500);
          }
        });
      });
    });
  }

  // ===== Topline date click → focus the day picker =====
  function setupToplineDateJump() {
    const trigger = doc.getElementById("topline-date-jump");
    if (!trigger) return;
    trigger.addEventListener("click", function () {
      const picker = doc.getElementById("specific-date-filter");
      if (!picker) return;
      // Scroll the picker into the middle of the viewport, then focus it. On
      // browsers that expose showPicker() we also open the native calendar.
      const navStrip = doc.querySelector(".nav-strip");
      const offsetHeight = navStrip ? navStrip.getBoundingClientRect().height + 24 : 24;
      const top = picker.getBoundingClientRect().top + window.scrollY - offsetHeight;
      window.scrollTo({ top: top, behavior: prefersReducedMotion ? "auto" : "smooth" });
      setTimeout(function () {
        try {
          picker.focus({ preventScroll: true });
          if (typeof picker.showPicker === "function") picker.showPicker();
        } catch (_) { /* not all browsers expose showPicker */ }
      }, prefersReducedMotion ? 0 : 320);
    });
  }

  // ===== View toggle (Cards / Table) =====
  function setupViewToggle() {
    const buttons = Array.from(doc.querySelectorAll(".view-toggle-btn[data-view]"));
    if (!buttons.length) return;

    const STORAGE_KEY = "tnmi.dashboard.view";
    function readPref() {
      try { return localStorage.getItem(STORAGE_KEY); } catch (_) { return null; }
    }
    function writePref(value) {
      try { localStorage.setItem(STORAGE_KEY, value); } catch (_) { /* ignore */ }
    }

    function applyView(view) {
      doc.body.classList.toggle("view-table", view === "table");
      buttons.forEach(function (b) {
        const isActive = b.dataset.view === view;
        b.classList.toggle("is-active", isActive);
        b.setAttribute("aria-selected", isActive ? "true" : "false");
      });
    }

    buttons.forEach(function (b) {
      b.addEventListener("click", function () {
        const view = b.dataset.view;
        if (!view) return;
        applyView(view);
        writePref(view);
      });
    });

    // Priority: ?view= URL param > localStorage > "cards"
    let initial = "cards";
    try {
      const params = new URLSearchParams(window.location.search);
      const urlView = params.get("view");
      if (urlView === "table" || urlView === "cards") initial = urlView;
      else {
        const saved = readPref();
        if (saved === "table") initial = "table";
      }
    } catch (_) { /* ignore */ }
    applyView(initial);
  }

  // ===== Pull Latest — trigger ingest pipeline and poll status =====
  function setupPullLatest() {
    const btn = doc.getElementById("pull-latest-btn");
    if (!btn) return;
    const label = btn.querySelector(".pull-latest-label");
    const icon = btn.querySelector("i");
    const ORIGINAL_LABEL = label ? label.textContent : "Pull Latest";

    const operatorTokenMeta = doc.querySelector('meta[name="x-tnmi-operator-token"]');
    const operatorToken = operatorTokenMeta ? operatorTokenMeta.content : null;
    function authHeaders() {
      const headers = { "Content-Type": "application/json" };
      if (operatorToken) headers["X-TNMI-Operator-Token"] = operatorToken;
      return headers;
    }

    function setUiState(state, labelText, opts) {
      opts = opts || {};
      btn.classList.remove("is-running", "is-success", "is-error");
      if (state === "running") {
        btn.classList.add("is-running");
        btn.disabled = true;
      } else if (state === "success") {
        btn.classList.add("is-success");
        btn.disabled = true;
      } else if (state === "error") {
        btn.classList.add("is-error");
        btn.disabled = false;
      } else {
        btn.disabled = false;
      }
      if (label) label.textContent = labelText;
      if (icon) {
        icon.className = "fa-light " + (
          state === "running" ? "fa-arrows-rotate"
          : state === "success" ? "fa-circle-check"
          : state === "error"   ? "fa-triangle-exclamation"
          :                       "fa-arrows-rotate"
        );
      }
      if (opts.title) btn.title = opts.title;
    }

    let polling = false;

    function poll() {
      if (polling) return;
      polling = true;
      const tick = function () {
        fetch("/pipelines/news/status", { headers: authHeaders(), cache: "no-store" })
          .then(function (r) { return r.json(); })
          .then(function (state) {
            if (state.status === "running") {
              setUiState("running", "Pulling…", {
                title: "Ingest in progress since " + (state.started_at || "now"),
              });
              setTimeout(tick, 2000);
              return;
            }
            polling = false;
            if (state.status === "finished") {
              const r = state.result || {};
              setUiState("success", "Refreshing…", {
                title:
                  "Done. items_seen=" + (r.items_seen ?? "?") +
                  " items_saved=" + (r.items_saved ?? "?") +
                  " analyses_saved=" + (r.analyses_saved ?? "?") +
                  " failures=" + (r.failures ?? "?"),
              });
              setTimeout(function () { window.location.reload(); }, 600);
              return;
            }
            if (state.status === "failed") {
              setUiState("error", "Pull failed", { title: state.error || "Unknown error" });
              setTimeout(function () { setUiState("idle", ORIGINAL_LABEL, { title: "Fetch the latest newspaper articles" }); }, 5000);
              return;
            }
            setUiState("idle", ORIGINAL_LABEL, { title: "Fetch the latest newspaper articles" });
          })
          .catch(function () {
            polling = false;
            setUiState("error", "Status error", { title: "Could not reach /pipelines/news/status" });
            setTimeout(function () { setUiState("idle", ORIGINAL_LABEL, { title: "Fetch the latest newspaper articles" }); }, 4000);
          });
      };
      tick();
    }

    btn.addEventListener("click", function () {
      setUiState("running", "Starting…");
      fetch("/pipelines/news/run", { method: "POST", headers: authHeaders() })
        .then(function (r) { return r.ok ? r.json() : Promise.reject(r); })
        .then(function () {
          setUiState("running", "Pulling…");
          poll();
        })
        .catch(function () {
          setUiState("error", "Trigger failed", { title: "Could not POST /pipelines/news/run" });
          setTimeout(function () { setUiState("idle", ORIGINAL_LABEL, { title: "Fetch the latest newspaper articles" }); }, 4000);
        });
    });

    // If the page loaded while a previous ingest is still running, resume polling.
    fetch("/pipelines/news/status", { headers: authHeaders(), cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (state) {
        if (state.status === "running") {
          setUiState("running", "Pulling…", {
            title: "Ingest already in progress since " + (state.started_at || "now"),
          });
          poll();
        }
      })
      .catch(function () { /* not fatal */ });
  }

  function boot() {
    markReveal();
    animateIn();
    setupCounters();
    setupFilterDeck();
    setupThemesClickThrough();
    setupCardActions();
    setupScrollSpy();
    setupSourceBars();
    setupChatbot();
    setupPullLatest();
    setupViewToggle();
    setupToplineDateJump();
    setupStanceCorrection();
  }

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
