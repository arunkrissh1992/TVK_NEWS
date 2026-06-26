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
    let districtFilter = ""; // canonical district name or ""
    let departmentFilter = ""; // lowercased department or ""
    let issueFilter = ""; // public_issue text (from a district's top-issue click) or ""

    // --- Pagination -------------------------------------------------------
    // The grid can hold several hundred relevant stories. Rather than scroll a
    // single enormous page, we show one page at a time over the *filtered* set,
    // so every active filter still applies and the page stays short.
    const PAGE_SIZE = 12;
    let currentPage = 1;
    let lastFilterSig = null; // when the filter set changes, jump back to page 1
    const pager = doc.getElementById("narrative-pagination");
    const pagerPrev = pager ? pager.querySelector("[data-page-prev]") : null;
    const pagerNext = pager ? pager.querySelector("[data-page-next]") : null;
    const pagerStatus = pager ? pager.querySelector("[data-page-status]") : null;

    function scrollToNarratives() {
      const sec = doc.getElementById("narratives");
      if (sec) sec.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block: "start" });
    }

    function isMatch(record) {
      const matcher = matchers[currentFilter] || matchers.all;
      if (!matcher(record)) return false;
      if (themeFilterIds && !themeFilterIds.has(record.dataset.rawId)) return false;
      if (sourceFilter && record.dataset.source !== sourceFilter) return false;
      if (districtFilter && record.dataset.districtCanonical !== districtFilter) return false;
      if (
        departmentFilter &&
        (record.dataset.department || "").trim().toLowerCase() !== departmentFilter
      )
        return false;
      if (
        issueFilter &&
        (record.dataset.publicIssue || "").trim().toLowerCase() !== issueFilter
      )
        return false;
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

    // Expose so other helpers (themes, date filter, district map) can re-trigger.
    window.__tnmiApplyFilter = function (opts) {
      opts = opts || {};
      if (opts.themeIds !== undefined) {
        themeFilterIds = opts.themeIds && opts.themeIds.size ? opts.themeIds : null;
      }
      if (opts.dateDays !== undefined) {
        dateWindowDays = Number(opts.dateDays) || 0;
      }
      if (opts.district !== undefined) {
        districtFilter = opts.district || "";
      }
      if (opts.department !== undefined) {
        departmentFilter = (opts.department || "").trim().toLowerCase();
      }
      if (opts.issue !== undefined) {
        issueFilter = (opts.issue || "").trim().toLowerCase();
      }
      applyFilter(true);
    };

    function renderPager(matchedCount, totalPages, startIndex, shownCount) {
      if (visibleCount) {
        if (matchedCount === 0) {
          visibleCount.textContent = "No matching stories";
        } else {
          const from = startIndex + 1;
          const to = startIndex + shownCount;
          let text = "Showing " + from + "–" + to + " of " + matchedCount + " stories";
          if (matchedCount !== records.length) {
            text += " · " + records.length + " total";
          }
          visibleCount.textContent = text;
        }
      }
      if (!pager) return;
      pager.hidden = totalPages <= 1;
      if (pagerStatus) pagerStatus.textContent = "Page " + currentPage + " of " + totalPages;
      if (pagerPrev) pagerPrev.disabled = currentPage <= 1;
      if (pagerNext) pagerNext.disabled = currentPage >= totalPages;
    }

    function applyFilter(animate) {
      // Any change to the active filters returns to page 1; paging through the
      // same filtered set leaves currentPage untouched (signature is unchanged).
      const sig = [
        currentFilter, searchQuery, sourceFilter, dateWindowDays, specificDate,
        themeFilterIds ? Array.from(themeFilterIds).join(",") : "",
        districtFilter, departmentFilter, issueFilter,
      ].join("|");
      if (sig !== lastFilterSig) {
        currentPage = 1;
        lastFilterSig = sig;
      }

      // The matching set drives everything: pagination slices it, the deck
      // counts mirror it, and both card + table views render the same page.
      const matching = records.filter(isMatch);
      const totalPages = Math.max(1, Math.ceil(matching.length / PAGE_SIZE));
      if (currentPage > totalPages) currentPage = totalPages;
      if (currentPage < 1) currentPage = 1;
      const startIndex = (currentPage - 1) * PAGE_SIZE;
      const pageItems = matching.slice(startIndex, startIndex + PAGE_SIZE);
      const pageSet = new Set(pageItems);

      // Cards newly revealed on THIS page (were hidden) get the entry animation.
      const justShown = pageItems.filter(function (el) { return el.hidden; });

      // Authoritative, synchronous visibility: only the current page's cards
      // are shown; everything else (off-page or non-matching) is hidden.
      const visibleRawIds = new Set();
      records.forEach(function (r) {
        const visible = pageSet.has(r);
        r.hidden = !visible;
        if (visible && r.dataset.rawId) visibleRawIds.add(r.dataset.rawId);
        r.style.removeProperty("transform");
        r.style.removeProperty("opacity");
      });

      // Mirror the visible page to the table-view rows so both views stay in sync.
      doc.querySelectorAll('[data-role="latest-row"]').forEach(function (row) {
        row.hidden = !visibleRawIds.has(row.dataset.rawId);
      });

      renderPager(matching.length, totalPages, startIndex, pageItems.length);

      if (!animate || prefersReducedMotion || !hasGsap() || !justShown.length) {
        return;
      }
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

    // Page navigation keeps the filter set fixed and just moves the window.
    if (pagerPrev) {
      pagerPrev.addEventListener("click", function () {
        if (currentPage > 1) {
          currentPage -= 1;
          applyFilter(true);
          scrollToNarratives();
        }
      });
    }
    if (pagerNext) {
      pagerNext.addEventListener("click", function () {
        currentPage += 1;
        applyFilter(true);
        scrollToNarratives();
      });
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
  // ===== District map + department rail — geographic drill-down filters =====
  function setupRegionExplorer() {
    const tiles = Array.from(doc.querySelectorAll(".district-shape"));
    const deptRows = Array.from(doc.querySelectorAll(".dept-row"));
    if (!tiles.length && !deptRows.length) return;

    const mapSvg = doc.getElementById("district-map");
    const regionLayout = doc.querySelector(".region-layout");
    const hoverName = doc.getElementById("map-hover-name");
    const hoverCount = doc.getElementById("map-hover-count");
    const hoverDefault = {
      name: hoverName ? hoverName.textContent : "",
      count: hoverCount ? hoverCount.textContent : "",
    };
    const countBadges = Array.from(doc.querySelectorAll(".district-count"));

    const clearBtn = doc.getElementById("region-clear");
    const detailName = doc.getElementById("district-detail-name");
    const detailTotal = doc.getElementById("district-detail-total");
    const detailCounts = doc.getElementById("district-detail-counts");
    const detailIssues = doc.getElementById("district-detail-issues");
    const detailMlas = doc.getElementById("district-detail-mlas");
    const detailHint = doc.getElementById("district-detail-hint");
    const statewideMlasHtml = detailMlas ? detailMlas.innerHTML : "";

    let districtData = [];
    const dataNode = doc.getElementById("district-data");
    if (dataNode) {
      try {
        districtData = JSON.parse(dataNode.textContent || "[]");
      } catch (_err) {
        districtData = [];
      }
    }
    const byDistrict = {};
    districtData.forEach(function (tile) {
      byDistrict[tile.district] = tile;
    });

    const statewide = {
      name: detailName ? detailName.textContent : "",
      total: detailTotal ? detailTotal.textContent : "",
      hint: detailHint ? detailHint.textContent : "",
    };

    let activeDistrict = "";
    let activeDept = "";

    function chip(kind, label, count) {
      return (
        '<div class="detail-chip detail-chip--' + kind + '"><dt>' + label +
        "</dt><dd>" + count + "</dd></div>"
      );
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, function (ch) {
        return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
      });
    }

    function renderMlas(info) {
      if (!detailMlas) return;
      if (!info || !info.mlas || !info.mlas.length) {
        detailMlas.innerHTML = "";
        return;
      }
      const rows = info.mlas
        .map(function (m) {
          return (
            '<li><span class="mla-constituency">' + escapeHtml(m.constituency) +
            '</span><span class="mla-name">' + escapeHtml(m.mla) +
            '</span><span class="party-chip party-chip--' + escapeHtml(m.party_css) +
            '" title="' + escapeHtml(m.party) + '">' + escapeHtml(m.party_short) + "</span></li>"
          );
        })
        .join("");
      detailMlas.innerHTML =
        "<h4>Constituencies &amp; MLAs (" + info.mlas.length + ")</h4>" +
        '<ul class="mla-list">' + rows + "</ul>";
    }

    function renderDetail() {
      if (!detailName) return;
      if (!activeDistrict) {
        detailName.textContent = statewide.name;
        detailTotal.textContent = statewide.total;
        detailCounts.innerHTML = "";
        detailIssues.innerHTML = "";
        if (detailMlas) detailMlas.innerHTML = statewideMlasHtml;
        detailHint.textContent = statewide.hint;
        return;
      }
      const info = byDistrict[activeDistrict];
      if (!info) return;
      detailName.textContent = info.district;
      detailTotal.textContent =
        info.total + (info.total === 1 ? " story" : " stories") + " in current coverage";
      const chips = [];
      if (info.negative) chips.push(chip("negative", "Negative", info.negative));
      if (info.people) chips.push(chip("people", "People issues", info.people));
      if (info.mixed) chips.push(chip("mixed", "Mixed", info.mixed));
      if (info.positive) chips.push(chip("positive", "Positive", info.positive));
      if (info.neutral) chips.push(chip("neutral", "Neutral", info.neutral));
      detailCounts.innerHTML = chips.join("");
      if (info.top_issues && info.top_issues.length) {
        detailIssues.innerHTML =
          "<h4>Top issues — tap one to see its stories</h4><ul class=\"issue-links\">" +
          info.top_issues
            .map(function (row) {
              return '<li><button type="button" class="issue-link" data-issue="' +
                escapeHtml(row.issue) + '">' +
                '<span class="issue-text">' + escapeHtml(row.issue) + "</span>" +
                '<span class="issue-count">' + row.count + "</span></button></li>";
            })
            .join("") +
          "</ul>";
      } else {
        detailIssues.innerHTML = "";
      }
      renderMlas(info);
      detailHint.textContent = info.total
        ? "Narrative cards below are filtered to " + info.district + "."
        : "No stories tagged to " + info.district + " in the current window.";
    }

    function syncUI() {
      tiles.forEach(function (tile) {
        const isActive = tile.dataset.district === activeDistrict;
        tile.classList.toggle("is-active", isActive);
        tile.setAttribute("aria-selected", isActive ? "true" : "false");
      });
      if (mapSvg) mapSvg.classList.toggle("has-selection", Boolean(activeDistrict));
      // When a district is picked, swap the middle column from Priority Alerts
      // to that district's detail — no separate box needed.
      if (regionLayout) regionLayout.classList.toggle("show-district", Boolean(activeDistrict));
      countBadges.forEach(function (badge) {
        badge.classList.toggle(
          "is-active-count",
          Boolean(activeDistrict) && badge.dataset.district === activeDistrict
        );
      });
      deptRows.forEach(function (row) {
        const isActive = row.dataset.department === activeDept;
        row.classList.toggle("is-active", isActive);
        row.setAttribute("aria-pressed", isActive ? "true" : "false");
      });
      if (clearBtn) clearBtn.hidden = !activeDistrict && !activeDept;
      renderDetail();
      if (window.__tnmiApplyFilter) {
        // Reset any issue drill-down when the district/department changes.
        window.__tnmiApplyFilter({ district: activeDistrict, department: activeDept, issue: "" });
      }
    }

    // Tap a district's top issue → filter the cards to that issue (within the
    // district). Tapping the active issue again clears it. Delegated because
    // renderDetail() rebuilds these buttons on every selection.
    if (detailIssues) {
      detailIssues.addEventListener("click", function (event) {
        const btn = event.target.closest(".issue-link");
        if (!btn) return;
        const wasActive = btn.classList.contains("is-active");
        detailIssues.querySelectorAll(".issue-link").forEach(function (b) {
          b.classList.remove("is-active");
        });
        if (!wasActive) btn.classList.add("is-active");
        if (window.__tnmiApplyFilter) {
          window.__tnmiApplyFilter({ issue: wasActive ? "" : btn.getAttribute("data-issue") || "" });
        }
        const sec = doc.getElementById("narratives");
        if (sec) sec.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block: "start" });
      });
    }

    function toggleDistrict(district) {
      activeDistrict = activeDistrict === district ? "" : district;
      syncUI();
    }

    tiles.forEach(function (tile) {
      tile.addEventListener("click", function () {
        toggleDistrict(tile.dataset.district || "");
      });
      tile.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          toggleDistrict(tile.dataset.district || "");
        }
      });
      tile.addEventListener("mouseenter", function () {
        if (!hoverName) return;
        const info = byDistrict[tile.dataset.district];
        hoverName.textContent = tile.dataset.district;
        hoverCount.textContent = info
          ? info.total + (info.total === 1 ? " story" : " stories")
          : "0 stories";
      });
      tile.addEventListener("mouseleave", function () {
        if (!hoverName) return;
        hoverName.textContent = activeDistrict || hoverDefault.name;
        const info = byDistrict[activeDistrict];
        hoverCount.textContent = activeDistrict && info
          ? info.total + (info.total === 1 ? " story" : " stories")
          : hoverDefault.count;
      });
    });

    deptRows.forEach(function (row) {
      row.addEventListener("click", function () {
        const dept = row.dataset.department || "";
        activeDept = activeDept === dept ? "" : dept;
        syncUI();
      });
    });

    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        activeDistrict = "";
        activeDept = "";
        syncUI();
      });
    }

    // Click-away: clicking anywhere that isn't part of the explorer's
    // interactive surface (or the filtered results being read) clears the
    // selection. Keeps browsing cards/filters from resetting the drill-down.
    const KEEP_SELECTION_SELECTOR = [
      ".district-shape",
      ".dept-row",
      "#region-clear",
      ".district-detail",
      ".narrative-card",
      ".narrative-table",
      ".filter-bar",
      ".fcard",
      ".view-toggle",
      ".chat-panel",
    ].join(", ");
    doc.addEventListener("click", function (event) {
      if (!activeDistrict && !activeDept) return;
      if (event.target.closest && event.target.closest(KEEP_SELECTION_SELECTOR)) return;
      activeDistrict = "";
      activeDept = "";
      syncUI();
    });
  }

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
  // ===== Ask AI — floating assistant open/close =====
  function setupAskAiFab() {
    const wrap = doc.getElementById("ask-ai");
    const toggle = doc.getElementById("ai-fab-toggle");
    const panel = doc.getElementById("ai-fab-panel");
    if (!wrap || !toggle || !panel) return;
    const closeBtn = doc.getElementById("ai-fab-close");
    const navBtn = doc.getElementById("nav-ask-ai");
    const input = doc.getElementById("chat-question");

    function setOpen(open) {
      panel.hidden = !open;
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      wrap.classList.toggle("is-open", open);
      if (open && input) setTimeout(function () { input.focus(); }, 30);
    }

    toggle.addEventListener("click", function () { setOpen(panel.hidden); });
    if (closeBtn) closeBtn.addEventListener("click", function () { setOpen(false); });
    if (navBtn) navBtn.addEventListener("click", function () { setOpen(true); });
    doc.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && !panel.hidden) setOpen(false);
    });
  }

  function setupChatbot() {
    const form = doc.getElementById("chat-form");
    const input = doc.getElementById("chat-question");
    const submit = doc.getElementById("chat-submit");
    const status = doc.getElementById("chat-status");
    const thread = doc.getElementById("chat-thread");
    if (!form || !input || !submit || !status || !thread) return;

    const submitIcon = submit.querySelector("i");
    const originalIcon = submitIcon ? submitIcon.className : "fa-light fa-arrow-up";
    const history = [];
    let busy = false;

    function el(tag, cls) {
      const node = doc.createElement(tag);
      if (cls) node.className = cls;
      return node;
    }

    function scrollToBottom() {
      thread.scrollTop = thread.scrollHeight;
    }

    function autoGrow() {
      input.style.height = "auto";
      input.style.height = Math.min(120, input.scrollHeight) + "px";
    }

    function setBusy(value) {
      busy = value;
      submit.disabled = value;
      input.disabled = value;
      submit.classList.toggle("is-loading", value);
      if (submitIcon) submitIcon.className = value ? "fa-light fa-circle-notch" : originalIcon;
    }

    // ---- tiny, safe markdown -> HTML (escape first, then format) ----
    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function inlineFmt(text) {
      // citations [1] or [1, 2] -> jump chips
      text = text.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, function (_m, nums) {
        return nums
          .split(/\s*,\s*/)
          .map(function (n) {
            return (
              '<sup class="chat-cite" data-ref="' + n +
              '" role="button" tabindex="0" title="Jump to source ' + n + '">' + n + "</sup>"
            );
          })
          .join("");
      });
      text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
      text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      text = text.replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>");
      text = text.replace(/(^|[^_\w])_([^_\n]+)_(?![_\w])/g, "$1<em>$2</em>");
      return text;
    }

    function renderMarkdown(raw) {
      const links = [];
      let src = escapeHtml(raw).replace(/\r\n?/g, "\n");
      // pull links out before inline formatting so URLs are never mangled
      src = src.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, function (_m, label, url) {
        const idx = links.length;
        links.push('<a href="' + url + '" target="_blank" rel="noreferrer">' + inlineFmt(label) + "</a>");
        return "@@LINK" + idx + "@@";
      });

      const lines = src.split("\n");
      let html = "";
      let listType = null;
      let listItems = [];
      let para = [];
      function flushPara() {
        if (para.length) { html += "<p>" + inlineFmt(para.join("<br>")) + "</p>"; para = []; }
      }
      function flushList() {
        if (listType) { html += "<" + listType + ">" + listItems.join("") + "</" + listType + ">"; listType = null; listItems = []; }
      }
      for (let i = 0; i < lines.length; i++) {
        const t = lines[i].trim();
        if (!t) { flushPara(); flushList(); continue; }
        let m;
        if ((m = t.match(/^(#{1,6})\s+(.*)$/))) {
          flushPara(); flushList();
          const lvl = Math.min(4, m[1].length + 2);
          html += "<h" + lvl + ' class="chat-md-h">' + inlineFmt(m[2]) + "</h" + lvl + ">";
        } else if ((m = t.match(/^[-*+]\s+(.*)$/))) {
          flushPara();
          if (listType !== "ul") { flushList(); listType = "ul"; }
          listItems.push("<li>" + inlineFmt(m[1]) + "</li>");
        } else if ((m = t.match(/^\d+[.)]\s+(.*)$/))) {
          flushPara();
          if (listType !== "ol") { flushList(); listType = "ol"; }
          listItems.push("<li>" + inlineFmt(m[1]) + "</li>");
        } else {
          flushList();
          para.push(t);
        }
      }
      flushPara(); flushList();
      html = html.replace(/@@LINK(\d+)@@/g, function (_m, i) { return links[Number(i)] || ""; });
      return html || "<p></p>";
    }

    // ---- message DOM ----
    function addUserMessage(text) {
      const msg = el("div", "chat-msg chat-msg--user");
      const avatar = el("div", "chat-avatar");
      avatar.innerHTML = '<i class="fa-light fa-user" aria-hidden="true"></i>';
      const main = el("div", "chat-msg-main");
      const bubble = el("div", "chat-bubble");
      bubble.textContent = text;
      main.appendChild(bubble);
      msg.append(avatar, main);
      thread.appendChild(msg);
      scrollToBottom();
    }

    function addAiMessage() {
      const msg = el("div", "chat-msg chat-msg--ai");
      const avatar = el("div", "chat-avatar");
      avatar.innerHTML = '<i class="fa-light fa-wand-magic-sparkles" aria-hidden="true"></i>';
      const main = el("div", "chat-msg-main");
      const bubble = el("div", "chat-bubble");
      main.appendChild(bubble);
      msg.append(avatar, main);
      thread.appendChild(msg);
      return { main: main, bubble: bubble };
    }

    function renderEvidenceInto(main, evidence) {
      if (!evidence || !evidence.length) return;
      const wrap = el("div", "chat-evidence");
      const head = el("div", "chat-evidence-head");
      head.textContent = evidence.length + " source" + (evidence.length === 1 ? "" : "s");
      wrap.appendChild(head);
      evidence.forEach(function (item, index) {
        const card = el("article", "chat-evidence-item");
        card.dataset.ref = String(index + 1);

        const title = el("div", "chat-evidence-title");
        const num = el("span", "chat-evidence-num");
        num.textContent = String(index + 1);
        title.appendChild(num);
        title.appendChild(doc.createTextNode(item.title || "Untitled item"));

        const meta = el("div", "chat-evidence-meta");
        meta.textContent = [item.source_name, item.stance, item.district]
          .filter(function (x) { return x && x !== "unspecified"; })
          .join(" · ");

        const snippet = el("p", "chat-evidence-snippet");
        snippet.textContent = item.snippet || item.summary || "";

        const link = el("a", "chat-evidence-link");
        link.href = item.source_url || "#";
        link.target = "_blank";
        link.rel = "noreferrer";
        link.innerHTML = 'Open source <i class="fa-light fa-arrow-up-right-from-square" aria-hidden="true"></i>';

        card.append(title, meta, snippet, link);
        wrap.appendChild(card);
      });
      main.appendChild(wrap);
    }

    // citation chips jump to (and flash) the matching evidence card
    function activateCite(cite) {
      const ref = cite.getAttribute("data-ref");
      const main = cite.closest(".chat-msg-main");
      if (!ref || !main) return;
      const target = main.querySelector('.chat-evidence-item[data-ref="' + ref + '"]');
      if (!target) return;
      target.scrollIntoView({ block: "nearest", behavior: "smooth" });
      target.classList.remove("is-flash");
      void target.offsetWidth;
      target.classList.add("is-flash");
    }
    thread.addEventListener("click", function (event) {
      const cite = event.target.closest && event.target.closest(".chat-cite");
      if (cite) activateCite(cite);
    });
    thread.addEventListener("keydown", function (event) {
      const t = event.target;
      if ((event.key === "Enter" || event.key === " ") && t.classList && t.classList.contains("chat-cite")) {
        event.preventDefault();
        activateCite(t);
      }
    });

    function seedWelcome() {
      const ai = addAiMessage();
      ai.bubble.innerHTML = renderMarkdown(
        "Hello — I'm your briefing assistant. Ask me about the stored Tamil Nadu news and I'll answer **only from saved evidence**, with source links you can verify."
      );
      const suggestions = el("div", "chat-suggestions");
      ["Main people issues today", "Negatives about water", "What is positive for TVK?"].forEach(function (q) {
        const chip = el("button", "chat-chip");
        chip.type = "button";
        chip.textContent = q;
        chip.addEventListener("click", function () {
          if (busy) return;
          input.value = q;
          autoGrow();
          submitForm();
        });
        suggestions.appendChild(chip);
      });
      ai.main.appendChild(suggestions);
    }

    function submitForm() {
      if (form.requestSubmit) form.requestSubmit();
      else form.dispatchEvent(new Event("submit", { cancelable: true }));
    }

    async function ask(question) {
      status.classList.remove("is-error");
      addUserMessage(question);
      input.value = "";
      autoGrow();

      const priorHistory = history.slice(-8);
      history.push({ role: "user", content: question });

      const ai = addAiMessage();
      ai.bubble.innerHTML = '<span class="chat-typing"><i></i><i></i><i></i></span>';
      scrollToBottom();
      setBusy(true);
      status.textContent = "Searching the stored evidence…";

      let answerRaw = "";
      let evidence = [];
      let usedAi = false;
      let modelName = "";
      let firstDelta = true;
      let finished = false;

      function finalize() {
        if (finished) return;
        finished = true;
        if (firstDelta) ai.bubble.innerHTML = renderMarkdown("_No answer was produced. Please try again._");
        renderEvidenceInto(ai.main, evidence);
        if (answerRaw.trim()) history.push({ role: "assistant", content: answerRaw });
        status.classList.remove("is-error");
        status.textContent = usedAi
          ? "Answered by " + (modelName || "local AI") + (evidence.length ? " · " + evidence.length + " source" + (evidence.length === 1 ? "" : "s") : "")
          : (evidence.length ? "Local AI offline — showing evidence only." : "No matching stored evidence found.");
        scrollToBottom();
      }

      function handleEvent(evt) {
        if (!evt || !evt.type) return;
        if (evt.type === "evidence") {
          evidence = Array.isArray(evt.evidence) ? evt.evidence : [];
        } else if (evt.type === "delta") {
          if (firstDelta) { firstDelta = false; ai.bubble.innerHTML = ""; }
          answerRaw += evt.text || "";
          ai.bubble.innerHTML = renderMarkdown(answerRaw);
          scrollToBottom();
        } else if (evt.type === "done") {
          usedAi = !!evt.used_ai;
          modelName = evt.model_name || "";
        }
      }

      try {
        const response = await fetch("/chat/stream", {
          method: "POST",
          headers: buildJsonHeaders(),
          body: JSON.stringify({ question: question, limit: 6, history: priorHistory }),
        });
        if (!response.ok || !response.body) throw new Error("HTTP " + response.status);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const chunk = await reader.read();
          if (chunk.done) break;
          buffer += decoder.decode(chunk.value, { stream: true });
          let nl;
          while ((nl = buffer.indexOf("\n")) >= 0) {
            const line = buffer.slice(0, nl).trim();
            buffer = buffer.slice(nl + 1);
            if (!line) continue;
            try { handleEvent(JSON.parse(line)); } catch (_e) { /* ignore partial line */ }
          }
        }
        const tail = buffer.trim();
        if (tail) { try { handleEvent(JSON.parse(tail)); } catch (_e) { /* ignore */ } }
        finalize();
      } catch (streamErr) {
        // Streaming unavailable — fall back to the non-streaming endpoint.
        try {
          const r = await fetch("/chat/ask", {
            method: "POST",
            headers: buildJsonHeaders(),
            body: JSON.stringify({ question: question, limit: 6 }),
          });
          if (!r.ok) throw new Error("HTTP " + r.status);
          const payload = await r.json();
          firstDelta = false;
          answerRaw = payload.answer || "";
          ai.bubble.innerHTML = renderMarkdown(answerRaw || "_No answer was returned._");
          evidence = Array.isArray(payload.evidence) ? payload.evidence : [];
          usedAi = !!payload.used_ai;
          modelName = payload.model_name || "";
          finalize();
        } catch (fallbackErr) {
          finished = true;
          ai.bubble.innerHTML = renderMarkdown("_Could not reach the assistant. Please try again._");
          status.textContent = "Could not answer now. Please try again.";
          status.classList.add("is-error");
        }
      } finally {
        setBusy(false);
        input.focus();
      }
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      if (busy) return;
      const question = (input.value || "").trim();
      if (question.length < 3) {
        status.textContent = "Please ask a more specific question.";
        status.classList.add("is-error");
        return;
      }
      ask(question);
    });

    input.addEventListener("input", autoGrow);
    input.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submitForm();
      }
    });

    seedWelcome();
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
  // ===== Table view column sort =====
  function setupTableSort() {
    const table = doc.querySelector(".narrative-table");
    if (!table) return;
    const tbody = table.querySelector("tbody");
    if (!tbody) return;
    const headers = Array.from(table.querySelectorAll("th[data-sort]"));
    let sortKey = null;
    let sortDir = 1;
    headers.forEach(function (th) {
      th.classList.add("is-sortable");
      th.setAttribute("tabindex", "0");
      th.setAttribute("role", "button");
      function doSort() {
        const key = th.getAttribute("data-sort");
        if (sortKey === key) sortDir = -sortDir;
        else { sortKey = key; sortDir = 1; }
        headers.forEach(function (h) { h.removeAttribute("aria-sort"); });
        th.setAttribute("aria-sort", sortDir > 0 ? "ascending" : "descending");
        const rows = Array.from(tbody.querySelectorAll('tr[data-role="latest-row"]'));
        rows.sort(function (a, b) {
          const av = (a.dataset[key] || "").toString();
          const bv = (b.dataset[key] || "").toString();
          return av.localeCompare(bv, undefined, { numeric: true, sensitivity: "base" }) * sortDir;
        });
        // Re-append in sorted order; pagination visibility is unchanged, so the
        // rows on the current page render sorted relative to one another.
        rows.forEach(function (r) { tbody.appendChild(r); });
      }
      th.addEventListener("click", doSort);
      th.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); doSort(); }
      });
    });
  }

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
  // ===== Intelligence graph — actor scorecards, entity browser, dossier modal =====
  function setupIntelligence() {
    const scorecards = doc.getElementById("actor-scorecards");
    const chips = doc.getElementById("intel-entity-chips");
    const tabs = Array.from(doc.querySelectorAll(".intel-tab"));
    const modal = doc.getElementById("entity-modal");
    const modalBody = doc.getElementById("entity-modal-body");
    if (!scorecards && !chips) return;

    const PORTRAYAL_COLORS = {
      positive: "#12b76a",
      negative: "#d92d20",
      mixed: "#f79009",
      neutral: "#98a2b3",
    };

    function esc(value) {
      return String(value == null ? "" : value).replace(/[&<>"']/g, function (ch) {
        return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
      });
    }

    function api(path) {
      return fetch(path, { headers: buildJsonHeaders() }).then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
    }

    function favClass(fav) {
      if (fav == null) return "fav-none";
      if (fav >= 60) return "fav-good";
      if (fav <= 40) return "fav-bad";
      return "fav-mid";
    }

    // SVG donut from a portrayal split — the at-a-glance reputation read.
    function donut(split, size) {
      size = size || 92;
      const order = ["positive", "neutral", "mixed", "negative"];
      const total = order.reduce(function (s, k) { return s + (split[k] || 0); }, 0);
      const r = size / 2 - 8;
      const cx = size / 2;
      const c = 2 * Math.PI * r;
      let offset = 0;
      let segs = "";
      if (total === 0) {
        segs = '<circle cx="' + cx + '" cy="' + cx + '" r="' + r + '" fill="none" stroke="#e9edf3" stroke-width="12"/>';
      } else {
        order.forEach(function (k) {
          const val = split[k] || 0;
          if (!val) return;
          const len = (val / total) * c;
          segs +=
            '<circle cx="' + cx + '" cy="' + cx + '" r="' + r + '" fill="none" stroke="' +
            PORTRAYAL_COLORS[k] + '" stroke-width="12" stroke-dasharray="' + len + " " + (c - len) +
            '" stroke-dashoffset="' + (-offset) + '" transform="rotate(-90 ' + cx + " " + cx + ')"/>';
          offset += len;
        });
      }
      return '<svg class="donut" viewBox="0 0 ' + size + " " + size + '" width="' + size + '" height="' + size + '">' + segs + "</svg>";
    }

    // Favourability sparkline across weeks (50 = neutral baseline).
    function sparkline(series) {
      const pts = series.map(function (b) { return b.favorability; });
      const w = 132, h = 30, pad = 3;
      const xs = pts.length > 1 ? (w - 2 * pad) / (pts.length - 1) : 0;
      let d = "";
      let started = false;
      pts.forEach(function (val, i) {
        if (val == null) return;
        const x = pad + i * xs;
        const y = h - pad - ((val / 100) * (h - 2 * pad));
        d += (started ? " L" : "M") + x.toFixed(1) + "," + y.toFixed(1);
        started = true;
      });
      const baseY = h - pad - (0.5 * (h - 2 * pad));
      return (
        '<svg class="spark" viewBox="0 0 ' + w + " " + h + '" width="' + w + '" height="' + h + '">' +
        '<line x1="0" y1="' + baseY + '" x2="' + w + '" y2="' + baseY + '" stroke="#e4e7ec" stroke-width="1" stroke-dasharray="2 2"/>' +
        (d ? '<path d="' + d + '" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>' : "") +
        "</svg>"
      );
    }

    function momentumBadge(mom) {
      if (mom == null || mom === 0) return "";
      const up = mom > 0;
      return (
        '<span class="intel-momentum ' + (up ? "is-up" : "is-down") + '">' +
        '<i class="fa-light fa-arrow-trend-' + (up ? "up" : "down") + '" aria-hidden="true"></i> ' +
        (up ? "+" : "") + mom + "</span>"
      );
    }

    function roleLabel(card) {
      const bits = [];
      if (card.role) bits.push(card.role.replace(/_/g, " "));
      if (card.party) bits.push(card.party);
      return bits.join(" · ");
    }

    function renderScorecards(cards) {
      if (!scorecards) return;
      scorecards.removeAttribute("aria-busy");
      if (!cards.length) {
        scorecards.innerHTML = '<p class="intel-empty">No named figures have enough coverage yet. As more articles are analysed by name, scorecards appear here.</p>';
        return;
      }
      scorecards.innerHTML = cards
        .map(function (c) {
          const fav = c.favorability;
          return (
            '<button type="button" class="intel-card" data-entity-slug="' + esc(c.slug) + '">' +
            '<div class="intel-card-top">' + donut(c.portrayal_split, 64) +
            '<div class="intel-card-id"><span class="intel-card-name">' + esc(c.name) +
            (c.is_tvk ? ' <i class="intel-tvk" title="TVK">★</i>' : "") + "</span>" +
            '<span class="intel-card-role">' + esc(roleLabel(c) || "public figure") + "</span></div></div>" +
            '<div class="intel-card-metrics">' +
            '<span class="intel-fav ' + favClass(fav) + '">' + (fav == null ? "—" : fav) + '<small>fav</small></span>' +
            '<span class="intel-mentions">' + c.mention_count + '<small>mentions</small></span>' +
            momentumBadge(c.momentum) + "</div>" +
            '<div class="intel-card-spark">' + sparkline(c.timeseries) + "</div>" +
            "</button>"
          );
        })
        .join("");
    }

    function renderChips(entities) {
      if (!chips) return;
      chips.removeAttribute("aria-busy");
      if (!entities.length) {
        chips.innerHTML = '<p class="intel-empty">No entities of this type yet.</p>';
        return;
      }
      chips.innerHTML = entities
        .map(function (e) {
          return (
            '<button type="button" class="intel-chip dom-' + esc(e.dominant) +
            '" data-entity-slug="' + esc(e.slug) + '" title="' + esc(e.name) + ' — favourability ' +
            (e.favorability == null ? "n/a" : e.favorability) + '">' +
            '<span class="intel-chip-name">' + esc(e.name) + "</span>" +
            '<span class="intel-chip-count">' + e.mention_count + "</span></button>"
          );
        })
        .join("");
    }

    function evidenceList(items) {
      if (!items.length) return "";
      return (
        '<ul class="dossier-evidence">' +
        items
          .map(function (ev) {
            const link = ev.source_url
              ? '<a href="' + esc(ev.source_url) + '" target="_blank" rel="noopener">' + esc(ev.title || ev.summary || "(story)") + "</a>"
              : esc(ev.title || ev.summary || "(story)");
            return (
              '<li class="dossier-ev dom-' + esc(ev.portrayal) + '">' +
              '<span class="dossier-ev-meta">' + esc(ev.date) + " · " + esc(ev.portrayal) +
              (ev.district ? " · " + esc(ev.district) : "") + (ev.needs_review ? ' · <em>unverified</em>' : "") + "</span>" +
              '<span class="dossier-ev-title">' + link + "</span>" +
              '<span class="dossier-ev-src">' + esc(ev.source_name) + "</span></li>"
            );
          })
          .join("") +
        "</ul>"
      );
    }

    function renderDossier(d) {
      const subtitle = [d.role && d.role.replace(/_/g, " "), d.party, d.portfolio, d.district]
        .filter(Boolean)
        .map(esc)
        .join(" · ");
      const co = d.co_mentions
        .map(function (c) {
          return '<button type="button" class="intel-chip dom-neutral" data-entity-slug="' + esc(c.slug) + '"><span class="intel-chip-name">' + esc(c.name) + '</span><span class="intel-chip-count">' + c.count + "</span></button>";
        })
        .join("");
      const cats = (d.top_categories || []).map(function (t) { return '<span class="dossier-tag">' + esc(t) + "</span>"; }).join("");
      const dists = (d.top_districts || []).map(function (t) { return '<span class="dossier-tag">' + esc(t.district) + " (" + t.count + ")</span>"; }).join("");
      modalBody.innerHTML =
        '<header class="dossier-head">' +
        '<div class="dossier-title"><h3 id="entity-modal-name">' + esc(d.name) +
        (d.is_tvk ? ' <i class="intel-tvk" title="TVK">★</i>' : "") + "</h3>" +
        (d.name_ta ? '<span class="dossier-ta">' + esc(d.name_ta) + "</span>" : "") +
        (subtitle ? '<span class="dossier-sub">' + subtitle + "</span>" : "") +
        '<span class="dossier-type">' + esc(d.entity_type) + "</span></div>" +
        donut(d.portrayal_split, 104) +
        "</header>" +
        '<div class="dossier-metrics">' +
        '<div class="dossier-metric"><b class="' + favClass(d.favorability) + '">' + (d.favorability == null ? "—" : d.favorability) + "</b><span>favourability</span></div>" +
        '<div class="dossier-metric"><b>' + d.mention_count + "</b><span>mentions</span></div>" +
        '<div class="dossier-metric"><b>' + d.mention_count_30d + "</b><span>last 30d</span></div>" +
        '<div class="dossier-metric"><b>' + d.severe_count + "</b><span>high/critical</span></div>" +
        "</div>" +
        '<div class="dossier-trend"><span class="dossier-label">Favourability trend (' + d.timeseries.length + " wks)</span>" + sparkline(d.timeseries) + "</div>" +
        (co ? '<div class="dossier-block"><span class="dossier-label">Appears alongside</span><div class="intel-chips">' + co + "</div></div>" : "") +
        (cats ? '<div class="dossier-block"><span class="dossier-label">Top issues</span><div class="dossier-tags">' + cats + "</div></div>" : "") +
        (dists ? '<div class="dossier-block"><span class="dossier-label">Top districts</span><div class="dossier-tags">' + dists + "</div></div>" : "") +
        '<div class="dossier-block"><span class="dossier-label">Recent evidence</span>' + evidenceList(d.evidence) + "</div>";
    }

    let lastFocus = null;
    function openModal(slug) {
      if (!modal) return;
      lastFocus = doc.activeElement;
      modalBody.innerHTML = '<p class="intel-loading">Loading dossier…</p>';
      modal.hidden = false;
      doc.body.classList.add("modal-open");
      api("/api/entities/" + encodeURIComponent(slug))
        .then(renderDossier)
        .catch(function () {
          modalBody.innerHTML = '<p class="intel-empty">Could not load this dossier.</p>';
        });
    }
    function closeModal() {
      if (!modal) return;
      modal.hidden = true;
      doc.body.classList.remove("modal-open");
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }

    // Delegate clicks: any element carrying data-entity-slug opens the dossier.
    doc.addEventListener("click", function (event) {
      const trigger = event.target.closest("[data-entity-slug]");
      if (trigger) {
        event.preventDefault();
        openModal(trigger.getAttribute("data-entity-slug"));
        return;
      }
      if (event.target.closest("[data-entity-close]")) closeModal();
    });
    doc.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && modal && !modal.hidden) closeModal();
    });

    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (t) {
          const active = t === tab;
          t.classList.toggle("is-active", active);
          t.setAttribute("aria-selected", active ? "true" : "false");
        });
        if (chips) chips.setAttribute("aria-busy", "true");
        api("/api/entities?limit=60&entity_type=" + encodeURIComponent(tab.dataset.entityType))
          .then(renderChips)
          .catch(function () { if (chips) chips.innerHTML = '<p class="intel-empty">Could not load entities.</p>'; });
      });
    });

    if (scorecards) {
      api("/api/actors?limit=12").then(renderScorecards).catch(function () {
        scorecards.innerHTML = '<p class="intel-empty">Could not load key figures.</p>';
      });
    }
    if (chips) {
      api("/api/entities?limit=60&entity_type=person").then(renderChips).catch(function () {
        chips.innerHTML = '<p class="intel-empty">Could not load entities.</p>';
      });
    }
  }

  function boot() {
    animateIn();
    setupCounters();
    setupFilterDeck();
    setupRegionExplorer();
    setupThemesClickThrough();
    setupCardActions();
    setupScrollSpy();
    setupSourceBars();
    setupChatbot();
    setupAskAiFab();
    setupPullLatest();
    setupViewToggle();
    setupTableSort();
    setupToplineDateJump();
    setupStanceCorrection();
    setupIntelligence();
  }

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
