(function () {
  document.documentElement.classList.add("motion-enabled");
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function normalize(value) {
    return (value || "").toString().trim().toLowerCase();
  }

  function animateIn() {
    if (prefersReducedMotion) {
      document.documentElement.classList.add("is-ready");
      return;
    }
    requestAnimationFrame(function () {
      document.documentElement.classList.add("is-ready");
    });
  }

  function setupEvidenceFilters() {
    const records = Array.from(document.querySelectorAll('[data-role="latest-record"]'));
    const search = document.getElementById("evidence-search");
    const stanceFilter = document.getElementById("stance-filter");
    const sourceFilter = document.getElementById("source-filter");
    const visibleCount = document.getElementById("visible-evidence-count");

    if (!records.length || !search || !stanceFilter || !sourceFilter || !visibleCount) {
      return;
    }

    const sources = Array.from(new Set(records.map((record) => record.dataset.source).filter(Boolean))).sort();
    for (const source of sources) {
      const option = document.createElement("option");
      option.value = source;
      option.textContent = source;
      sourceFilter.append(option);
    }

    function applyFilters() {
      const query = normalize(search.value);
      const stance = normalize(stanceFilter.value);
      const source = sourceFilter.value;
      let shown = 0;

      for (const record of records) {
        const matchesQuery = !query || normalize(record.textContent).includes(query);
        const matchesStance = !stance || normalize(record.dataset.stance) === stance;
        const matchesSource = !source || record.dataset.source === source;
        const isVisible = matchesQuery && matchesStance && matchesSource;
        record.hidden = !isVisible;
        if (isVisible) {
          shown += 1;
        }
      }

      visibleCount.textContent = `${shown} of ${records.length} evidence records visible`;
    }

    search.addEventListener("input", applyFilters);
    stanceFilter.addEventListener("change", applyFilters);
    sourceFilter.addEventListener("change", applyFilters);
    applyFilters();
  }

  document.addEventListener("DOMContentLoaded", function () {
    animateIn();
    setupEvidenceFilters();
  });
})();
