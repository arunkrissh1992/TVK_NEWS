# TVK Knowledge Vault — Political Intelligence Memory Layer

Date: 2026-06-10 (v2 — expanded to full intelligence-loop scope after user review)
Status: Proposed, pending user review
Workspace: `TVK_NEWS`

## 1. Product summary

A **derived "second brain" layer** on top of the existing pipeline: an
Obsidian-compatible markdown vault, regenerated nightly from the database,
holding **one dossier per political object** — every person (CM, ministers,
MLAs, party figures, rivals), party, media outlet, department, district,
constituency, scheme, ongoing issue, narrative, and tracked promise. Every
article, X post, and YouTube transcript that mentions an object lands as a
dated, evidence-linked bullet in that object's dossier, wikilinked (`[[...]]`)
to every other object it touches.

Humans browse it in the Obsidian app (graph view, backlinks, search) starting
from a `Home.md` war-room page. AI agents (the briefing chatbot, Claude Code)
read and grep it for long-horizon context that per-article analysis and
vector RAG cannot give: *"everything we know about X, organized over time."*

The vault is **not** a new source of truth. The DB (`raw_items`,
`ai_analysis`) remains canonical; the vault is a regenerable artifact, exactly
like the RAG index built by `pipelines/build_rag_index.py`.

## 2. The intelligence loop

"An eye watching everything" is only useful if seeing leads to action and
action gets measured. The whole system is one loop; the vault is its memory:

```
   SENSE ──► LINK ──► UNDERSTAND ──► WARN ──► ACT ──► MEASURE
     ▲        (resolve    (synthesize,  (early    (action   (did sentiment
     │         entities,   diagnose)     warning,  tracker,   recover? promise
     │         issues,                   watch-    playbooks) kept? narrative
     │         narratives)               lists)                died?)
     └────────────────────── LEARN ◄──────────────────────────┘
              (flywheel: every human correction → gold label;
               every measured outcome → better playbooks)
```

- SENSE and LEARN already exist (ingestion + flywheel).
- LINK, UNDERSTAND, WARN are the vault core (§5–§12).
- ACT and MEASURE are the response loop (§13) — the "alter things if
  something goes bad" requirement, made concrete and measurable.

## 3. What it answers — asks mapped to mechanisms

Items marked ◆ are additions beyond the user's explicit asks.

| Ask | Mechanism |
|---|---|
| "Every news/article/feed/social post linked together" | Entity resolver turns `political_actors`/`district`/`department`/`scheme` strings into canonical entities; every item lands in the dossiers of everything it mentions, wikilinked both ways |
| "See people sentiment" | Sentiment ledger per district/issue/actor, **split by channel** (newspapers vs X vs YouTube) and **bias-weighted by source profile** (§9); weekly trends in every dossier |
| "Understand caste % in the district" | Static **reference layer**: curated public aggregate data (Census, published surveys, ECI statistics) rendered into district dossiers with source + as-of date. No live feed exists; aggregate-only (§14) |
| "What every minister / MLA / CM is doing" | Person dossiers with auto-built activity feeds from the `actor_mentions` contract extension (§8) |
| "Who is doing good and who is doing bad" | Actor scorecards: per-actor portrayal trend + issue-resolution involvement + promise-keep rate, every line evidence-cited |
| "What is concentrated, what is working / not working" | Department + issue dossiers: coverage share, sentiment/severity trend deltas, resolution signals (§9) |
| "If something goes wrong, find what went wrong and how to fix it" | Issue dossiers aggregate per-article `root_cause` / `recommended_step` / `risk_if_ignored` / `action_owner` (already produced today); weekly Diagnosis Report ranks deteriorating issues with rolled-up causes and actions |
| "An eye watching everything" | ◆ Composite indices (district pulse, department performance, leader standing) + watchlists + early-warning spike detection (§11–§12) — summary numbers that drill down to verbatim quotes |
| "Alter things if something goes bad" | ◆ Action tracker: every recommended action gets a status (proposed→assigned→done) and an **outcome measurement** — sentiment before vs after (§13) |
| "Win elections" | ◆ Electoral intelligence layer: constituency dossiers with ECI history, swing/margin trends, issue salience per AC, alliance arithmetic, candidate track records from public record (§15) |
| ◆ Promises decide elections | **Commitment tracker**: every announced promise (ours AND rivals') becomes a node with deadline + status (kept/stalled/broken); broken-promise alerts before opponents weaponize them; rivals' broken promises become talking points (§10) |
| ◆ Not all outlets are equal | **Source dossiers**: per-outlet leaning, reliability, reach — sentiment indices weight by profile so one hostile high-volume outlet can't drown the signal (§9) |
| ◆ Stories vs ground problems | **Narrative tracking** separate from issues: a story frame ("law and order collapsing") has a lifecycle — emerging→spreading→peak→decaying — tracked across channels with echo velocity (§10) |
| ◆ What govt did vs what people heard | **Coverage-vs-delivery gap**: official announcements/data compared against media pickup — surfaces under-communicated wins and over-exposed failures (§16) |
| ◆ Politics runs on a calendar | **Political calendar**: elections, assembly sessions, budget, festivals, scheme anniversaries — proactive comms windows, not just reactive monitoring (§5) |

## 4. Position in the architecture

```
 GATHER ──► CLASSIFY ──► DB (source of truth: raw_items ⋈ ai_analysis)
                          │
                          ├──► RAG index (chunk_embeddings)      [exists]
                          ├──► dashboard / alerts / briefings    [exists]
                          │
                          └──► KNOWLEDGE VAULT (this spec)       [new]
                                 resolve → aggregate → render → synthesize
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
        Analysts (Obsidian app:        AI agents (chatbot, Claude:
        Home war room, graph,          grep + dossier-first retrieval)
        backlinks)
```

Design rules, in priority order:

1. **DB owns facts; vault owns accumulation + synthesis.** Regenerating the
   vault from scratch must always be safe.
2. **One note per canonical object, never per article.** Articles stay in the
   DB, referenced by `raw_item_id`. Keeps the vault at ~2k notes instead of
   50k+ article stubs that would make the graph unreadable.
3. **Count stories, not reprints.** All indices deduplicate via the existing
   cluster machinery — 14 outlets reprinting one wire story is one data
   point with reach 14, not 14 independent signals. (v2 review fix)
4. **Weight by source profile.** Sentiment indices adjust for outlet leaning
   and reliability so volume ≠ truth. Raw unweighted numbers stay visible
   alongside. (v2 review fix)
5. **Every synthesized claim cites evidence** (`raw_item_id` + quote).
   `needs_human_review` items render with an "unverified" marker.
   Conflicting sources are **flagged as disputes, never averaged away**.
6. **Human sections are sacred.** Each note has an "Analyst notes" section
   the generator never touches.

## 5. Vault layout

```
vault/
  Home.md                     # war room: today's pulse, top risks, broken-promise
                              # watch, narrative momentum, calendar next-7-days
  Calendar.md                 # political calendar (elections, sessions, budget,
                              # festivals, scheme anniversaries, court dates)
  _meta/
    CONVENTIONS.md            # note schema + linking rules — doubles as Claude skill
    templates/
  Entities/
    People/vijay.md           # role: cm — the headline dossier
    People/<minister|mla|rival>.md
    Parties/tvk.md, dmk.md, aiadmk.md, bjp.md, ...
    Organizations/<org>.md
  Sources/<outlet>.md         # ◆ leaning, reliability, reach, history with us
  Government/
    Departments/<dept>.md
    Schemes/<scheme>.md
  Geography/
    Districts/<district>.md   # 38, seeded from tnmi/districts.py
    Constituencies/<ac>.md    # 234 ACs (phase V5)
  Issues/<slug>.md            # ground problems (water, power, jobs...)
  Narratives/<slug>.md        # ◆ story frames with lifecycle + spread metrics
  Commitments/<slug>.md       # ◆ one per tracked promise (ours and rivals')
  Playbooks/<issue-type>.md   # ◆ response playbooks (formalizes the mock
                              #   PublicIssueProfile knowledge)
  Watchlists/<name>.md        # ◆ user-defined watch + thresholds
  Briefings/<YYYY-MM-DD>.md   # existing daily briefings, relocated
  Reports/<YYYY-Www>-diagnosis.md   # weekly "what's wrong & how to fix"
  Archive/<type>/<slug>/<YYYY-MM>.md  # rolled-off evidence
```

`Home.md` is the "eye": regenerated nightly, it is the single page that shows
everything at a glance — every number a wikilink that drills down to dossiers
and from there to verbatim Tamil quotes.

### Note anatomy (district example)

```markdown
---
type: district
slug: madurai
name: Madurai
aliases: [மதுரை, Madurai District]
pulse_index: 38          # 0 worst – 100 best, machine-readable for Dataview
pulse_trend: falling
tags: [district]
---

## Ground reality (reference — curated, slow-moving)
<!-- ref:begin -->
Population, urbanization, community composition (% by group), economy notes,
key constituencies and sitting MLAs — each figure with source + as-of date,
rendered from configs/reference/districts.demographics.yaml.
<!-- ref:end -->

## Current state (auto-synthesized nightly)
<!-- synthesis:begin model=gemma3:27b-cloud date=2026-06-10 -->
Grounded summary: dominant issues, sentiment direction, actors in play,
active narratives. Every sentence cites ^raw ids. Disputes flagged, not
averaged.
<!-- synthesis:end -->

## Pulse (auto-generated)
| week | news +/−/0 (wtd) | social +/−/0 | stories | top issue | risk |
|------|------------------|--------------|---------|-----------|------|
| W23  | 3/11/6           | 1/19/4       | 12      | [[Issues/drinking-water-madurai]] | ▲ escalating |

## Evidence log (auto-generated, last 90 days; older → Archive/)
- 2026-06-09 · negative · high — Water protest at Tallakulam; corporation
  blamed ([[Issues/drinking-water-madurai]], [[People/<mayor>]],
  [[Government/Departments/municipal-administration]]) ^raw:12345

## Analyst notes (human — generator never edits below this line)
```

Person notes add: `role`, `party`, `portfolio`, `is_tvk` frontmatter; an
**activity feed** (what they did, from `actor_mentions`); a **scorecard**
(portrayal split, 4-week trend, promise-keep rate, issues attached); and a
**relations** section (allies/rivals/faction, each edge evidence-cited).

Commitment note (◆ new type):

```markdown
---
type: commitment
by: "[[People/<who>]]"
party: tvk            # ours vs rivals' — both tracked
what: "24h water supply for Madurai by Dec 2026"
promised_on: 2026-03-14
deadline: 2026-12-31
status: in_progress    # announced|in_progress|kept|stalled|broken
---
## Status history          (dated, evidence-cited)
## Coverage                (who is talking about it — silence is also signal)
```

Bilingual aliases in frontmatter are mandatory — Obsidian resolves
`[[மதுரை]]` and `[[Madurai]]` to the same note, which is how the vault stays
coherent across Tamil and English coverage.

## 6. Entity resolution layer (the make-or-break piece)

`ai_analysis.political_actors` is free-text LLM output: "EPS",
"Edappadi K. Palaniswami", and "எடப்பாடி பழனிசாமி" must become **one** node.

New tables (added to `db/schema.sql`, mirrored in `tnmi/storage.py`):

```sql
CREATE TABLE entities (
    id BIGSERIAL PRIMARY KEY,
    entity_type VARCHAR(32) NOT NULL,   -- person|party|org|source|department|district|constituency|scheme
    slug VARCHAR(160) NOT NULL,
    canonical_name VARCHAR(255) NOT NULL,
    name_ta VARCHAR(255) NOT NULL DEFAULT '',
    role VARCHAR(64) NOT NULL DEFAULT '',       -- cm|minister|mla|mp|party_official|other
    party VARCHAR(64) NOT NULL DEFAULT '',
    district VARCHAR(128) NOT NULL DEFAULT '',
    portfolio VARCHAR(128) NOT NULL DEFAULT '',
    is_tvk BOOLEAN NOT NULL DEFAULT false,
    status VARCHAR(16) NOT NULL DEFAULT 'active',  -- active|candidate|retired
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,  -- source profiles: leaning, reliability, reach
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_entities_slug UNIQUE (slug)
);

CREATE TABLE entity_aliases (
    id BIGSERIAL PRIMARY KEY,
    entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias VARCHAR(255) NOT NULL,
    lang VARCHAR(8) NOT NULL DEFAULT '',
    CONSTRAINT uq_entity_alias UNIQUE (entity_id, alias)
);

CREATE TABLE item_entities (
    id BIGSERIAL PRIMARY KEY,
    raw_item_id BIGINT NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
    analysis_id BIGINT REFERENCES ai_analysis(id) ON DELETE SET NULL,
    entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    mention_field VARCHAR(32) NOT NULL,  -- political_actors|district|department|scheme|target|source
    surface VARCHAR(255) NOT NULL,
    portrayal VARCHAR(32) NOT NULL DEFAULT '',      -- from actor_mentions (v17+)
    action_summary TEXT NOT NULL DEFAULT '',
    resolver_version VARCHAR(32) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    CONSTRAINT uq_item_entity UNIQUE (raw_item_id, entity_id, mention_field, surface)
);

CREATE TABLE issues (
    id BIGSERIAL PRIMARY KEY,
    slug VARCHAR(160) NOT NULL,
    title TEXT NOT NULL,
    title_ta TEXT NOT NULL DEFAULT '',
    category VARCHAR(128) NOT NULL DEFAULT '',
    status VARCHAR(16) NOT NULL DEFAULT 'active',   -- active|watch|dormant|resolved
    district VARCHAR(128) NOT NULL DEFAULT '',
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT uq_issues_slug UNIQUE (slug)
);

CREATE TABLE item_issues (
    id BIGSERIAL PRIMARY KEY,
    raw_item_id BIGINT NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
    issue_id BIGINT NOT NULL REFERENCES issues(id) ON DELETE CASCADE,
    similarity DOUBLE PRECISION NOT NULL DEFAULT 0,
    linked_by VARCHAR(16) NOT NULL,      -- cluster|llm|human
    CONSTRAINT uq_item_issue UNIQUE (raw_item_id, issue_id)
);

-- ◆ story frames, distinct from ground issues; same join pattern
CREATE TABLE narratives (
    id BIGSERIAL PRIMARY KEY,
    slug VARCHAR(160) NOT NULL,
    frame TEXT NOT NULL,                 -- "law and order collapsing"
    status VARCHAR(16) NOT NULL DEFAULT 'emerging',  -- emerging|spreading|peak|decaying|dormant
    disputed BOOLEAN NOT NULL DEFAULT false,         -- rumor/misinfo flag
    pushed_by VARCHAR(128) NOT NULL DEFAULT '',      -- who benefits / originates
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT uq_narratives_slug UNIQUE (slug)
);
CREATE TABLE item_narratives (
    id BIGSERIAL PRIMARY KEY,
    raw_item_id BIGINT NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
    narrative_id BIGINT NOT NULL REFERENCES narratives(id) ON DELETE CASCADE,
    similarity DOUBLE PRECISION NOT NULL DEFAULT 0,
    linked_by VARCHAR(16) NOT NULL,
    CONSTRAINT uq_item_narrative UNIQUE (raw_item_id, narrative_id)
);

-- ◆ promises/commitments — ours and rivals'
CREATE TABLE commitments (
    id BIGSERIAL PRIMARY KEY,
    slug VARCHAR(160) NOT NULL,
    entity_id BIGINT NOT NULL REFERENCES entities(id),   -- who promised
    issue_id BIGINT REFERENCES issues(id) ON DELETE SET NULL,
    district VARCHAR(128) NOT NULL DEFAULT '',
    what TEXT NOT NULL,
    promised_on DATE,
    deadline DATE,
    status VARCHAR(16) NOT NULL DEFAULT 'announced',  -- announced|in_progress|kept|stalled|broken
    evidence_raw_item_id BIGINT REFERENCES raw_items(id),
    last_checked TIMESTAMPTZ,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT uq_commitments_slug UNIQUE (slug)
);

-- ◆ the ACT half of the loop: recommendations with status + measured outcome
CREATE TABLE action_items (
    id BIGSERIAL PRIMARY KEY,
    issue_id BIGINT REFERENCES issues(id) ON DELETE SET NULL,
    source_analysis_id BIGINT REFERENCES ai_analysis(id) ON DELETE SET NULL,
    description TEXT NOT NULL,
    owner VARCHAR(128) NOT NULL DEFAULT '',
    priority VARCHAR(16) NOT NULL DEFAULT 'low',
    status VARCHAR(16) NOT NULL DEFAULT 'proposed',  -- proposed|assigned|done|dropped
    acted_on DATE,
    outcome_note TEXT NOT NULL DEFAULT '',           -- filled by MEASURE pass (§13)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Resolver (`tnmi/resolver.py`):

- Alias map: exact match → normalized/fuzzy fallback → unmatched surfaces
  become `status='candidate'` rows for human confirmation (same review-queue
  pattern as classification).
- **Districts reuse `canonical_district()` from `tnmi/districts.py`** —
  already solved, including Tamil names and suffix variants.
- **Sources are entities too** (`entity_type='source'`), seeded from
  `configs/sources.*.yaml`, profile fields in `metadata_json`.
- Seed data `configs/entities.seed.yaml`: 38 districts, parties, departments,
  outlets, people roster (CM, cabinet, opposition leaders, TVK leadership)
  with bilingual aliases. The roster encodes the user's lens (TVK governs;
  Vijay = CM) as **config, not code** — resolves the `_ACTOR_KEYWORDS`
  open question noted in project memory.
- Idempotent backfill CLI: `pipelines/resolve_entities.py`, stamped with
  `resolver_version`.

## 7. Aggregation & indices (deterministic, no LLM)

`tnmi/vault/aggregate.py` computes per dossier per ISO week, always
**story-deduplicated** (cluster representative counts once, reach recorded):

- mention counts by `source_type` (news/x/youtube) — media vs social split;
- sentiment/portrayal splits, raw AND source-weighted, with 4-week deltas;
- severity mix; share-of-coverage per department ("what is concentrated");
- working/not-working heuristic v1:
  `not_working = negative share rising AND recurring ≥ 3 weeks AND severity ≥ high`;
  `working = negative share falling ≥ 2 consecutive weeks OR resolved signal`.

Composite indices (◆) — each a plain arithmetic formula, every input
drillable to quotes, raw components always shown beside the composite:

| Index | Lives in | Blend (v1) |
|---|---|---|
| **District pulse** (0–100) | district dossiers, Home, tile map | weighted negative share (news) + social negative share + severity-weighted issue load + escalation velocity |
| **Department performance** | department dossiers, Home | portrayal trend + issue resolution rate + coverage-vs-delivery gap |
| **Leader standing** | person dossiers | portrayal trend (weighted) + activity volume + promise-keep rate |
| **Narrative momentum** | narrative dossiers, Home | unique outlets this week vs last + cross-channel echo count + news→social lag |
| **Escalation risk** | issue dossiers, Home, alerts | severity × momentum × sentiment extremity × category sensitivity (communal/law-and-order categories weight higher) |

These indices ARE the "eye watching everything": Home.md shows five numbers
per district/department/leader, and every number opens into evidence.

## 8. Contract extension — prompt v17

One prompt bump covering both new signals (mock analyzer kept at parity so
tests and the no-API path stay green; backfill via `reanalyze_items.py`):

```python
class ActorMention(BaseModel):
    name: str                 # surface form as written
    role_hint: str = ""       # cm|minister|mla|party_official|official|other
    action: str = ""          # what they did/said, one line
    portrayal: Stance = Stance.NEUTRAL   # how THIS item portrays THIS actor
    is_tvk: bool | None = None

class CommitmentMention(BaseModel):     # only when item announces a promise
    by: str                   # who promised (surface form)
    what: str                 # the commitment, one line
    deadline_text: str = ""   # "by December", "within 100 days" — raw
```

`AIAnalysis.actor_mentions: list[ActorMention]` and
`AIAnalysis.commitment_mentions: list[CommitmentMention]` (JSONB columns,
default `[]`). PROMPT_VERSION → `tvk-portrayal-v17`. Neither joins
`LABEL_FIELDS` until the flywheel needs them as training targets.

## 9. Source intelligence (◆)

Per-outlet dossier built from the source registry + observed behavior:

- **Leaning profile**: rolling stance distribution vs the corpus average —
  computed, not asserted; human can pin a curated leaning that overrides.
- **Reliability**: how often its claims end up disputed/corrected.
- **Reach**: priority from the source registry + echo behavior (who follows
  whom — does X chatter pick up this paper's frames?).
- Used to weight indices (§7) and displayed on every evidence bullet's hover
  context. Raw unweighted numbers remain visible — weighting aids judgment,
  never replaces it.

## 10. Issues, narratives, commitments — three different things

- **Issue** = ground problem (Madurai water shortage). Promoted from
  `clusters.py` themes (threshold or human). Lifecycle:
  active→watch→dormant→resolved. Carries root-cause rollup + actions.
- **Narrative** = story frame ("government can't manage law and order").
  May reference many issues; may be pushed by an actor; may be false.
  Lifecycle: emerging→spreading→peak→decaying→dormant, with `disputed` flag
  for rumor/misinfo (per-article `verification_checklist` already exists —
  narrative dossiers aggregate verification status). Spread metrics: which
  outlets carry it, news→social echo lag, velocity.
- **Commitment** = a promise with a deadline. Extracted via v17, deduped into
  `commitments` rows, tracked to kept/stalled/broken. **Both directions**:
  ours (deliver before opposition notices slippage) and rivals' (their broken
  promises are talking points — the system drafts them, citing evidence).

A nightly check flags: commitments past deadline with no delivery evidence →
`stalled`; opposition narrative forming around a stalled commitment →
escalation alert.

## 11. Early warning (◆ — the WARN stage)

- **Spike detection**: per entity/issue/district, mentions and negative share
  vs trailing 4-week baseline; z-score above threshold → alert into Home +
  the existing dashboard alert rail (`select_priority_alerts` pattern).
- **Velocity**: time from first newspaper story → first X echo → YouTube
  segment; fast cross-channel echo = escalation precursor.
- **Watchlists**: `Watchlists/<name>.md` — user-curated entity/issue lists
  with per-list thresholds (e.g. "anything communal in southern districts at
  any severity").
- Alerts are vault notes AND dashboard JSON — same data, two surfaces.

## 12. Synthesis (the "understanding" layer)

Links give navigation; synthesis gives comprehension. Via the existing Ollama
path (`settings.ollama_model`, zero API tokens):

1. **Nightly current-state refresh** — for each dossier with new evidence,
   rewrite only the `<!-- synthesis -->` block from: previous block + new
   evidence bullets. Grounding contract: only what evidence supports, cite
   `^raw:<id>` per sentence, mark unverified items, **flag contradictions
   between sources explicitly instead of averaging them**.
2. **Weekly Diagnosis Report** — the "what went wrong & how to fix"
   deliverable: top deteriorating issues (§7 heuristic + escalation risk),
   each with rolled-up root causes, deduped recommended actions + owners,
   risk-if-ignored, related narratives and commitments; plus a what's-working
   section and a coverage-vs-delivery gap section (§16). Written to
   `Reports/` and `reports/generated/`.

## 13. The response loop (◆ — ACT and MEASURE)

What "alter things if something goes bad" means in practice:

1. Analyzer already emits `recommended_step`/`action_owner`/`action_priority`
   per article → deduped into `action_items` attached to issues.
2. Humans update status (proposed→assigned→done) — dashboard widget later;
   CLI/vault edit first.
3. **MEASURE pass** (nightly): for every action `done` with `acted_on` date,
   compare the issue's sentiment/severity trend before vs after; write the
   verdict into `outcome_note` and the issue dossier ("acted on June 3;
   negative share fell 4 weeks straight" — or "no measurable recovery").
4. Outcomes accumulate into `Playbooks/<issue-type>.md`: which responses
   actually moved sentiment for which issue types. **The playbooks learn.**
5. Message library: `talking_points`/`draft_statement_*` (already per-article)
   aggregate into party/issue dossiers with performance notes.

This closes the loop: the system doesn't just recommend — it remembers
whether its recommendations worked, and its advice improves.

## 14. Reference data layer — demographics & ground truth (honesty rules)

- **Public aggregate data only** — Census of India, published caste survey
  results, Election Commission statistics, government records. Stored in
  `configs/reference/districts.demographics.yaml`, each figure with `source`
  and `as_of`. Rendered read-only into district dossiers.
- **No "live" caste feed exists** — composition changes on census timescales.
  What IS live: news/social sentiment per district, juxtaposed against the
  static composition for context.
- **Aggregate-only, by design.** District/constituency percentages are
  standard psephology. No individual-level caste inference, no profiling of
  named persons, no auto-attribution of sentiment to a community from news
  text. Extends the platform's existing non-goals.
- ◆ **Civic/economic indicators** (V6): public statistics that drive
  sentiment — commodity prices, rainfall/agriculture, power-cut data,
  unemployment — as reference time series in district dossiers, so the eye
  sees *why* sentiment moves, not just that it moved.
- ◆ **Commissioned surveys** (future): when the party runs ground surveys,
  results ingest as a reference layer — the calibration for media-derived
  sentiment.

## 15. Electoral intelligence (◆ — the "win elections" layer)

All from public data; aggregate only:

- **Constituency dossiers** (234 ACs): AC→district mapping, sitting MLA
  ([[person]] link), ECI results history (vote shares, margins, turnout,
  swing per election), alliance-era vote-transfer patterns, demographic
  context from §14, **issue salience** (which tracked issues dominate this
  AC's coverage), local body results.
- **Candidate assessment**: person dossiers double as candidate dossiers —
  public track record, portrayal trend in their region, baggage (reported
  cases/controversies, evidence-cited), promise-keep rate.
- **Winnability view** (later, after V5 data exists): margin trend +
  issue alignment + alliance arithmetic + incumbency factors — presented as
  factors with evidence, **not a black-box probability**; analysts judge.
- **Campaign mode** (election season): daily rally/coverage tracking, message
  performance from §13's library, rival promise/attack monitoring, booth-level
  *result* analysis post-election (public ECI data) to validate models.
- Honesty: media+social sentiment is a proxy; surveys calibrate; no
  individual voter data exists or is wanted (see non-goals).

## 16. Coverage-vs-delivery gap (◆)

V6 ingests official public feeds — TN DIPR press releases, scheme dashboards,
assembly proceedings/questions (all public government data):

- "Government delivered X" (official) vs "media covered X" (corpus) →
  **under-communicated wins** (delivered, never covered → comms opportunity)
  and **over-exposed failures** (one incident, 40 stories → narrative
  response, §10).
- Department dossiers gain a delivered-vs-covered section; the weekly
  Diagnosis Report gets a "free wins" list.

## 17. AI consumption — how this becomes "memory"

- **Dossier-first retrieval for the chatbot**: query → alias match → load
  dossier current-state + recent evidence (exact, cheap) → augment with
  vector RAG for verbatim quotes. Embeddings answer "find similar text";
  dossiers answer "what do we know about X over time".
- **Vault notes indexed into RAG**: synthesized sections chunked under
  `chunk_version='vault-v1'` so semantic search covers accumulated knowledge.
- **Agents enter through Home.md and Watchlists** — the same war-room page
  humans use is the agent's orientation page.
- `_meta/CONVENTIONS.md` doubles as a Claude skill so any agent writing notes
  follows the schema.
- Honest framing: **external, retrievable memory** — not infinite context.
  The win is that knowledge survives sessions and accumulates.

## 18. Pipelines & scheduling

```
pipelines/build_vault.py
  --resolve      # alias-resolve new analyses → item_entities (+ commitments dedupe)
  --aggregate    # weekly rollups + indices + early-warning scores
  --render       # upsert markdown (idempotent, deterministic ordering)
  --synthesize   # nightly LLM refresh (only dossiers with new evidence)
  --measure      # action-outcome + commitment-deadline checks
  --full         # all of the above
```

- Runs after the daily pipeline; added as a step in `tnmi/flywheel.py`'s
  nightly pass (idempotent, mirrors flywheel discipline).
- `Settings.vault_dir: Path = Path("vault")` (env-overridable).
- Vault **committed to git**: dossier history is a time machine; human notes
  and reference data must be versioned. Diff noise controlled by
  deterministic rendering, 90-day evidence caps, monthly `Archive/` rolloff.

## 19. Phased delivery

| Phase | Scope | Done means |
|---|---|---|
| **V0 — Entity registry** | `entities`/`entity_aliases`/`item_entities` + storage records; seed YAML (districts via `districts.py`, parties, depts, **sources**, people roster); `tnmi/resolver.py`; backfill CLI | Backfill resolves ≥90% of `political_actors` surfaces; candidates queued, not dropped; bilingual alias tests green |
| **V1 — Vault builder** | render People/Parties/Sources/Districts/Departments/Schemes dossiers; evidence logs + wikilinks; **Home.md + Calendar.md**; `_meta/CONVENTIONS.md` + Claude skill; briefings relocated | Opens in Obsidian; graph shows real cross-links; double regeneration → zero diff; human-section preservation tested |
| **V2 — Issues & narratives** | `issues`/`item_issues`/`narratives`/`item_narratives`; promotion from `clusters.py`; lifecycles; spread/echo metrics; disputed flag | Recurring stories become single nodes with timelines; narrative momentum computed; dormant/resolved transitions work |
| **V3 — Actor intel & promises** | prompt v17 (`actor_mentions` + `commitment_mentions`) + mock parity; reanalyze backfill; activity feeds, scorecards, relations; `commitments` tracking both directions | "Who is doing good/bad this month" answerable from any person note; stalled/broken promises flagged with evidence |
| **V4 — Synthesis, warning, response loop** | nightly current-state refresh; indices (§7); spike/velocity alerts + watchlists; weekly Diagnosis Report; `action_items` + MEASURE pass + playbooks | Diagnosis names deteriorating issues with causes + actions; no uncited claims; an action marked done gets an outcome verdict within 4 weeks |
| **V5 — Ground & electoral layer** | demographics reference YAML; `Constituencies/` + ECI history + issue salience; candidate views; chatbot dossier-first retrieval; vault-v1 RAG chunks | District dossiers show composition-vs-sentiment; constituency dossiers answer "what decides this seat"; chatbot cites dossiers |
| **V6 — Source expansion & gap analysis** | official feeds (DIPR releases, scheme dashboards, assembly Qs); Instagram/FB (SourceType exists); comments mining (aggregate); civic/economic indicators; survey ingestion; coverage-vs-delivery reports | Weekly report includes "free wins" (delivered, uncovered) and indicator-correlated sentiment context |

Each phase ships with tests (pytest, mock-analyzer path stays green per
`QUALITY_BAR.md`), is idempotent, and works offline (no required cloud calls
except optional synthesis quality).

## 20. Non-goals

- No per-article notes in the vault (DB + dashboard already serve that).
- No live demographic feeds; no individual-level caste/identity inference; no
  citizen profiling, lists, or surveillance — **aggregate public data only**
  (restates the platform's founding non-goals).
- No individual voter targeting; electoral analytics stay at
  booth/AC/district aggregate level using public ECI data.
- No disinformation tooling: the narrative module *detects and responds to*
  frames with evidence-cited counter-messaging; it does not fabricate or
  astroturf.
- No claim that media+social sentiment equals ground truth — it is a
  channel-tagged proxy; surveys calibrate it.
- No black-box winnability scores — factors with evidence, humans judge.
- Vault never overrides DB facts; synthesis never invents uncited claims.
- Obsidian the app stays optional for humans — pipelines depend only on
  plain files + git.
