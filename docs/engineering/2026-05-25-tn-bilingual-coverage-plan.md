# Tamil Nadu Public Media Intelligence — Bilingual Full-Coverage Plan

**Status:** Living plan, executed in phases  
**Owner:** TVK Office of the Leader  
**Last updated:** 2026-05-25

---

## North Star

A confidential public-media intelligence platform that monitors **every relevant
Tamil and English source covering Tamil Nadu**, in both languages, end-to-end:
ingestion, analysis, briefing, and report — so the office of the party leadership
gets the same calibre of intelligence that proprietary tools (Brandwatch, Meltwater,
Sprinklr) sell to US campaigns for $50–100k/year — built on open-source primitives
we own and audit.

Not a SaaS. Not a public dashboard. A private operations tool.

---

## Core principles

1. **Tamil-first, English-equal.** Every operator screen, every AI output, every
   PDF report is available in both languages. The user picks. The AI never
   forces a translation when the source is already Tamil.
2. **Source neutrality.** We monitor every major newspaper, X account, and
   YouTube channel that covers TN public affairs — regardless of political
   alignment. The chief is briefed on the full picture, not a partisan slice.
3. **Evidence preserved.** Every analysis line is backed by a verifiable
   newspaper-quote URL. Nothing is invented.
4. **Local-first AI.** OpenAI is the preferred analyser when available, but the
   product must keep working when OpenAI is unavailable. We run AI4Bharat
   IndicBERT and IndicTrans2 locally as the fallback.
5. **Open-source only.** Every dependency is auditable. No proprietary black
   boxes in the analysis pipeline. Visflow / Multiply company tools are the
   only exception (UI shell, hosting).
6. **Confidential by construction.** Operator token guards, no public routes,
   no third-party trackers, all data on private infra.

---

## Architecture (target)

```
                    ┌──────────────────────────────────────────┐
                    │           SOURCE REGISTRY                │
                    │  (Media Cloud + curated + GDELT cross)   │
                    └──────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
     ┌────────────┐            ┌────────────┐            ┌────────────┐
     │ Newspapers │            │ X / Social │            │  YouTube   │
     │  News-     │            │  tweepy +  │            │   yt-dlp + │
     │  Please /  │            │  X API     │            │  Whisper + │
     │  trafilatu │            │            │            │  PaddleOCR │
     │  ra        │            │            │            │            │
     └─────┬──────┘            └──────┬─────┘            └─────┬──────┘
           │                          │                        │
           └──────────────┬───────────┴────────────────────────┘
                          ▼
                ┌─────────────────────┐
                │  RAW EVIDENCE LAKE  │  ← Postgres + S3 (object: HTML, audio, video, frames)
                │  (raw_items)        │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────────────────────┐
                │       LANGUAGE & NORMALISATION      │
                │  detect_language, IndicTrans2,      │
                │  language-pair: original ⇄ English  │
                └──────────┬──────────────────────────┘
                           ▼
                ┌─────────────────────────────────────┐
                │           AI ANALYSIS               │
                │                                     │
                │  Primary  : OpenAI GPT-5.4-mini     │
                │  Fallback : AI4Bharat IndicBERT     │
                │             (local Tamil model)     │
                │  Demo     : MockAIAnalyzer          │
                │                                     │
                │  Output: stance, relevance,         │
                │  summary_en, summary_ta, party,     │
                │  people, why, next step, evidence   │
                └──────────┬──────────────────────────┘
                           ▼
                ┌─────────────────────────────────────┐
                │   RETRIEVAL + CLUSTERING            │
                │  embedding (OpenAI or local) →      │
                │  turbovec/InMemory index →          │
                │  recurring-theme detection          │
                └──────────┬──────────────────────────┘
                           ▼
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
  ┌──────────┐                       ┌────────────────┐
  │ Dashboard│                       │ Distribution   │
  │  (TA/EN) │                       │  PDF, email,   │
  │  cards + │                       │  WhatsApp,     │
  │  table + │                       │  API, alerts   │
  │  trends  │                       │                │
  └──────────┘                       └────────────────┘
```

---

## Open-source dependency matrix

| Layer | Tool | Role | Status |
|---|---|---|---|
| Article extraction | **trafilatura** | HTML → clean text | ✓ integrated |
| Multi-source crawler | **News-Please** | 100s of newspaper layouts | ⏳ Phase B |
| Source registry | **Media Cloud** | 70k+ curated news sources | ⏳ Phase B |
| Global cross-reference | **GDELT** | Tone / theme / event tags worldwide | ⏳ Phase E |
| Tamil/Indic NLP | **AI4Bharat IndicBERT** | Sentiment, stance, NER (Tamil) | ⏳ Phase C |
| Translation | **IndicTrans2** | Tamil ⇄ English | ⏳ Phase C |
| Audio → text | **faster-whisper** | Tamil ASR | ⏳ Phase D |
| Frame OCR | **PaddleOCR** (Tamil + Eng) | Video frame text | ⏳ Phase D |
| Video pull | **yt-dlp** | Authorised channel pull | ⏳ Phase D |
| Embeddings | OpenAI **text-embedding-3-small** + local bag-of-words | Theme clustering | ✓ integrated |
| Vector index | **turbovec / In-memory** | Similarity / retrieval | ✓ integrated |
| Orchestration | FastAPI + BackgroundTasks (now), **Airflow** later | Pipelines | ✓ partial |
| OSINT inspiration | **OpenCTI architecture**, **SpiderFoot modules** | Connector pattern | ⏳ reference |
| AI primary | **OpenAI GPT-5.4-mini** / reports w/ **GPT-5.5** | Analysis | ✓ integrated, fallback to mock |
| UI shell | Inter / Font Awesome Pro / VDOE tokens | Frontend | ✓ integrated |

---

## Phase plan

Each phase is **~1 week** with a working dashboard at the end. We don't
"start everything at once."

### Phase A — Bilingual UI + bilingual AI output

**Goal:** Every screen and every AI line is available in English **and** Tamil.

- [ ] Top-right language toggle (EN ⇄ TA) persisted in cookie
- [ ] Translate all UI strings: kickers, labels, navigation, footers, buttons,
      filter dropdown options
- [ ] AI prompt change: produce both `summary_english` **and**
      `summary_original_tamil` for Tamil articles (and vice versa)
- [ ] Briefing lines (Party / People / Why / Next step) populated in **both**
      languages — choose at render time based on toggle
- [ ] All UI copy stored in `apps/api/i18n/{en.json, ta.json}` with a small
      `t()` helper in Jinja
- [ ] Tamil Noto Sans + Anek Tamil fonts already in place — just route through
      the toggle

### Phase B — Full Tamil Nadu source coverage

**Goal:** Every district has ≥ 1 source. Every major newspaper, English + Tamil.

- [ ] Integrate **Media Cloud API** — pull all `india` + `india_tamil_nadu`
      collection sources into our registry
- [ ] Curated additions to fill gaps (Daily Thanthi, Dinakaran, Maalai Malar,
      Dinamalar, Hindu Tamil Thisai, Dinamani, Puthiyathalaimurai web, Vikatan
      group, Polimer News, Sun News, Hindu English Chennai, Times of India
      Chennai, Deccan Chronicle, Indian Express Chennai, The Federal, Cauvery
      News, News18 Tamil Nadu, etc.)
- [ ] Adopt **News-Please** as the long-tail scraper for sites without RSS
- [ ] Source registry stores: name, language, district focus, RSS/sitemap URL,
      Media Cloud media_id, priority
- [ ] Dashboard filter: "by district" (38 districts), "by language"

### Phase C — Native Tamil NLP + offline-safe analysis

**Goal:** The product works **without** OpenAI. Tamil is analysed natively.

- [ ] Add a `LocalTamilAnalyzer` that uses AI4Bharat IndicBERT for
      stance/relevance/sentiment, and IndicTrans2 for English summary
- [ ] Fallback chain in `_FallbackAnalyzer`:
      `OpenAI → LocalTamilAnalyzer → MockAIAnalyzer`
- [ ] Add `tnmi.local_models` package that lazy-loads Hugging Face models on
      first use; cached on disk
- [ ] Confidence score: OpenAI = 0.9, Local = 0.75, Mock = 0.5 — surface in UI
- [ ] Operator-settings page shows which analyser ran each run

### Phase D — X (Twitter), YouTube, multimedia

**Goal:** Beyond newspapers — capture public political conversation across media.

- [ ] X ingestion: tweepy with official X API, configured for ~500 TN-relevant
      handles (politicians, journalists, party accounts, district admins).
      Existing `x_ingestion.py` is the base.
- [ ] YouTube ingestion: official YouTube Data API to discover recent uploads
      from TN news channels (Polimer, Sun News, Puthiya Thalaimurai, Thanthi
      TV, Pudhuyugam, News7 Tamil, etc.)
- [ ] yt-dlp to pull authorised clips, faster-whisper for Tamil transcripts,
      PaddleOCR on sampled frames for banner/poster text
- [ ] X posts and video transcripts flow through the same AIAnalysis pipeline
      as newspaper articles — same stance, same briefing lines

### Phase E — Cross-reference + theme intelligence

**Goal:** Spot patterns the chief can't see on a single dashboard.

- [ ] **GDELT** integration: query their REST API for any TN article we ingest,
      annotate with global theme tags, tone, and "similar global coverage"
- [ ] Theme detection upgrade: switch from bag-of-words to real OpenAI
      embeddings (or local sentence-transformers fallback)
- [ ] Time-series stance trend: per-district, per-newspaper, per-topic
- [ ] Coordinated-coverage detection: when ≥ 3 newspapers run very similar
      stories within 24 hours, flag for the chief
- [ ] Narrative tracking: each theme has a 7-day momentum score

### Phase F — Distribution

**Goal:** The chief reads the briefing in the format they prefer.

- [ ] One-click **PDF brief**: generated from current dashboard state,
      bilingual, official-letterhead style
- [ ] **Email**: scheduled daily 7am IST send to a small list
- [ ] **WhatsApp / Telegram** bot for top-3 stories of the day (read-only)
- [ ] **API**: signed-URL endpoint other party apps (district commanders) can
      poll for their slice
- [ ] **CSV / Excel** export of the table view for analysts

### Phase G — Confidentiality + audit

**Goal:** Operations-grade controls.

- [ ] SSO / OAuth login (Google Workspace or party-issued)
- [ ] Role-based access: leader (full), district leader (own district),
      analyst (review queue only), auditor (read-only)
- [ ] Audit log: who saw what when, who marked-reviewed, who escalated
- [ ] Encrypted at rest (DB + S3), TLS only
- [ ] No third-party trackers, no public assets (host Font Awesome locally,
      remove Google Fonts CDN, ship Inter as a static font)
- [ ] Daily backup, 30-day retention, off-site copy

---

## Source coverage matrix (target by end of Phase B)

**English newspapers covering TN:**
- The Hindu (Chennai), Times of India (Chennai), Deccan Chronicle, Indian
  Express (Chennai), The Federal, News Minute, Hindustan Times (TN bureau)

**Tamil newspapers — daily:**
- Daily Thanthi, Dinakaran, Dinamalar, Dinamani, Maalai Malar, Maalai Sudar,
  Hindu Tamil Thisai, Tamil Murasu, Makkal Kural, Theekkathir

**Tamil weeklies / magazines:**
- Ananda Vikatan, Kumudam, Junior Vikatan, Kungumam, Nakkheeran, Puthiya Thalaimurai

**Tamil TV / web news channels:**
- Sun News, Polimer News, Puthiya Thalaimurai, Thanthi TV, News7 Tamil,
  Pudhuyugam, Lotus News, Captain News, Behindwoods Tamil

**X (Twitter):**
- All TN MPs/MLAs (~234 MLAs + 39 MPs)
- District collectors and police chiefs
- Major party handles (TVK, DMK, AIADMK, BJP, Congress, etc.)
- TN beat journalists (~50-100)

**YouTube:**
- Top 20 TN news YouTube channels with daily upload cadence

**Total target:** ~600–800 ongoing sources monitored daily.

---

## Bilingual language strategy (Phase A detail)

The UI runs in one of two modes — toggle in the top-right, sticky cookie:

- **EN mode** — every label, button, kicker, filter, footer in English.
  Article headlines + Tamil evidence quotes remain in their original language
  (Tamil if the source was Tamil). English AI summary shown.

- **TA mode** — every label, button, kicker, filter, footer in Tamil.
  Article headlines + evidence quotes remain in their original language.
  Tamil AI summary shown.

**AI output table:**

| Field | EN mode shows | TA mode shows |
|---|---|---|
| Headline | original (Tamil or Eng) | original |
| Summary | `summary_english` | `summary_tamil` |
| Party action | English | Tamil |
| People impact | English | Tamil |
| Why | English | Tamil |
| Next step | English | Tamil |
| Evidence quote | original (always Tamil if Tamil article) | original |
| Source name | original | original |

Both `summary_english` and `summary_tamil` are produced by the AI from the
same source article. If the article is Tamil, `summary_tamil` is the AI's
edited Tamil version; `summary_english` is the AI's English rendering.

---

## Confidentiality (recap of Phase G)

- Hosted in a private VPC (Visflow infra) — no public internet routes
- Operator token guard on every endpoint (already in place)
- No telemetry, no analytics, no third-party scripts
- All open-source dependencies vendored at known versions, audited quarterly
- Encrypted backups in a separate cloud account

---

## Success metrics (90-day)

| Metric | Target |
|---|---|
| Newspaper sources monitored daily | ≥ 50 |
| X handles monitored daily | ≥ 500 |
| YouTube channels monitored daily | ≥ 20 |
| Articles ingested per day | ≥ 1500 |
| AI analyses per day | ≥ 1500 |
| Briefing pages: time-to-load | < 1.5s (cached) |
| Briefing pages: time-to-fresh-data | < 4 minutes from publish |
| Languages supported | Tamil + English, fully |
| Coverage gap (districts with 0 sources) | 0 |
| Days OpenAI was unavailable but product still worked | 100% |

---

## What we already have (today)

- Editorial briefing dashboard with VDOE token system
- 5 selectable filter cards (All / Positive / Negative / Mixed / People Issues)
- 4-line briefing per article (Party / People / Why / Next step)
- Recurring Themes panel (clustered narratives)
- Trends panel (14-day stance over time, department + district breakdown)
- People Issues + Supportive Signals columns with Why + Next step
- Per-card actions (Mark reviewed / Escalate) → review_decisions table
- Pull Latest button → background ingest with fallback analyser
- Cards + Table view toggle
- Day picker + range filter + source filter + search
- Operator token guard on every endpoint
- 93-test pytest suite, CI-ready

We're already on the path. This plan is about how to evolve it into the
election-grade platform you described.

---

## Next concrete step

**Phase A, sub-task 1:** Add the bilingual UI toggle. Smallest visible change,
highest leverage. Then we work down the list.

I'll start there unless you want me to start somewhere else.
