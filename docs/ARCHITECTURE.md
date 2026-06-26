# TVK Political Intelligence — Architecture

The platform is a **self-improving political-intelligence system** for Tamil
Nadu: it gathers public media, classifies it through the TVK lens, learns from
every human correction, and continuously trains a dedicated model — **without
ever being able to silently regress**.

## The big picture

```
                         ┌──────────────────────────────────────────────┐
                         │                  GATHER                       │
                         │  newspapers · YouTube (Whisper) · X · manual  │
                         └──────────────────┬───────────────────────────┘
                                            ▼
                         ┌──────────────────────────────────────────────┐
                         │            CLASSIFY (student)                 │
                         │  LLM cascade: OpenAI → local Tamil → Gemma    │
                         │  → mock · two axes: tvk_portrayal (headline)  │
                         │  + stance_toward_government · action playbook │
                         └──────────────────┬───────────────────────────┘
                                            ▼
       ┌─────────────────────────── VALIDATE (tnmi/validation.py) ─────────────┐
       │  teacher model re-judges (or confidence threshold without teacher)    │
       │      agree → SILVER label          disagree/unsure → review queue     │
       └──────────┬────────────────────────────────────┬──────────────────────┘
                  ▼                                     ▼
     ┌─────────────────────────┐          ┌──────────────────────────────┐
     │  LABEL STORE             │          │  HUMAN REVIEW (dashboard)    │
     │  (tnmi/labeling.py)      │◄─────────│  corrections → GOLD labels   │
     │  bronze / silver / gold  │          │  (active learning: humans    │
     │  + provenance + buckets  │          │   only see what AI is unsure │
     └──────────┬───────────────┘          │   about)                     │
                ▼                          └──────────────────────────────┘
     ┌─────────────────────────┐
     │  TRAIN (tnmi/training.py)│   silver+gold, MINUS the frozen gold
     │  StubTrainer (CI/dry-run)│   test buckets — we never train on
     │  PeftLoraTrainer (GPU)   │   what we measure on
     └──────────┬───────────────┘
                ▼
     ┌─────────────────────────┐
     │  EVAL (tnmi/eval.py)     │   frozen 20% of GOLD = the yardstick
     │  per-field acc/P/R/F1    │   (HELD_OUT_TEST_BUCKETS)
     └──────────┬───────────────┘
                ▼
     ┌─────────────────────────┐
     │  GATE (tnmi/registry.py) │   promote_if_better: a candidate goes
     │  model_registry table    │   live ONLY if it beats the incumbent
     └──────────┬───────────────┘   on gold. Worse models stay shelved.
                ▼
        the live "political intelligence" model
        (serves the next day's classification — loop repeats)
```

One command runs the whole loop: `python -m pipelines.flywheel`
(orchestration in `src/tnmi/flywheel.py`; safe to cron nightly — every step is
idempotent).

## Why this design cannot collapse

"Train on and on with the data it gathers" fails when a model trains on its own
unverified output: errors compound, bias amplifies, and the drift is invisible.
Four mechanisms prevent that here:

1. **Teacher ≠ student** (`tnmi/validation.py`). Silver labels require an
   *independent* model to agree (or, fallback, high student confidence). A
   model never validates itself.
2. **Humans own the yardstick** (`tnmi/labeling.py`). Only human-verified GOLD
   labels gate promotion. Review corrections are mined into gold automatically.
3. **Frozen test set** (`tnmi/eval.py`). A deterministic 20% of gold
   (`split_bucket` = sha256 of item+field) is excluded from all training, so
   eval numbers are honest by construction.
4. **The promotion gate** (`tnmi/registry.py`). `promote_if_better` keeps the
   incumbent live unless a candidate measurably beats it. With zero gold
   labels, promotion is refused outright — the system never promotes blind.

Provenance (`ai` / `ai_high_conf` / `teacher_model` / `human` / `enriched`) is
stored on every label so any poisoned source can be traced and excluded later.

## Data tiers

| Tier   | Source                                  | Used for                       |
|--------|------------------------------------------|--------------------------------|
| BRONZE | raw model output, unverified             | audit trail, candidate pool    |
| SILVER | teacher-agreed or high-confidence        | bulk training signal           |
| GOLD   | human-verified (review corrections)      | training + the **only** eval/promotion yardstick |

Stored in `labeled_examples` — one row per (item, field, tier); re-labelling a
tier upserts, so each item carries a clean bronze→silver→gold progression.

## GPU deployment path

The interfaces are GPU-ready today; the heavy stack is intentionally not a
laptop dependency.

1. **Box**: any CUDA machine (a single 24 GB GPU — e.g. RTX 4090 / A10G — is
   enough for LoRA on a 2–9 B base model).
2. **Install**: `pip install torch transformers peft datasets accelerate`.
3. **Base model**: a Tamil-capable open model — e.g. Gemma 2/3 (already used
   via Ollama here), Sarvam-1, or an IndicBERT-class encoder for pure
   classification.
4. **Train**: `python -m pipelines.flywheel --peft --base-model <hf-model>`
   — `PeftLoraTrainer` fine-tunes with LoRA on the distillation dataset
   (article → JSON label block), versioned by dataset fingerprint, saved under
   `artifacts/`.
5. **Serve**: load the adapter in the inference path (e.g. export to Ollama or
   serve via transformers) and wire it as an analyzer in the cascade; pass it
   as the flywheel's `--candidate` so the gate scores the *actual* artifact.
6. **Repeat**: each nightly pass retrains on the grown label store; the gate
   decides whether the new weights ship.

The `StubTrainer` runs the identical orchestration with no GPU, so CI and dev
machines exercise the full loop end to end.

## Module map

| Concern                  | Module                      | Key entry points |
|--------------------------|-----------------------------|------------------|
| Contracts/enums          | `src/tnmi/contracts.py`     | `AIAnalysis`, `LabelTier`, `LabelProvenance`, `LABEL_FIELDS` |
| Label store              | `src/tnmi/labeling.py`      | `record_label`, `record_bronze_from_analysis`, `promote_corrections_to_gold`, `export_dataset` |
| Eval harness             | `src/tnmi/eval.py`          | `evaluate_classifier`, `score_predictions`, `HELD_OUT_TEST_BUCKETS` |
| Validation/routing       | `src/tnmi/validation.py`    | `validate_analysis` |
| Model registry/gate      | `src/tnmi/registry.py`      | `register_model`, `promote_if_better`, `get_live_model` |
| Training                 | `src/tnmi/training.py`      | `build_distillation_dataset`, `StubTrainer`, `PeftLoraTrainer` |
| Orchestration            | `src/tnmi/flywheel.py`      | `run_flywheel`, `harvest_labels`, `train_and_gate` |
| CLIs                     | `pipelines/`                | `flywheel.py`, `train_classifier.py`, `eval_classifier.py` |

## Operating it

```bash
# Nightly (cron / DAG): full pass — harvest, train, eval, gated promote
python -m pipelines.flywheel

# After reviewers correct cards in the dashboard: mine gold + measure
python -m pipelines.eval_classifier --promote-corrections

# Inspect the label store / promotion history
python -m pipelines.flywheel --json report.json
```

The flywheel's leverage compounds with use: every reviewed card grows gold,
better gold makes the gate sharper, a sharper gate lets training run more
aggressively — while the dashboard keeps serving from whatever model is
provably best so far.
