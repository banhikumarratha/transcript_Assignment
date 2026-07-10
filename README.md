# Transcript Intelligence — Take-Home Solution

A pipeline that ingests ~100 call transcripts across three call types
(**support**, **external**, **internal**), categorizes them by topic, analyzes
sentiment, and surfaces stakeholder insights.

> **On the data:** the assignment's sample dataset was not provided to me, so
> per the FAQ ("you're welcome to generate synthetic data… just be transparent")
> I generated **99 synthetic transcripts (33 per call type)** with a seeded,
> reproducible generator. The transcripts deliberately bake in a few realistic
> patterns (billing/outage calls skew negative; internal incident calls echo
> themes that appear in support) so the analysis has real signal to find. In a
> live setting the exact same pipeline runs on the real transcripts unchanged —
> only the loader cell points at a different folder.

## How to run

```bash
pip install google-genai pydantic scikit-learn pandas matplotlib jupyter
# put your key in transcript_intelligence/.env  ->  GEMINI_API_KEY=...
jupyter notebook   # then run the three task notebooks top to bottom
```

The notebooks are **Gemini-powered** (Google `gemini-2.5-flash`) via the official
**`google-genai` SDK with structured output** — each LLM call passes a **Pydantic
`response_schema`**, so Gemini is constrained to return valid, typed JSON (no
markdown fences, no wrapper objects, no manual JSON repair). Requires
`GEMINI_API_KEY` in `.env`. **Run the notebooks in order (1 → 2 → 3)** — they
chain through saved files in `output/`: notebook 1 loads the transcripts and
classifies them (→ `output/labeled.csv`), notebook 2 adds sentiment
(→ `output/scored.csv`), and notebook 3 reads that and builds the insights.
Loading and the classify/score LLM calls happen once, in notebooks 1 and 2, so all
three stay consistent (`data/` stays source-only). Notebooks ship **unexecuted** —
run them to populate outputs and charts.

**Batched LLM calls.** To stay under Gemini's per-minute rate limit, all
transcripts are sent in a **single batched request** per step (the schema is
`list[Model]`, keyed by call ID). So categorization is one call, sentiment is one
call — not ~100 each.

## Project layout

```
transcript_intelligence/
├── data/
│   ├── transcripts/{support,external,internal}/*.txt   # raw transcripts (99)
│   └── metadata.csv                                    # index + ground-truth seeds
├── 01_task1_categorization.ipynb   # Task 1 — Gemini topic categorization
├── 02_task2_sentiment.ipynb        # Task 2 — frustration-aware sentiment
├── 03_task3_insights.ipynb         # Task 3 — stakeholder insights
├── outputs/                        # exported charts + CSVs
└── README.md
```

The synthetic-data generator is preserved as a guarded appendix cell at the
bottom of notebook 1 (`if False:` so it won't overwrite the saved data).

---

## Task 1 — Topic categorization (Gemini)

**Approach: LLM classification into a fixed taxonomy.**

- **Gemini** classifies each transcript into a curated taxonomy (Billing, Auth,
  API, Reliability, Data, Adoption, Renewal, Product Feedback), returning a
  category and a one-line reason. An LLM reads nuance and generalizes zero-shot far
  better than keyword rules.
- The per-call **reason** makes each label auditable, and the notebook prints one
  example transcript per topic to confirm the labels look right.

Category counts and examples render inline in notebook 1.

## Task 2 — Sentiment across call types

**Model: Gemini.** Each call is read by the LLM, which returns the customer's
overall sentiment (label + score) plus how they sounded at the **start vs end**
and their single most negative **low point** — context a lexicon can't capture.
Two design decisions matter:

1. **Score the customer voice.** We pass only the customer's utterances, so the
   agent's politeness doesn't inflate the read.
2. **Track the swing and the peak.** From the open/close scores we derive the
   open→close **swing** (`sentiment_delta`), and the low point gives a
   **frustration peak** — a call with a sharp negative moment isn't "positive"
   just because it ends courteously.

**Validation:** scores track the generator's ground-truth sentiment seeds
monotonically (neg < mixed < pos), which gives confidence the scorer works.

**What the trends mean:**

- **External calls are the most positive; support the least; internal the most
  variable** (and home to the only outright-negative calls). Renewal/expansion/
  QBR conversations are relationship-building, so positivity is expected — but
  it can mask risk (see churn radar). Support sits lowest because people call
  when something's broken.
- **Frustration peaks are deepest in support and internal-incident calls.** The
  average sentiment looks fine; the *peak* is where the pain hides. A leader
  watching only averages would miss it — which is the argument for tracking the
  peak metric operationally.
- **Internal calls carry the negatives that customers never hear** — incident
  postmortems and escalations. That's a healthy signal (problems surface
  internally) but also a place to watch for morale/on-call strain.

Charts: `02_sentiment_by_type.png`, `03_sentiment_mix.png`,
`05_sentiment_over_time.png`.

## Task 3 — Additional insights

**1. Cross-tier issue propagation (Eng leads / PMs).** Do themes that spike in
*support* resurface in *internal* calls, and how long is the lag? Every major
support theme (billing, auth, API, reliability) also appears in internal
discussions — but if the lag is weeks, customers are hurting well before
engineering formally picks it up. Shrinking that lag is a concrete
early-warning win. → `insight_cross_tier_propagation.csv`

**2. Churn-risk radar (Sales / CS leaders).** Ranks external accounts by a score
combining negative sentiment with risk language ("evaluating alternatives",
"shook our confidence", "move on"). Lets CS intervene *before* renewal instead
of reacting to a cancellation. The top hits are all "Churn Risk & Escalation"
calls — the radar correctly floats them. → `insight_churn_risk.csv`

**3. Topic × sentiment hotspots (Product leaders).** The intersection of *high
volume* and *low sentiment* is where product investment buys the most goodwill.
A topic can be common but fine (how-to) or rare but toxic (outages); the hotspot
is both. Reliability and API topics rank high — painful *and* frequent. →
`insight_topic_hotspots.csv`, `04_topic_hotspots.png`

**4. Agent / AM coaching signal (described, not implemented).** Using the
`sentiment_delta` (open→close swing) we already compute, we could rank *how much
each rep improves customer sentiment over a call*, normalized for topic
difficulty. Support leaders get an objective coaching metric ("who consistently
turns hard calls around?"); AMs get a read on which conversations recover and
which spiral. Building it needs consistent rep IDs on each transcript, which our
synthetic set doesn't model richly — hence described, not coded — but the
`sentiment_open/close/delta` columns are already in the output, so it's a short
hop.

---

## Notes & honest limitations

- Synthetic data is template-driven, so vocabulary is narrower than real calls.
- Classification and sentiment both depend on Gemini, called fresh on every run;
  results can vary slightly run to run.
- The dataset is seeded (`SEED = 42`) and reproducible end-to-end.
