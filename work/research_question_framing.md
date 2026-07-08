# Research Question and Provisional Lane

**Assignment:** ML-02 · Week 1 · Setup
**Author:** Emir Çabalak
**Status:** Provisional — lane may be confirmed or changed by the end of Week 4.

---

## 1. Provisional lane

**Lane 2 — Refresh / Content Opportunity Scoring** (one of the four core lanes).

I am *not* choosing an advanced / mentor-gated lane. For reference, the two advanced lanes
(A1 — AI Referral Opportunity, A2 — Growth/Recovery/Momentum Prediction) require mentor approval
of the data contract, validation plan, and public-output rules before any modeling; they are out
of scope for this provisional framing.

Why Lane 2: my Week-1 run already lives here. The starter pipeline
(`scripts/01`–`05`) produces a ranked refresh queue with reason codes and a Precision@K
evaluation, and in `notebooks/02` I compared a transparent hand rule (`stale × visible`) against
a small decision tree on that same task. So I already have a baseline, an evaluation metric, and a
first observation to build on — the lazy, honest continuation is to sharpen the question this
pipeline is already answering, rather than start a different lane from zero.

## 2. The question

> Among pages that are already **visible** in search, **which ones should a content editor
> review first for a refresh** (update / expand / protect / prune / monitor), so that limited
> editing time goes to the pages where a refresh is most likely to matter?

This is a **prioritization / ranking** question, not a "does X cause Y" question.

## 3. Unit of analysis

One **content page** (`content_id`) over its trailing performance window (the 30/90-day
columns in the starter dataset; warehouse-shaped daily facts later if the lane needs stronger
time-window labels). One row in, one priority score out.

## 4. Output

A **ranked review queue**: page → priority score → suggested action → reason codes →
confidence label. Concretely the shape the pipeline already emits in `outputs/refresh_queue.csv`,
but with the question, label, and reason codes made explicit and defensible.

## 5. The decision this informs, and the action someone takes

- **Decision:** which pages land at the top of an editor's weekly refresh worklist.
- **Actor:** a content editor / SEO owner with time to review only a handful of pages per week.
- **Action:** open the top-ranked pages and act on the reason code — refresh a stale-but-visible
  page, expand a thin one, protect a declining high-value one, or simply monitor. The score
  orders the queue; it does not auto-edit anything. A human still decides per page.

## 6. Cost of a wrong recommendation

Getting this wrong is not free, which is exactly why ordering matters:

- **False positive** (page ranked high but a refresh won't help): wasted editor hours, and risk
  of *degrading* a page that was fine — re-editing a stable, well-performing page can lose
  existing rankings.
- **False negative** (a page that genuinely needed attention buried low in the queue): a slow,
  invisible decline that no one gets to in time.
- Because editor time is the scarce resource, the practical cost is mostly **opportunity cost** —
  the queue's *ordering* (Precision@K near the top) matters more than a global accuracy number.

## 7. Why this is not just "train a model"

- **The label is a judgement, not a fact.** "Should be refreshed" is not recorded in the data;
  I have to *define* a leakage-safe proxy (e.g. declining-with-demand, or stale-and-visible) and
  argue it is reasonable. Choosing and defending that proxy is most of the real work — a model
  can only rank as well as the label lets it.
- **Leakage is the main danger.** In `notebooks/02` a tree that was allowed to see `trend_pct`
  (a future-window signal) scored near-perfectly and was useless — it had peeked at the answer.
  The honest pipeline validates with **client-holdout** so a client's pages never sit in both
  train and test. Guarding against this matters more than model choice.
- **A transparent baseline may be enough.** The hand rule was competitive with the tree at some
  cutoffs. If a readable rule ranks nearly as well, that is the better product: an editor can see
  *why* a page is on the list. The model has to *earn* its extra complexity, and only on
  **held-out** data.
- **The output is a recommendation for a human, not an automated action.** The deliverable is a
  prioritized, explainable queue that a person overrides freely — not an autonomous system.

## 8. Cautious framing (public-safety)

- Everything here is **observational and directional**. Signals in this anonymized starter data
  are *associated with* outcomes; I will not claim any of them is a proven Google ranking factor.
- Findings will be reported as *observed on this sample*, with effect sizes and clear caveats,
  never as "we proved the algorithm."
- No client data or unsafe fields leave the analysis; only the anonymized starter dataset (and,
  later, a mentor-approved release) is used.

## 9. First evidence (Week-1, already in this repo)

From `notebooks/01`–`02` on the 30,000-row anonymized starter sample:

- Keyword `search_volume` was **near-uncorrelated** with actual `impressions_90d` — "high search
  volume ⇒ more traffic" did not hold here. Demand and delivered visibility are different things,
  which is why a refresh queue should key off *delivered* performance, not intent volume.
- Holding **position tier fixed**, most of the apparent CTR gap between content types shrank —
  a reminder to position-adjust before reading CTR as a page-quality signal.
- On a proper train/test split, a small decision tree still out-ranked the hand rule out-of-sample,
  but the readable rule stayed competitive — supporting a "transparent-baseline-first" plan.

*Directional observations on a sample, not causal claims.*
