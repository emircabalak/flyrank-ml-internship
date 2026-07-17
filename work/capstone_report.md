# Capstone Report: Lane 2, Refresh / Content Opportunity Scoring

- **Author:** Emir Çabalak
- **Lane:** Lane 2, Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/emircabalak/flyrank-ml-internship
- **Date:** 2026-07-18 (in progress, fill fully in weeks 7-8)

> Working draft. Sections keep the template's prompts as guidance; the two blockquote notes
> under sections 3 and 5 are captured findings to fold into the final paper so they are not
> rebuilt from memory. Numbers here must match a fresh re-run of the notebooks.

## 0. Abstract

Five sentences, written last, placed first: question → data → method → headline result →
what the output is for. This is the top of your deployed paper.

## 1. Problem framing

What decision does this support? Name the unit of analysis (page, client, day…), the output
(score, rank, cluster, report), the action a human takes from it, and the cost of a wrong
call. Why does data/ML help here at all?

Source material: `work/notebooks/w01_research_question.ipynb` (ML-02) and
`w02_ml_task_framing.ipynb` (ML-03). Task = ranking/scoring, one row = one page, output = a
ranked refresh queue, action = an editor works the top ~50 pages a month.

## 2. Data safety

Which data you used and which columns you deliberately excluded (and why). Leakage risks you
considered — especially label-derived fields (`trend_direction`, `trend_pct`) and pseudonymous
IDs (grouping only, never features). Confirm nothing client-identifying appears anywhere in
`work/`.

Source material: `w03_data_contract.ipynb` (ML-04) and the full leakage audit in
`w06_validation_audit.ipynb` (ML-09), including the escalation table and the disclosed
population gate.

## 3. Baseline

The transparent rule or score you built first. Why it's a fair comparison, and its numbers on
the same data and metric as your model.

Source material: `w04_baseline_score.ipynb` (ML-07), the Week-4 CTR-fix rule.

> **Note to fold in (captured 2026-07-18): the 24 vs 74 framing.** The Week-2 lecture tells a
> canonical story: a hand rule picked the right pages 24% of the time (12 of 50) and the model
> was about 3x better (~74%). My headline numbers do NOT match that, and the reason is a
> stronger, fairer baseline, not a weaker model. My Week-4 CTR-fix, scored as a decline ranker
> on the same client-grouped folds, reaches P@50 = 0.591, not 0.24. Against that higher bar the
> model's lift is about 1.4x (P@50 0.80-0.84), not 3x. I report it this way on purpose: a weak
> baseline manufactures a dramatic lift, and the honest comparison uses the best transparent
> rule I could build. State this explicitly so a reader who knows the 24→74 story is not
> confused by the smaller gap.

## 4. Model / analysis

Your method and why it fits the lane. The exact feature list (and what you left out on
purpose). The target or proxy definition, in one sentence.

Source material: `w05_model.ipynb` (ML-08) and the harness under `work/experiments/`. Method =
Logistic Regression over leakage-safe features, chosen over 11 other families; the one durable
improvement was widening the training population to sub-threshold pages. Target is the
`is_declining` proxy, a defined stand-in for "worth a refresh slot", never the trend columns
it is derived from.

## 5. Evaluation

Your split (grouped by client? time-aware?) and why. Metrics, model vs baseline **on the same
split**. What the errors look like — a short error analysis beats a big metric table.

Source material: `w05_model.ipynb` (client-grouped folds + 6-client lockbox) and
`w06_validation_audit.ipynb` (random vs grouped before/after, cluster bootstrap CI).

> **Note to fold in (captured 2026-07-18): decision value is not prediction quality.** What I
> measured is ranking quality: on held-out clients the model ranks declining pages above the
> 0.598 base rate (observed AUC 0.625, top-50 precision 0.80-0.84). What I did NOT measure, and
> must not claim, is decision value: whether refreshing the pages it flags actually recovers
> traffic. That is an intervention question and needs an A/B test or a refresh-vs-hold holdout,
> which this observational snapshot cannot answer. The honest limitation for the paper: this is
> decision-support for prioritising an editor's queue, not evidence that the refreshes pay off.
> Also carry the statistical caveat from ML-09: the widened-vs-visible P@50 gain of +0.08 has a
> 95% cluster-bootstrap interval of [-0.02, +0.12] over six lockbox clients, so it is directional,
> not a measured improvement. Deployment, monitoring, and governance (framework sections 19
> J/K/L) also belong in the limitations, not claimed as done.

## 6. Interpretation

What the model/clusters actually found. Feature importances or cluster profiles in plain
words. Surprises and negative results — a well-understood "no effect" is a valid result.

Source material: coefficient and per-client-AUC analysis in `w05_model.ipynb`, and the
mix-effect finding in `w06_validation_audit.ipynb` (pooled vs within-client cohort gaps).

## 7. Recommendation

The ranked actions or decisions your output supports, and how a FlyRank editor would use them
tomorrow. State your confidence and the limits explicitly.

Source material: `w07_action_playbook.ipynb`. Keep the confidence language honest: directional,
decision-support, and bounded by the two notes above.

## 8. Reproducibility

The exact commands to re-run everything from a fresh clone, your random seeds, and your
environment (`pip freeze` highlights or `requirements.txt` deltas). If you claim a sealed or
holdout evaluation, two things must be committed: the cell/script that builds the sealed
frame, and the metrics file it produced.

Source material: `work/experiments/run_experiments.py` (setup/full/verify), `lockbox.json`,
and `ledger.jsonl` are the sealed-evaluation receipts.

## 9. Acknowledgments & data credit

One short section at the bottom of the deployed paper: "Built on the FlyRank ML Internship
dataset" **linking to https://flyrank.ai**.

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> language everywhere · base rate next to every precision@K · no causal claims without an
> experiment · no "predicted Google's algorithm" · no client-identifying details · numbers
> match a fresh re-run.
