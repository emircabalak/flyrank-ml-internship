# Which already-visible pages should an editor refresh first?

- **Author:** Emir Çabalak
- **Lane:** Lane 2, Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/emircabalak/flyrank-ml-internship
- **Deployed paper:** see `submission/paper_url.txt`
- **Reproduces from:** `work/notebooks/capstone.ipynb` (all numbers and both charts)

## 0. Abstract

FlyRank runs content for many clients and cannot hand-review every page, so editors need a short list of what to fix first. I framed that as a ranking problem on the anonymized internship dataset (30,000 pages, 32 clients): rank visible pages by how likely they are to be declining, using only signals knowable before the outcome. A Logistic Regression over leakage-safe features beats a fair Week-4 rule baseline on client-grouped folds, reaching Precision@50 of 0.81 against a 0.60 base rate, and widening the training data to sub-threshold pages helps the top of the queue. The output is a ranked refresh queue with a reason code per page, not a claim about Google or about whether a refresh pays off. It is decision-support for prioritizing an editor's week.

## 1. Problem framing

A FlyRank editor owns hundreds of live pages and has time for maybe fifty reviews a month. The hard part is not writing the fix, it is choosing which pages deserve one. Rewrite a page that was fine and you burn hours and risk rankings it already had; miss a page that is slipping and it keeps bleeding traffic. One row is one page, the output is a ranked queue with a reason code and an action, the actor is an editor working the top of the list, and ranking is worth automating because the decline signals are many and tangled. The final call still belongs to a person.

Source: `w01_research_question.ipynb`, `w02_ml_task_framing.ipynb`.

## 2. Data safety

Model, baseline, and validation run on the 30,000-row anonymized starter slice committed to the repo (32 clients, trailing 90-day window), so anyone can rerun without gated access. The data contract verified the same lane on the full warehouse release (March 2026, ~9.8M page-days). Excluded on purpose: `trend_direction`, `trend_pct`, and the derived `is_declining` (label source); `impressions_90d`, `impressions_last_30d`, `ctr`, `avg_position` (windows that overlap the label); and the pseudonymous IDs (grouping and splits only). No client names, domains, URLs, or raw queries appear anywhere.

Source: `w03_data_contract.ipynb`, `w06_validation_audit.ipynb`.

## 3. Baseline

The Week-4 CTR-fix rule, scored as a decline ranker on the same client-grouped folds, reaches ROC-AUC 0.568 and Precision@50 0.600. A naive staleness ranker sits at 0.482 / 0.664. These are a fair bar, not a strawman: the canonical 24%-to-74% lecture story uses a much weaker baseline, so my smaller gap is honest, not a weaker model.

Source: `w04_baseline_score.ipynb`.

## 4. Model

Logistic Regression over the leakage-safe features (prior-window traffic, page age and freshness, length, keyword context, content type and intent). Chosen over eleven other families in the exhaustive search under `work/experiments/`; the one durable improvement was widening the training population to sub-threshold pages while still scoring only visible ones. Target is the `is_declining` proxy, a defined stand-in for "worth a refresh slot", never the trend columns it is derived from.

Source: `w05_model.ipynb`, `work/experiments/`.

## 5. Evaluation

Client-grouped 5-fold, so no client appears in both train and test. Visible base rate 0.598.

| Method | ROC-AUC | Precision@50 |
|---|---|---|
| Baseline: Week-4 CTR-fix | 0.568 | 0.600 |
| Baseline: staleness | 0.482 | 0.664 |
| LogReg (visible-train) | 0.607 | 0.740 |
| **LogReg (widened-train)** | **0.637** | **0.808** |

The lift over the CTR-fix baseline is about 1.35x on Precision@50, not 3x. Honest checks: a random split scores 0.685 AUC versus 0.607 grouped, a 0.078 memorization gap; and leakage escalates fast, from 0.63 honest to 0.95 once the last-30-day window is added and 0.997 with the raw trend column.

Source: `w05_model.ipynb`, `w06_validation_audit.ipynb`.

## 6. Interpretation

The model leans on prior-window engagement first (log clicks and sessions), then page age. A key negative result from the audit: the pooled "declining pages are shorter" gap flips sign once you compare within a client, so it is a mix effect across clients, not a within-site lever. Older pages are steadier, not stalest: pages past a year decline far less often (0.43) than pages three to six months old (0.69).

Source: `w05_model.ipynb`, `w06_validation_audit.ipynb`.

## 7. Recommendation

The top-50 queue assigns one action per page: 19 refresh, 15 rewrite title and meta, 14 relevance and internal links, and a couple to watch or expand. For 48% of visible pages the rule finds no clear lever and says so. Two archetypes dominate: maturing assets that are slipping, and neglected earners that pull traffic but get no attention. Confidence is directional and decision-support only, bounded by the limits above.

Source: `w07_action_playbook.ipynb`.

## 8. Reproducibility

Clone the repo, run `work/notebooks/capstone.ipynb` top to bottom on the committed data slice; it regenerates every number and both charts. The weekly notebooks hold each step, and `work/experiments/run_experiments.py` with `ledger.jsonl` and `lockbox.json` are the sealed-search receipts. Splits use fixed seeds.

## 9. Acknowledgments and data credit

Built on the FlyRank ML Internship dataset, linking to **https://flyrank.ai**. Crediting the data source is standard research practice and tells readers this is real production search data, used in anonymized form.

---

*Observed and directional results on an anonymized sample. No causal claims, no client-identifying details, and no claim to have proven Google's ranking algorithm.*
