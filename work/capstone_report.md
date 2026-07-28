# Which already-visible pages should an editor refresh first?

- **Author:** Emir Çabalak
- **Lane:** Lane 2, Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/emircabalak/flyrank-ml-internship
- **Deployed paper:** see `submission/paper_url.txt`
- **Reproduces from:** `work/notebooks/capstone.ipynb` (all numbers and both charts)

## 0. Abstract

FlyRank publishes pages for many clients and cannot check every one by hand, so editors need a short list of what to fix first. I turned that into a ranking task on the anonymized internship data (30,000 pages, 32 clients): sort the visible pages by how likely each one is to be losing traffic, using only signals we would know before the drop. A plain logistic regression on those safe signals beats a fair rule-based baseline when it is tested on clients it never saw, getting about 41 of its top 50 pages right (Precision@50 of 0.81) against a base rate of 0.60. Training on the smaller pages too lifts the top of the list a little. The output is a ranked refresh list with a reason for each page. It makes no claim about Google and does not prove a refresh wins traffic back; it just helps an editor decide what to look at first.

## 1. Problem framing

A FlyRank editor looks after hundreds of live pages and has time to review maybe fifty a month. The hard part is not writing the fix, it is picking which pages deserve one. Fix a page that was fine and you waste hours and risk the ranking it already had; miss a page that is slipping and it keeps losing traffic while no one notices. One row is one page, the output is a ranked list with a reason and an action, the person using it is an editor working from the top down, and this is worth automating because the hints that a page is slipping are many and mixed together. The final call still belongs to a person.

Source: `w01_research_question.ipynb`, `w02_ml_task_framing.ipynb`.

## 2. Data safety

The model, the baseline, and the testing all run on the 30,000-row anonymized slice in the repo (32 clients, last 90 days), so anyone can rerun it without special access. The data contract checked the same setup on the full warehouse release (March 2026, about 9.8M page-days). Left out on purpose: `trend_direction`, `trend_pct`, and the label built from them; `impressions_90d`, `impressions_last_30d`, `ctr`, `avg_position` (windows that overlap the label); and the IDs (grouping and splitting only). No client names, domains, links, or real search queries appear anywhere.

Source: `w03_data_contract.ipynb`, `w06_validation_audit.ipynb`.

## 3. Baseline

The Week-4 CTR-fix rule, scored as a way to rank slipping pages on the same client-grouped folds, gets ROC-AUC 0.568 and Precision@50 0.600. A plain staleness ranker sits at 0.482 / 0.664. These are a fair bar, not an easy target: the well-known 24%-to-74% lecture story uses a much weaker baseline, so my smaller gap is honest, not a weaker model.

Source: `w04_baseline_score.ipynb`.

## 4. Model

Logistic regression on the safe signals (earlier 30-day traffic, page age and freshness, length, keyword context, content type and intent). Picked over eleven other model types in the search under `work/experiments/`; the one change that reliably helped was training on the smaller pages too while still scoring only the visible ones. The target is the "slipping" proxy, a stand-in for "worth a refresh slot", never the trend columns it is built from.

Source: `w05_model.ipynb`, `work/experiments/`.

## 5. Evaluation

5-fold grouped by client, so no client is in both training and testing. Base rate for slipping pages is 0.598.

| Method | ROC-AUC | Precision@50 |
|---|---|---|
| Baseline: Week-4 CTR-fix | 0.568 | 0.600 |
| Baseline: staleness | 0.482 | 0.664 |
| LogReg (visible-train) | 0.607 | 0.740 |
| **LogReg (widened-train)** | **0.637** | **0.808** |

The gain over the CTR-fix baseline is about 1.35 times on Precision@50, not 3 times. Two honest checks: a random split scores 0.685 versus 0.607 grouped, a 0.078 memorizing gap; and cheating escalates fast, from 0.63 honest to 0.95 once the last-30-day window is added and 0.997 with the raw trend column.

Source: `w05_model.ipynb`, `w06_validation_audit.ipynb`.

## 6. Interpretation

The model leans on earlier traffic first (clicks and sessions), then page age. One useful negative result from the audit: the "slipping pages are shorter" gap flips sign once you compare pages within the same client, so it is a mix across clients, not a real within-site lever. Older pages hold up better, they are not the stalest: pages over a year old slip far less often (0.43) than pages three to six months old (0.69).

Source: `w05_model.ipynb`, `w06_validation_audit.ipynb`.

## 7. Recommendation

The top-50 list gives one action per page: 19 refresh, 15 rewrite title and description, 14 relevance and internal links, and a couple to watch or expand. For 48% of visible pages the rule finds no clear lever and says so. Two kinds of page fill the list: solid pages starting to slip, and quiet earners that pull traffic but get no attention. The confidence is a hint for prioritizing, not proof, and it is bounded by the limits above.

Source: `w07_action_playbook.ipynb`.

## 8. Reproducibility

Clone the repo and run `work/notebooks/capstone.ipynb` top to bottom on the committed data; it rebuilds every number and both charts. The weekly notebooks hold each step, and `work/experiments/run_experiments.py` with `ledger.jsonl` and `lockbox.json` are the search receipts. Splits use fixed seeds.

## 9. Acknowledgments and data credit

Built on the FlyRank ML Internship dataset, linking to **https://flyrank.ai**. Naming the data source is standard practice and tells readers this is real search data, used in disguised form.
