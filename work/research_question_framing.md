# Research Question and Provisional Lane

**Assignment:** ML-02 · Week 1
**Author:** Emir Çabalak
**Status:** Provisional; the lane may change by the end of Week 4.

## Lane

I'm going with **Lane 2, Refresh / Content Opportunity Scoring**. It's a core lane, not one of the advanced ones. The two advanced lanes stay mentor-gated for now, so I'm not touching them at this stage.

The reason is mostly that my Week-1 work already sits here. The starter pipeline produces a ranked refresh queue with reason codes and a Precision@K evaluation, and in notebook 02 I compared a hand rule (`stale × visible`) against a small decision tree on that exact task. So I already have a baseline, a metric, and a first observation. Sharpening the question the pipeline is already answering makes more sense than starting a different lane from scratch.

## Question

> Among pages that are already visible in search, which ones should a content editor look at first for a refresh, so that limited editing time goes where it's most likely to matter?

This is a prioritization problem, not a causal one. I'm ordering a worklist, not claiming that any signal moves rankings.

## Unit of analysis

One content page over its recent performance window. The starter dataset gives me the 30- and 90-day columns; if the lane needs stronger time-window labels later I'll move to warehouse-shaped daily facts. One page in, one priority score out.

## Output

A ranked review queue: page, priority score, suggested action, reason codes, confidence label. Structurally it's what the pipeline already writes to `outputs/refresh_queue.csv`, but with the question, the label, and the reason codes made explicit and defensible rather than implied.

## The decision and the action

The decision is which pages land at the top of an editor's weekly refresh list. The person acting on it is a content editor who can realistically review a handful of pages a week. They open the top of the queue and act on the reason code: refresh a stale but still-visible page, expand a thin one, protect a declining high-value one, or just keep watching. The score orders the queue. A human still decides page by page, and nothing is edited automatically.

## Cost of being wrong

Editor time is the scarce resource, so the ordering is where the cost lives.

Rank a page high when a refresh won't help, and you've burned review hours. Worse, re-editing a stable page can lose rankings it already had, so a bad "yes" isn't neutral. Bury a page that genuinely needed attention, and it declines quietly with nobody getting to it in time. Because the constraint is attention rather than accuracy, what matters most is Precision@K near the top of the list, not a global score.

## Why this isn't just "train a model"

The label is a judgment, not something recorded in the data. "Should be refreshed" doesn't exist as a column; I have to define a leakage-safe proxy like declining-with-demand or stale-and-visible, and argue it's reasonable. Picking and defending that proxy is most of the actual work. A model can only rank as well as the label lets it.

Leakage is the real risk. In notebook 02, a tree allowed to see `trend_pct` scored almost perfectly and was worthless, because it had already seen the answer. The honest pipeline validates with client-holdout so a client's pages never sit in both train and test. Getting that right matters more than which model I pick.

A readable baseline might be enough. The hand rule stayed competitive with the tree at some cutoffs, and if a rule an editor can actually read ranks nearly as well, that's the better product. The model has to earn its complexity on held-out data before it's worth shipping.

And the output is a recommendation for a person, not an automated action. What I'm building is an explainable queue someone can override, not a system that acts on its own.

## Framing

Everything here is observational. The signals in this anonymized sample are associated with outcomes; I won't call any of them a proven Google ranking factor. I'll report findings as observed on this sample, with effect sizes and caveats, and no client data or unsafe fields leave the analysis.

## First evidence (Week 1, in this repo)

From notebooks 01 and 02, on the 30,000-row anonymized sample:

- Keyword `search_volume` was basically uncorrelated with actual `impressions_90d`. "High search volume means more traffic" didn't hold here, which is a good argument for keying the queue off delivered performance rather than intent volume.
- Once I held position tier fixed, most of the CTR gap between content types disappeared. Worth position-adjusting before reading CTR as a quality signal.
- On a proper train/test split the small tree still out-ranked the hand rule out of sample, but the rule stayed close. That's what pushes me toward a transparent-baseline-first plan.

These are directional observations on a sample, not causal claims.
