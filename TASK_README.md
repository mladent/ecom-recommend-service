# Task for Data Scientist - Recommendation Engine + LLM agent

**IMPORTANT NOTE**

*The project is not fully tested.* LLM-based solutions are very slow; although optimised for batch processing, they still lack parallelised API calls.

There are two preloaded datasets to use:
- data_full.csv; the full Kaggle dataset
- data_minimum.csv; 10% of the Kaggle dataset

***For LLM API testing, copy data_minimum.csv to data.csv:***
`cp data_minimum.csv data.csv`

## Task 1

<!-- ```
Implement a module or agent of your choice for recommending product bundles e.g. rule-based, statistics, or ML-based solution. 
Please describe any reasoning behind your solution.
``` -->

### What was implemented (module choice: ML-based):

- A pluggable ML engine `src.recommendation_engine.BundleRecommendationEngine` that can host multiple recommenders and produce bundle recommendations via a single interface.
- Two concrete models:
  - [`src.recommendation_engine.NaiveBayesBundleRecommender`](src/recommendation_engine.py) for fast, sparse, high‑dimensional binary features using `MultiLabelBinarizer`.
  - `src.recommendation_engine.SVMBundleRecommender` for improved separation with configurable kernel and scaling.
- An ensemble mode in `src.recommendation_engine.BundleRecommendationEngine` that averages model probabilities to reduce variance and improve robustness across different transaction patterns.

### Reasoning behind the solution:

- **Sparse transaction data fits NB/SVM well.**  
  - Transactions are naturally multi‑hot feature vectors; `MultiLabelBinarizer` makes them suitable for classical ML classifiers, which is efficient and reliable for moderate datasets. 
  - This is implemented directly in `src.recommendation_engine.NaiveBayesBundleRecommender` and `src.recommendation_engine.SVMBundleRecommender` in `src/recommendation_engine.py`.
- **Binary bundle label simplifies training.**  
   - Each transaction is labeled positive if it contains any known bundle. 
   - This reduces complexity and lets models learn “bundle likelihood” from item co‑occurrence rather than a large multi‑label target space.
- **Complementary model strengths.**  
  - Naive Bayes is fast and works well with high‑dimensional sparse data, while SVM can capture more complex decision boundaries (especially with non‑linear kernels). 
  - This combination is exposed through the engine’s `ensemble` path in `src.recommendation_engine.BundleRecommendationEngine`.
- **Operational simplicity.**  
  - The engine exposes `fit_all()` and `recommend_bundles()` so new models can be added without changing callers. 
  - This is used directly in `main.train_recommenders` to register both recommenders and train them consistently.
- **Post‑processing is pragmatic.** 
  - Recommendations are filtered by confidence threshold and then aligned with actual bundles, keeping outputs actionable and consistent with discovered bundle definitions.


## Task 2

<!-- ```
Design and implement an LLM-based extension of the bundle recommendation model from Task 1. 
Clearly explain what the LLM improves, where it is used in the recommendation pipeline, and how its output is integrated into the final recommendations, and provide a working implementation.
``` -->


### LLM improvement:

The LLM adds out‑of‑stock (OOS) substitution intelligence. Instead of dropping bundles that contain unavailable items, it proposes context‑aware alternative items based on the in‑stock candidate pool, which keeps bundle recommendations usable and more realistic.

### Where it is used in the pipeline:
- It runs after ML scoring and bundle selection inside `src.recommendation_engine.BundleRecommendationEngine.recommend_bundles` in `src/recommendation_engine.py`.
- The engine first computes confidence (single model or ensemble), selects applicable bundles if the threshold is met, and **then** invokes the OOS resolution step when `LLM_OOS_ENABLED` is true.

### How LLM output is integrated:

1. The engine loads inventory and identifies OOS items.
2. For each missing item, it calls `src.utils.select_alternatives_with_llm` from `src/utils.py`.
3. The top alternative (by score) replaces the missing item if it meets the LLM_OOS_MIN_SCORE threshold from config (see config.py).
4. The final response includes:
    - bundles: updated bundles with substitutions applied.
    - bundle_substitutions: an audit trail showing original item, alternative item, and score.

This keeps the core ML recommendation signal intact while using the LLM to repair bundles into viable, in‑stock recommendations.

## Task 3 

<!-- ```
Provide a splitting to train and test datasets. Discuss possible different splitting
criteria. What other splitting criteria would you choose if you had access to richer
features (e.g., seasonality flags, user demographics) or larger historical datasets.
``` -->

Current implemented splitting in code is random train/test and k‑fold CV via `src.data_splitter.RandomSplit` and `src.data_splitter.KFoldSplit`. 
Those work when samples are independent and identically distributed (i.i.d.) and you want a fast, general estimate of model performance.

Because the dataset is expanded in preprocessing (normalized descriptions and added enriched attributes with categories), 
it partially simulates a richer dataset.

Alternative split criteria (with richer features or larger history):

 - **Time‑based/rolling split (for seasonality, drift)**: 
   - train on past, test on future; use rolling windows for backtesting.
 - **Seasonality‑stratified split (if you add seasonal flags)**: 
   - ensure each season is represented in both train/test, or hold out a season entirely for robustness.
 - **Customer‑level (group) split**: 
   - keep each customer in only one split to measure true personalization/generalization.
 - **Cold‑start split**: 
   - hold out new items or new users to evaluate onboarding performance.
 - **Demographic‑stratified split**: 
   - preserve user segment proportions (age, region, tier) or hold out a segment to test fairness.
 - **Basket‑size stratified split**: 
   - keep distribution of transaction lengths consistent.
 - **Bundle‑label stratified split**: 
   - preserve positive/negative label ratio (bundle‑present vs not).
 - **Geography/store split**: 
   - train on some regions, test on unseen regions to evaluate transferability.

If I had richer features and a larger history, I’d prioritize time‑based rolling splits plus customer‑level grouping, then add seasonality stratification to ensure the model learns stable patterns rather than leakage from the expansion step.

## Task 4

<!-- ```
How would you evaluate the business impact of the solution and share the outcome
with the internal stakeholders?
``` -->

### Evaluate impact with an offline→online plan:

 - Define KPIs: bundle attach rate, average order value, basket size, conversion, margin, return/cancel rate, and OOS substitution acceptance.
 - Offline evaluation: backtest on holdout data; report lift vs baseline rules and confidence intervals.
 - Online test: A/B or switchback test; measure KPI lift, guardrails (latency, stock‑out complaints), and segment effects (new vs returning customers).
 - Financial impact: convert KPI lift into revenue and margin uplift, with conservative/expected/optimistic scenarios.
 
### Sharing outcomes 

 via a brief stakeholder report and a 1‑page dashboard:

 - Summary: goal, test period, traffic share, overall lift, clear KPIs.
 - Breakdowns: segment and category impacts, OOS substitution effects.
 - Risks/next steps: drift monitoring, retraining cadence, rollout plan.

## Task 5

<!-- ```
Create an LLM-powered module or agent that handles situations where one or more
recommended bundle items are out of stock. Explain how the LLM is used to select
or generate suitable alternatives and provide a working implementation.
``` -->


This is implemented by the same OOS substitution flow described in **Task 2**. In short, the LLM is invoked after bundle selection in `BundleRecommendationEngine.recommend_bundles()` to replace missing items with in‑stock alternatives, and it returns a `bundle_substitutions` audit trail.

## Task 6

**Improving the Solution with Additional Data**


The current system uses transaction history only. 

### The following data might prove beneficial

| Data Source | Business Value |
| -- | -- |
| Customer Demographics | Segment-specific bundles, fairness |
| Product Metadata | Category, price, seasonality, brand |
| Temporal Features | Seasonality, trend detection, drift |
| Inventory Data | Real-time OOS handling, substitution |
| Customer Behavior | RFM scores, churn risk, lifetime value; grouping by similarity |
| Reviews/Feedback | Quality signals, sentiment, product fit |
| Website Behavior | Click-through, dwell time, cart abandonment |
| Geographic/Regional | Regional preferences, shipping costs |
| Competitor Pricing | Price elasticity, discount sensitivity |
| Supply Chain | Lead times, supplier quality, restock timing |
| Social Media | Collect information on current trends |

### Data Gathering & Preprocessing Architecture



| Raw Data Sources | Gathering method |
| -- | -- |
| Transaction History (web-store DB) | DB access |
| Customer Demographics (CRM) | DB access |
| Product Catalog (ERP) | DB access |
| Inventory (WMS) | DB access |
| Website Logs (Analytics) | DB access |
| Social Media | Web crawling, public and pay‑for‑access DBs |

### Key Improvements to Expect

| Metric | Current | With Add. Data | Mechanism |
| -- | -- | -- | -- |
| Bundle Attach Rate | Baseline | +8–15% | Personalization by segment |
| AOV (Avg Order Value) | Baseline | +5–12% | Premium bundles for VIPs |
| Conversion | Baseline | +3–7% | Seasonality-aware timing |
| OOS Substitution Accept Rate | ~60% | ~85% | LLM leverages richer product context |
| Model Retraining Frequency | Monthly | Weekly | Faster drift detection via RFM/seasonality |
| Fairness (by segment) | Unknown | Measurable | Demographic parity metrics |

 This roadmap keeps your existing architecture while scaling it to production-grade personalization.