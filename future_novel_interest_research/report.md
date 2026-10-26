# Deep-research field report

Generated from 10 JSON result files. Values marked uncertain are omitted.

## Table of contents

1. [Commerce and general sequential recommendation datasets](#commerce-and-general-sequential-recommendation-datasets) - kind: Comparative dataset audit | priority: High: H&M, Amazon Reviews 2023, dunnhumby. Medium: Instacart if restored, Retailrocket, Taobao, Steam. Low/auxiliary: Ali-CCP, Yelp, Tenrec, Ta-Feng. Reject or blocked: Yoochoose, Diginetica, JD/JData. This ranking prioritizes stable history, purchase-strength onset, semantic taxonomy, persistence and reproducible access rather than raw interaction count.
2. [Cross-domain, reading, mobility, and query-session datasets for emerging-interest prediction](#cross-domain-reading-mobility-and-query-session-datasets-for-emerging-interest-prediction) - kind: Comparative dataset audit and alternative task formulation | priority: High: Goodreads for semantic persistence and Foursquare NYC/Tokyo plus global variants for natural exploration. Medium: Amazon cross-domain after identity audit, and Gowalla for geography-only stress testing. Low/method-only: TripClick and TREC Session Track. Reject/blocked: AOL and unverified Yandex or third-party Douban-style cross-domain mirrors. None should replace an exposure-aware primary benchmark.
3. [Direct novel-interest exploration and forecasting: verified evidence map](#direct-novel-interest-exploration-and-forecasting-verified-evidence-map) - kind: Multi-paper research audit
4. [Interest evolution, shift, next-intent prediction, and proactive interest transition](#interest-evolution-shift-next-intent-prediction-and-proactive-interest-transition) - kind: Multi-paper adjacent-literature audit | priority: High: NNBR, NBR Reality Check, repeat/explore decomposition, TGB-Seq/distribution-shift stress testing, and ProRec-Video. Medium: proactive causal T-PRA and CFPRec. Low as direct prior work but useful baselines: RepeatNet, DIEN, ISLF, and structured-intent next-item recommendation.
5. [Source-verification corrections and additions after reviewing the direct-literature agent output; bibliographic identity is primary-source verified, while abstract-only method summaries may also use scholarly indexes.](#source-verification-corrections-and-additions-after-reviewing-the-direct-literature-agent-output-bibliographic-identity-is-primary-source-verified-while-abstract-only-method-summaries-may-also-use-scholarly-indexes)
6. [Media and social content datasets](#media-and-social-content-datasets) - kind: Comparative dataset audit | priority: High: KuaiRand and KuaiRec. Medium: frozen MovieLens 32M plus Tag Genome; MicroLens after exact-file and terms verification. Low: HetRec Last.fm. Reject/blocked: LFM-1b, Spotify MPD and Spotify skip sessions for reproducible longitudinal new-interest evaluation.
7. [Multi-interest modeling and generative retrieval](#multi-interest-modeling-and-generative-retrieval) - kind: Comparative method audit | priority: High: re-evaluate ComiRec or MIND and TIGER or LETTER under the identical Y_new protocol. Medium: P5-style text query generation and SUM's incremental interests. Method inspiration only: OneRec until reproducible artifacts exist. The first gate is a strong popularity/category-transition baseline; do not scale an LLM before proving user-conditioned signal.
8. [Public news and feed recommendation logs for future novel-interest prediction: MIND, EB-NeRD, Adressa, Globo.com, and auxiliary exposure logs](#public-news-and-feed-recommendation-logs-for-future-novel-interest-prediction-mind-eb-nerd-adressa-globo-com-and-auxiliary-exposure-logs) - kind: Comparative dataset audit | priority: 1) EB-NeRD: high, because its schema most directly fills MIND's missing timestamps, engagement, publication time, and exposure detail. 2) Adressa: medium-high for long-window positive-sequence robustness, subject to storage/access and exposure limitations. 3) MIND: medium-high as the existing English continuity benchmark but not sufficient alone for durable-interest claims. 4) Globo.com: low-medium because 16 days, no exposures, and no raw text sharply limit the estimand. 5) Yahoo!/NewsREEL auxiliaries: low until access/schema are re-verified.
9. [Novelty, unexpectedness, serendipity, exploration, and cross-domain recommendation: definitions, evidence, and reusable datasets](#novelty-unexpectedness-serendipity-exploration-and-cross-domain-recommendation-definitions-evidence-and-reusable-datasets) - kind: Multi-paper adjacent-objective and dataset audit | priority: Highest dataset-audit priority: KuaiRand, followed by KuaiRec and Amazon Reviews'23. High conceptual priority: SERAL's 10-active-day atomic-category label and Topic-Level temporal surprise annotation. Medium: Tenrec and Taobao Serendipity. Method-only medium: C2DSR. Low/direct-task reject: EMCDR/Douban and Open Bandit as standalone future-interest datasets, although both remain useful for transfer and exposure methodology respectively.
10. [Randomized-exposure, impression-slate, and explicit serendipity datasets: Yahoo! R6, Open Bandit Dataset, Coat, KuaiRand, FINN.no Slates, ContentWise Impressions, and Taobao Serendipity](#randomized-exposure-impression-slate-and-explicit-serendipity-datasets-yahoo-r6-open-bandit-dataset-coat-kuairand-finn-no-slates-contentwise-impressions-and-taobao-serendipity) - kind: Comparative audit of causal and subjective auxiliary datasets | priority: 1) KuaiRand: high. 2) FINN.no Slates: medium-high for exposure-aware robustness, not causal novelty. 3) ContentWise Impressions: medium. 4) Open Bandit Dataset: high only for OPE code validation. 5) Coat: medium only for selection-bias calibration. 6) Taobao Serendipity: medium only for subjective validation. 7) Yahoo R6: low until current access is re-verified, and method-only even if obtained.

<a id="commerce-and-general-sequential-recommendation-datasets"></a>
## Commerce and general sequential recommendation datasets

### Category: identity

#### name

Commerce and general sequential recommendation datasets

#### kind

Comparative dataset audit

#### venue_year

Public releases and challenges, 2015-2024; availability rechecked 2026-08-19

### Category: task_semantics

#### input_history

> For the long-history candidates, use chronologically earlier user/household purchases or reviews: Amazon review events, H&M transactions, dunnhumby baskets, and Instacart order-product rows. Retailrocket and Taobao provide shorter view/cart/purchase streams. Yelp supplies dated reviews, Steam supplies dated reviews and ownership/play signals, and Ali-CCP supplies display-click-conversion rows. Yoochoose and Diginetica identify sessions rather than durable users.

#### prediction_target

> Recommended CrossInterest target: the first future adoption of a product category or semantic concept absent from that user or household's frozen history, with item-level future events retained as evidence. Use purchase as the preferred target in H&M, dunnhumby, Instacart, Retailrocket, Taobao and Ta-Feng; review/rating is only a consumption proxy in Amazon/Yelp/Steam. Ali-CCP is better for exposed click/conversion prediction than long-horizon interest onset. Session sets only support next-click or next-buy within a session.

#### novel_interest_definition

> Define Y_new(u) from a frozen taxonomy mapping: future leaf category, product type, aisle/department, or semantic cluster not present in H(u). Report item novelty separately from category novelty. Do not call a newly observed item a new interest when its category already appears in history. For opaque category IDs, novelty is identifier-level only and cannot support semantic claims.

#### novelty_granularity_and_reference_window

> Amazon can support product, leaf-category and ancestor-category novelty over a multi-year prefix; H&M supports article, product-type, product-group, garment-group, section and department variants over a fixed historical window; dunnhumby supports product and commodity/subcommodity levels; Instacart supports product, aisle and department over prior orders. Retailrocket has category-tree IDs; Taobao has one opaque category ID; Yelp has flat business categories and geography; Steam has genres/tags. Freeze a pre-cutoff history window and compute novelty at every available hierarchy level.

#### interest_onset_definition

> Preferred onset is first future purchase in H&M, dunnhumby, Instacart, Retailrocket, Taobao or Ta-Feng, not first view. For Amazon/Yelp/Steam, first dated review or recommendation is a weaker positive proxy. Retailrocket and Taobao permit a funnel audit in which first view, cart, purchase and later repeat are separate labels. Ali-CCP onset is an exposed click or conversion within its logging window, not durable adoption.

#### persistence_signal

> H&M, dunnhumby and Instacart can test repeated purchase of the new category in later baskets; Amazon can test later reviews in the same new category but observes only reviewed purchases. Retailrocket and Taobao have multi-action signals yet short windows. Yelp is too sparse for most users; session datasets cannot measure persistence beyond one session. A minimum design should distinguish one-off onset, repeated category adoption, and repeat-item replenishment.

### Category: method_and_evaluation

#### method

> No source dataset natively defines future novel interest. Build deterministic per-user temporal cutoffs, freeze item-to-taxonomy mappings using only information available at the cutoff, generate category-level Y_new, and evaluate category generation/ranking plus item retrieval. Compare popularity, global category transition, user-conditioned co-occurrence, sequential recommenders and semantic/LLM encoders. Use the same-user temporal split for forecasting and a user-disjoint split for generalization.

#### evaluation_protocol

> Primary: select users with adequate pre-cutoff history and future support; predict all first future categories absent from H(u); use recall/precision/NDCG at category level plus exact-set and calibration metrics, and separately score sustained targets repeated in a later persistence window. Candidate-aware datasets should restrict negatives to contemporaneously available or exposed items; otherwise report full-catalog ranking and sampled-negative results separately. Never randomly split events across time. Evaluate repeat-versus-explore as disjoint strata and retain one row per user for user-level bootstrap confidence intervals.

#### generalization_axis

> Evaluate temporal future for seen users, user-disjoint held-out users, cold items inside known categories, and truly new-to-user categories. Amazon spans the broadest calendar interval; H&M and grocery datasets expose seasonality; Retailrocket/Taobao mainly test short-horizon temporal shift. Category-cold evaluation is only meaningful when taxonomy semantics are non-opaque.

#### main_findings_and_limits

> Best primary candidates are H&M, Amazon Reviews 2023 and dunnhumby, for complementary reasons: H&M has rich fashion taxonomy and purchase timestamps; Amazon has unmatched scale, duration and text; dunnhumby has dense full-store baskets and two-year household coverage. Instacart has an excellent order/aisle schema but its official release is no longer reproducibly reachable. Retailrocket and Taobao are useful short-window robustness checks. Ali-CCP helps exposure sensitivity but has opaque semantics. Session-only benchmarks must be rejected for longitudinal interest claims. Across all choices, first observed category is not necessarily first lifetime interest because histories are left-censored and exposure is missing.

### Category: dataset_schema

#### left_censoring_and_history_coverage

> Amazon's 27-year range is longest but includes only review writers and may miss earlier offline/online purchases. H&M observes only two years; dunnhumby intentionally samples frequent households and observes all purchases only at one retailer; Instacart exposes a bounded sequence (commonly 4-100 orders) without earlier lifetime history. Retailrocket and Taobao are severely left-censored by 4.5-month and 9-day windows. A target should be called 'newly observed in dataset history,' never a lifetime-new interest; run sensitivity analyses with minimum history length and burn-in periods.

#### feedback_and_exposure

> H&M, dunnhumby and Instacart contain purchases/baskets only. Amazon/Yelp contain ratings/reviews; Steam adds recommendation/play/ownership signals. Retailrocket contains view, add-to-cart and transaction events; Taobao contains page view, favorite, cart and buy. Ali-CCP contains display samples labeled for click and conversion and is the only serious exposure-side auxiliary in this group, although it lacks randomized propensities. Session challenges contain clicks and buys. None provides a complete randomized catalog-exposure log.

### Category: crossinterest_fit

#### leakage_and_missingness_risks

> Main risks are MNAR positives without impressions, left-censored histories, reviewer/loyalty-card selection, household-versus-user conflation, repeated users crossing random splits, future item metadata or taxonomy leakage, future popularity leakage, inventory/geography constraints, seasonality and sampled-negative inflation. H&M/Instacart/Kaggle test labels may be hidden or competition-shaped; build only from legitimately available historical files. Use user-level splits and cutoff-derived metadata/popularity. Never interpret never-observed items as dislikes.

#### recommended_role

- **primary benchmark:** H&M for semantic purchase novelty; Amazon Reviews 2023 for scale/text/long time; dunnhumby for dense household purchase trajectories.
- **primary if restored reproducibly:** Instacart, after documenting an owner-authorized snapshot, file hashes and terms.
- **robustness benchmark:** Retailrocket and Taobao for short-horizon multi-action onset; Steam for a digital-goods domain; possibly Ta-Feng after primary-source restoration.
- **auxiliary exposure validation:** Ali-CCP.
- **low-priority adjacent:** Yelp and heterogeneous Tenrec subsets.
- **reject for longitudinal claims:** Yoochoose and Diginetica session-only data; JD/JData until release verification.

#### priority

> High: H&M, Amazon Reviews 2023, dunnhumby. Medium: Instacart if restored, Retailrocket, Taobao, Steam. Low/auxiliary: Ali-CCP, Yelp, Tenrec, Ta-Feng. Reject or blocked: Yoochoose, Diginetica, JD/JData. This ranking prioritizes stable history, purchase-strength onset, semantic taxonomy, persistence and reproducible access rather than raw interaction count.

#### repeat_explore_and_persistence_audit

> For every usable dataset, partition future events into repeat item, new item within known category, first observed category, and first observed ancestor category. Define persistence as at least one additional purchase/review in the new category after a gap and report one-off versus sustained targets separately. H&M/dunnhumby/Instacart are best for this audit; Amazon is weaker because unreviewed repeat purchases are missing; Retailrocket/Taobao windows may be too short; session-only data cannot pass the persistence gate.

### Category: evidence

#### evidence_quotes_or_schema_facts

- Amazon official page: 571.54M reviews, 54.51M users, 48.19M items, 33 domains, May 1996-Sep 2023, with second-or-finer timestamps and rich item metadata.
- dunnhumby official page: The Complete Journey represents household-level transactions over two years for 2,500 frequent-shopping households, all of their purchases within the store, and direct-marketing contact history for select households.
- Yelp official page: 6,990,280 reviews, 150,346 businesses, 11 metropolitan areas, 200,100 pictures; the subset is intended for educational use.
- Steam official UCSD page: 7,793,069 reviews, 2,567,538 users, 15,474 items and 615 bundles; metadata includes purchases, plays, recommendations, bundles and prices.
- Instacart's original owner URL returned a 404 during verification, so a third-party mirror alone is insufficient provenance for a frozen benchmark.
- Tianchi's live Taobao meta description calls UserBehavior an Alibaba-provided implicit-feedback recommendation dataset; the standard five-column schema must still be verified from the downloaded archive before use.
- Yoochoose and Diginetica challenge logs are session recommendation resources; a session key must not be silently promoted to a stable user identity.

#### sources

- https://amazon-reviews-2023.github.io/ — authoritative statistics, date range, timestamp precision, metadata and downloads.
- https://arxiv.org/abs/2403.03952 — original Amazon Reviews 2023/BLaIR paper and benchmark positioning.
- https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data — official H&M competition distribution and schema route; account/rules required.
- https://www.dunnhumby.com/source-files/ — authoritative Complete Journey household count, two-year scope, purchase coverage and research purpose.
- https://www.instacart.com/datasets/grocery-shopping-2017 — original owner URL; verified 404 on 2026-08-19.
- https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset — principal Retailrocket distribution route; raw-file verification required.
- https://tianchi.aliyun.com/dataset/649 — official Taobao UserBehavior landing page.
- https://tianchi.aliyun.com/dataset/408 — official Ali-CCP landing page.
- https://business.yelp.com/data/resources/open-dataset/ — current Yelp scale, contents and educational-use statement.
- https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data — current Steam statistics, metadata and download links.
- https://github.com/yuangh-x/2022-NIPS-Tenrec — author repository and downloads for Tenrec.
- https://recsys.acm.org/recsys15/challenge/ — official RecSys 2015/YOOCHOOSE challenge context.
- https://competitions.codalab.org/competitions/11161 — historical CIKM Cup 2016/Diginetica challenge route.

### Uncertain fields

- candidate_space
- construct_h_f_ynew
- exposure_handling
- is_direct_match
- item_semantics
- official_sources
- release_and_license
- scale
- stable_user_identity
- taxonomy_and_opportunity_drift
- timestamps_and_order
- verification_status

<a id="cross-domain-reading-mobility-and-query-session-datasets-for-emerging-interest-prediction"></a>
## Cross-domain, reading, mobility, and query-session datasets for emerging-interest prediction

### Category: identity

#### name

Cross-domain, reading, mobility, and query-session datasets for emerging-interest prediction

#### kind

Comparative dataset audit and alternative task formulation

#### official_sources

- **Goodreads**
  https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html, https://github.com/MengtingWan/goodreads
- **Foursquare**
  https://sites.google.com/site/yangdingqi/home/foursquare-dataset
- **Gowalla**
  https://snap.stanford.edu/data/loc-gowalla.html, https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
- **Amazon_Reviews_2023**
  https://amazon-reviews-2023.github.io/, https://arxiv.org/abs/2403.03952
- **TripClick**
  https://tripdatabase.github.io/tripclick/, https://github.com/tripdatabase/tripclick, https://arxiv.org/abs/2103.07901
- **TREC_Session_Track**
  https://trec.nist.gov/data/session.html
- **historical_query_logs**
  https://www.kaggle.com/c/yandex-personalized-web-search-challenge, https://www.nytimes.com/2006/08/09/technology/09aol.html

### Category: task_semantics

#### prediction_target

> The proposed targets are the next first-observed book genre, venue category, geographic region, product domain, or query-intent cluster absent from a frozen history. Goodreads, Foursquare, Gowalla, and Amazon offer natural behavioral transitions but no explicit new-interest label. TripClick and TREC support next-query or session-subtopic prediction only; that is intent reformulation inside a task, not durable user-interest emergence.

#### novel_interest_definition

> Define novelty relative to the observed pre-cutoff semantic history: first positive event in a previously unseen book genre, venue category, product domain, or frozen query-intent cluster. New location, new book, or new URL alone is item novelty. A geographic move may reflect travel or opportunity rather than preference change, while a query reformulation may reflect task progress rather than a new personal interest.

#### interest_onset_definition

- **Goodreads:** First read/rated/reviewed book in a history-absent genre; adding a book to a shelf without reading should be a weaker onset tier.
- **Foursquare:** First check-in to a venue category absent from prior check-ins, preferably followed by another check-in in that category after a time gap.
- **Gowalla:** First check-in to a new spatial cluster, explicitly labeled geographic exploration rather than interest onset.
- **Amazon:** First dated review/rating in a previously unseen product domain or category; review is only a proxy for consumption.
- **TripClick_and_TREC:** First query or clicked document assigned to a new within-session intent cluster; this is search-intent transition, not a persistent interest onset.

#### persistence_signal

- **Goodreads:** Yes: later reads/ratings/reviews in the same newly entered genre, author, series, or topic can separate one-off exploration from repeated adoption.
- **Foursquare:** Yes within the capture window: later check-ins to the same new venue category or region, though repeated presence may reflect residence/travel opportunity.
- **Gowalla:** Yes for repeated spatial-cluster visits, but not semantic preference persistence.
- **Amazon:** Possible through later reviews in the same new domain/category; unreviewed repeat purchases are missing.
- **TripClick_and_TREC:** Only repeated intent within a short search session; no durable cross-session persistence signal.

### Category: method_and_evaluation

#### method

> Build domain-specific semantic events, freeze mappings before the test period, and use a temporal sequence model or hierarchical classifier to rank unseen semantic units. Add explicit transition features: recency, dwell time between events, semantic distance, geography, and prior cross-domain bridge concepts. Compare global transition matrices, user-conditioned co-occurrence, popularity, content similarity, recurrent/Transformer sequence models, and an LLM ranker. For query sessions, model reformulation and subtask transition separately rather than pooling them with user-interest forecasting.

#### evaluation_protocol

> For Goodreads, Foursquare, Gowalla, and Amazon, create non-overlapping burn-in/history, onset, and persistence windows per stable identity. Evaluate full semantic-vocabulary Recall/NDCG, macro target recall, semantic distance, calibration, and repeated-adoption accuracy; cluster uncertainty by user. Use same-user temporal prediction as the main axis and user-disjoint transfer as secondary. For query sessions, use official session/topic splits and next-intent metrics, report them separately, and do not claim durable-interest performance. Include global popularity, last-semantic-unit, transition-frequency, and shuffled-history controls.

#### generalization_axis

> Natural seen-user temporal generalization is available in Goodreads, Foursquare, Gowalla, and likely Amazon. Additional axes are unseen users, unseen items inside known semantic units, new semantic units for a user, cross-city mobility, and cross-product-domain transition. TripClick/TREC test within-session intent and query-frequency shift rather than long-term user generalization.

#### main_findings_and_limits

> Goodreads is the best low-cost semantic alternative: it has 876K stable users, 229M shelf interactions, 15M detailed reviews, rich book graphs, and later repeat signals, but genre labels are fuzzy and exposure is absent. Foursquare is the best natural transition stress test: it provides months of semantic check-ins and longer global mobility variants, but opportunity and travel confounding are severe. Gowalla offers much longer dense mobility but no venue taxonomy. Amazon can test genuine cross-product-domain transitions at scale after user-ID continuity is confirmed, but is already covered by commerce benchmarking. TripClick and TREC are valuable for representing intent evolution and reformulation; they do not answer the user-history-to-new-interest question. No release-verified dataset in this group directly labels sustained emerging interest or randomizes cross-domain opportunity.

### Category: dataset_schema

#### left_censoring_and_history_coverage

> Goodreads is a late-2017 snapshot of public shelves and cannot observe private/off-platform reading; user-entered dates can backfill older books. Foursquare/Gowalla begin after account and mobility histories already exist and observe only check-ins, so unseen categories may be rediscovery. Amazon observes only reviewed purchases. TripClick sessions omit cross-session identity, and TREC sessions are bounded experimental tasks. All targets must be called first observed in the released history, never lifetime-new.

#### feedback_and_exposure

- **Goodreads:** Shelf membership, read flag, rating, review flag/text, and reading/review dates; no impressions.
- **Foursquare_and_Gowalla:** Positive check-ins and optional friendship/profile data; no shown-but-not-visited venues.
- **Amazon:** Ratings, review text, helpfulness and timestamps; no product impressions.
- **TripClick:** Query-session click logs and clicked documents; the benchmark derives RAW and click-through-rate relevance labels.
- **TREC:** Multi-query session interactions and relevance judgments varying by year; not repeated natural consumption.
- **AOL_Yandex:** Historical query/click signals exist, but use/access is rejected or uncertain.

#### taxonomy_and_opportunity_drift

> Goodreads genres are overlapping, user-generated, and extracted at a snapshot; books and series accumulate over time. Foursquare category systems and venue availability may change, while geography, travel, opening hours, and city coverage strongly constrain opportunity. Gowalla has no taxonomy. Amazon categories and inventory drift across 27 years. TripClick's medical vocabulary, document index, and search system change over 2013-2020. Freeze semantic mappings, use only items first observed before scoring when possible, and report calendar/geography slices.

### Category: crossinterest_fit

#### leakage_and_missingness_risks

> Shared risks are left censoring, positive-unlabeled exposure, active-user selection, repeated users across random splits, future metadata/taxonomy leakage, popularity, seasonality, and treating first observed behavior as first lifetime interest. Goodreads adds user-entered/backfilled dates and fuzzy genre leakage. Foursquare/Gowalla add travel, residence, distance, city sampling, venue closure, and friend effects. Amazon adds reviewer selection and inventory drift. Query logs add privacy, session-boundary errors, result-position bias, clicked-document leakage, and task progress masquerading as interest change.

#### recommended_role

- **Goodreads:** Primary semantic robustness benchmark for natural cross-genre onset and persistence after a local timestamp/genre quality preflight.
- **Foursquare:** Primary alternative formulation for semantic and geographic exploration, with travel/opportunity strata.
- **Gowalla:** Geographic exploration robustness only.
- **Amazon:** Cross-product-domain robustness after cross-file identity verification; otherwise rely on the main commerce benchmark.
- **TripClick:** Method inspiration and within-session intent-transition benchmark only.
- **TREC:** Small controlled intent-reformulation benchmark only.
- **AOL_Yandex:** Reject AOL; block Yandex until current owner-authorized access is verified.

#### priority

> High: Goodreads for semantic persistence and Foursquare NYC/Tokyo plus global variants for natural exploration. Medium: Amazon cross-domain after identity audit, and Gowalla for geography-only stress testing. Low/method-only: TripClick and TREC Session Track. Reject/blocked: AOL and unverified Yandex or third-party Douban-style cross-domain mirrors. None should replace an exposure-aware primary benchmark.

#### repeat_explore_and_persistence_audit

> For every longitudinal dataset, separate repeat item/location, new item/location inside a known semantic unit, first observed semantic unit, and first observed ancestor/domain. Define persistence only from a later disjoint event in the new unit. Goodreads should stratify shelf-only, read, rating, and review-qualified onset. Foursquare should stratify home-city, travel, and new-city events and report return visits separately. Gowalla should use geographic terminology. Amazon should report reviewed-event missingness. TripClick/TREC cannot pass a long-term persistence gate and must remain in a separate within-session table.

### Category: evidence

#### evidence_quotes_or_schema_facts

- UCSD Goodreads page: late-2017 public-shelf collection; anonymized user/review IDs; 2,360,655 books, 876,145 users, and 228,648,342 user-book interactions; datasets join on book/user/review IDs.
- Goodreads page: fuzzy genres are extracted from popular shelves by simple keyword matching, and books may overlap across genre subsets.
- Foursquare author page: NYC/Tokyo contains 227,428 and 573,703 check-ins over about ten months, each with timestamp, GPS coordinates, and fine-grained venue-category semantics.
- Foursquare author page: the 18-month global release has 33,278,683 check-ins by 266,909 users at 3,680,126 venues in 415 cities across 77 countries.
- SNAP Gowalla page: 196,591 nodes, 950,327 friendship edges, and 6,442,890 check-ins from February 2009 to October 2010; each row has user, UTC time, latitude, longitude, and location ID.
- Amazon Reviews 2023 official page: 571.54M reviews, 54.51M users, 48.19M items, 33 domains, May 1996 to September 2023.
- TripClick official page: about 5.2M click-log entries from around 1.6M sessions between January 2013 and October 2020; access is request-based and redistribution is prohibited.
- NIST Session Track page lists separate 2010, 2011, 2012, 2013, and 2014 session datasets.

### Uncertain fields

- candidate_space
- construct_h_f_ynew
- exposure_handling
- input_history
- is_direct_match
- item_semantics
- novelty_granularity_and_reference_window
- release_and_license
- scale
- sources
- stable_user_identity
- timestamps_and_order
- venue_year
- verification_status

<a id="direct-novel-interest-exploration-and-forecasting-verified-evidence-map"></a>
## Direct novel-interest exploration and forecasting: verified evidence map

### Category: identity

#### name

Direct novel-interest exploration and forecasting: verified evidence map

#### kind

Multi-paper research audit

### Category: dataset_schema

#### left_censoring_and_history_coverage

> None of the verified methods proves that an interest absent from the observed window is new over the user's lifetime. Google/ACL history coverage is private; UpliftRec uses historical top-three categories, so sparse history directly changes 'unexpectedness'; NNBR restricts novelty to the observed prior baskets. All should be described as observed-history novelty, not ontological new interest.

#### taxonomy_and_opportunity_drift

> No verified paper fully releases taxonomy versions, item launch times, inventory, geography, or whether a target category was available during history. Google interest clusters are designer-controlled but private; UpliftRec categories or embedding clusters may drift; NNBR item availability is not exposure-audited. This prevents separating user-interest emergence from catalog opportunity.

### Category: crossinterest_fit

#### leakage_and_missingness_risks

> Major risks are finite-window left censoring, recommender-mediated MNAR exposure, future catalog/category availability, popularity and seasonality, using post-cutoff embeddings or clusters, and equating unobserved with disliked. Google/ACL cannot be audited from private logs. UpliftRec's randomized components help with exposure bias, but 'top-three' unexpectedness is history-frequency dependent. NNBR's purchase-only data lacks exposures and catalog opportunity; user-disjoint splitting does not remove left censoring.

#### repeat_explore_and_persistence_audit

> NNBR explicitly excludes every previously consumed item and is the only verified work here with a clean repeat-versus-explore item split. UpliftRec separates unexpected categories and unpopular items but does not define repeat versus first-ever category. Google/ACL optimize exploration but public reports do not expose a repeat/explore contingency table or per-interest persistence. None verifies sustained adoption with a later confirmation window.

### Category: evidence

#### evidence_quotes_or_schema_facts

> Google 2024 abstract: the framework 'recommends the next novel interests' through language-represented interest clusters and grounds them to items. ACL 2025 abstract: it 'decouple[s] novelty and user-alignment' and selects best-of-n novelty proposals with a user-aligned LLM; live metrics include watch activity, active users, and exploration diversity. UpliftRec full text: treatment is category exposure ratio; Yahoo!R3, Coat, and KuaiRec contain biased and randomized/unbiased parts; an item is unexpected when its category is not in the user's top three historical categories. NNBR abstract: an explore item is one the user 'has not consumed before,' and the target basket contains only novel items. PIG bibliographic fact: 'Personalized Interest Graphs for Theme-Driven User Behavior,' RecSys 2025, DOI 10.1145/3705328.3748133.

### Uncertain fields

- candidate_space
- construct_h_f_ynew
- evaluation_protocol
- exposure_handling
- feedback_and_exposure
- generalization_axis
- input_history
- interest_onset_definition
- is_direct_match
- item_semantics
- main_findings_and_limits
- method
- novel_interest_definition
- novelty_granularity_and_reference_window
- official_sources
- persistence_signal
- prediction_target
- priority
- recommended_role
- release_and_license
- scale
- sources
- stable_user_identity
- timestamps_and_order
- venue_year
- verification_status

<a id="interest-evolution-shift-next-intent-prediction-and-proactive-interest-transition"></a>
## Interest evolution, shift, next-intent prediction, and proactive interest transition

### Category: identity

#### name

Interest evolution, shift, next-intent prediction, and proactive interest transition

#### kind

Multi-paper adjacent-literature audit

#### venue_year

> RepeatNet AAAI 2019; DIEN AAAI 2019; ISLF IJCAI 2019; Intention-Aware Sequential Recommendation TKDE 2022; CFPRec IJCAI 2022; Next Basket Recommendation Reality Check TOIS 2023; Sequential Recommendation with Distribution Shifts SIGIR 2023; NNBR SIGIR 2023; TGB-Seq ICLR 2025; Proactive Recommendation/T-PRA AI Open 2025; ProRec-Video AAAI 2026

#### official_sources

> https://doi.org/10.1609/aaai.v33i01.33014806 ; https://doi.org/10.1609/aaai.v33i01.33015941 ; https://doi.org/10.24963/ijcai.2019/799 ; https://www.ijcai.org/proceedings/2019/0799.pdf ; https://doi.org/10.1109/TKDE.2021.3050571 ; https://doi.org/10.24963/ijcai.2022/521 ; https://www.ijcai.org/proceedings/2022/0521.pdf ; https://doi.org/10.1145/3587153 ; https://doi.org/10.1145/3539618.3591624 ; https://doi.org/10.1145/3604915.3608803 ; https://arxiv.org/abs/2308.01308 ; https://openreview.net/forum?id=8e2LirwiJT ; https://github.com/TGB-Seq/TGB-Seq ; https://doi.org/10.1016/j.aiopen.2025.08.003 ; https://doi.org/10.1609/aaai.v40i17.38475

### Category: task_semantics

#### input_history

> DIEN consumes a user's chronological behavior sequence plus a candidate target ad/item. ISLF consumes only the current anonymous session and models short-term, long-term, and latent factors. Intention-Aware Sequential Recommendation infers latent intents and structured transitions from an item sequence. CFPRec consumes historical check-in trajectories and context, then self-ensembles inferred multi-step future preferences. The distribution-shift framework uses historical interaction sequences under train/test distribution change. NNBR uses ordered prior purchase baskets. Proactive Recommendation and ProRec-Video use historical interests plus candidate transition content; ProRec-Video additionally profiles receptivity and adapts to simulated/dynamic feedback.

#### prediction_target

> RepeatNet predicts the next session item through repeat versus explore modes; explore means choosing outside the current session, not a new category. DIEN predicts CTR for a supplied candidate item. ISLF and the structured-intent model rank the next item in a session/sequence. CFPRec predicts the next POI, not a semantic interest label or the actual multi-step future sequence. The NBR Reality Check evaluates next-basket items while decomposing repeat and explore. The distribution-shift work predicts future interactions under shifted item popularity/user preference distributions. NNBR predicts only items never consumed before in the next basket. TGB-Seq predicts future temporal graph destinations/edges under low repetition, not semantic interest. The 2025 proactive-causal T-PRA line aims to steer future interest through recommendation interventions [uncertain acronym expansion/details]. ProRec-Video plans a hierarchical semantic interest-transition path and corresponding video sequence, then reports transition success and satisfaction.

#### is_direct_match

> Direct at item granularity: NNBR. Direct/closest but not yet a public forecasting benchmark: ProRec-Video and the 2025 proactive-causal T-PRA line. Evaluation-framework adjacent: the NBR Reality Check. Robustness-only: sequential recommendation under distribution shifts and TGB-Seq. Adjacent next-location planning: CFPRec. Repeat/explore mechanism only: RepeatNet. Representation or candidate-ranking only, not novel-interest forecasting: DIEN, ISLF, and structured-intent sequential recommendation. Modeling an evolving hidden state is not the same as predicting an observed new interest.

### Category: method_and_evaluation

#### method

> DIEN uses a GRU-based interest extractor with auxiliary next-behavior supervision and a target-aware interest-evolution layer. ISLF combines recurrent-VAE latent factors with long-/short-term session interest and attention. Structured-intent recommendation models discrete/latent intents and their transition structure before next-item ranking. CFPRec self-ensembles multi-step future-preference representations inferred from past periodic trajectories and fuses them with current preference. The SIGIR 2023 framework explicitly trains sequential recommenders for distribution shifts [uncertain implementation detail]. NNBR uses a bidirectional transformer with item/basket masking and swapping. The proactive-causal paper treats recommendations as interventions that steer interest [uncertain method detail]. ProRec-Video combines receptivity profiling, hierarchical transition planning, semantic video sequencing, agent-based simulation, Reflexion, and dynamic feedback adaptation.

#### generalization_axis

> DIEN: candidate CTR, including private online deployment, not user/category-disjoint. ISLF: later session events for anonymous sessions, with test items constrained to train vocabulary. Structured-intent work: seen-item next-item generalization. CFPRec: future check-in within existing POI vocabulary. Distribution-shift work: temporal/popularity/preference shift. NNBR: user-disjoint train/test but seen item vocabulary and item-level novelty. ProRec-Video: semantic transition beyond current preference [uncertain whether users, interest nodes, or items are held out].

### Category: dataset_schema

#### left_censoring_and_history_coverage

> Every observational formulation can mistake pre-dataset interests for new ones. This is especially severe for anonymous ISLF sessions, short CTR histories, and finite POI traces. NNBR correctly means unseen in observed baskets only. ProRec-Video's 'beyond historical preferences' also depends on a private or unreported lookback window. None proves lifetime novelty.

### Category: crossinterest_fit

#### recommended_role

> Primary adjacent label-construction reference: NNBR. Required evaluation references: NBR Reality Check and RepeatNet's repeat/explore gate. High-value negative controls and representation baselines: DIEN, ISLF, and structured-intent models, explicitly labeled as CTR/next-item rather than novel-interest prediction. Robustness benchmarks: sequential recommendation under distribution shifts and TGB-Seq, with TGB-Seq treated as method-only because destinations are not interest categories. Auxiliary planning method: CFPRec. Research-direction/method inspiration: proactive causal T-PRA and ProRec-Video, pending real-user data, taxonomy, and persistence validation.

#### priority

> High: NNBR, NBR Reality Check, repeat/explore decomposition, TGB-Seq/distribution-shift stress testing, and ProRec-Video. Medium: proactive causal T-PRA and CFPRec. Low as direct prior work but useful baselines: RepeatNet, DIEN, ISLF, and structured-intent next-item recommendation.

#### repeat_explore_and_persistence_audit

> RepeatNet explicitly routes repeat versus explore next-item generation, and the NBR Reality Check separately measures repetition/exploration ratios and performance; both are essential evaluation precedents. NNBR goes further by excluding every previously purchased item. DIEN, ISLF, structured-intent, CFPRec, distribution-shift results, and TGB-Seq link MRR can still be dominated by familiar-node structure unless re-evaluated with an explicit contingency table. ProRec-Video is semantically exploratory but does not publicly demonstrate later repeat adoption. CrossInterest should report repeat item, new item in known category, new category, return-after-inactivity, and persistent-new-category outcomes separately.

### Category: evidence

#### evidence_quotes_or_schema_facts

> DIEN's publisher abstract states that it predicts CTR for a target item and that the evolution layer is target-item-relative. ISLF's PDF defines each label as the next-click item and removes test items absent from training; its datasets are YOOCHOOSE and Diginetica. CFPRec's abstract says multi-step future preferences are inferred from learned past preference to improve next-POI prediction, so they are not observed labels. The distribution-shift paper states that train history and future serving data violate the IID assumption as popularity and preferences evolve. NNBR defines explore items as items not consumed before. ProRec-Video's publisher abstract describes receptivity profiling, hierarchical transition planning, dynamic feedback adaptation, two-dataset evaluation, 85% transition success, and 78.3% satisfaction.

#### sources

> RepeatNet: https://doi.org/10.1609/aaai.v33i01.33014806 . DIEN: https://doi.org/10.1609/aaai.v33i01.33015941 . ISLF: https://doi.org/10.24963/ijcai.2019/799 and https://www.ijcai.org/proceedings/2019/0799.pdf . Structured Intent Transition: https://doi.org/10.1109/TKDE.2021.3050571 . CFPRec: https://doi.org/10.24963/ijcai.2022/521 and https://www.ijcai.org/proceedings/2022/0521.pdf . NBR Reality Check: https://doi.org/10.1145/3587153 . Distribution Shifts: https://doi.org/10.1145/3539618.3591624 . NNBR: https://doi.org/10.1145/3604915.3608803 and https://arxiv.org/abs/2308.01308 . TGB-Seq: https://openreview.net/forum?id=8e2LirwiJT and https://github.com/TGB-Seq/TGB-Seq . Proactive Recommendation/T-PRA: https://doi.org/10.1016/j.aiopen.2025.08.003 . ProRec-Video: https://doi.org/10.1609/aaai.v40i17.38475 .

### Uncertain fields

- candidate_space
- construct_h_f_ynew
- evaluation_protocol
- exposure_handling
- feedback_and_exposure
- interest_onset_definition
- item_semantics
- leakage_and_missingness_risks
- main_findings_and_limits
- novel_interest_definition
- novelty_granularity_and_reference_window
- persistence_signal
- release_and_license
- scale
- stable_user_identity
- taxonomy_and_opportunity_drift
- timestamps_and_order
- verification_status

<a id="source-verification-corrections-and-additions-after-reviewing-the-direct-literature-agent-output-bibliographic-identity-is-primary-source-verified-while-abstract-only-method-summaries-may-also-use-scholarly-indexes"></a>
## Source-verification corrections and additions after reviewing the direct-literature agent output; bibliographic identity is primary-source verified, while abstract-only method summaries may also use scholarly indexes.

### Category: identity

#### venue_year

ACM RecSys 2025

### Category: evidence

#### sources

https://doi.org/10.1145/3705328.3748133, https://dblp.org/rec/conf/recsys/ZinmanCFBFE25, https://www.semanticscholar.org/paper/085c2a95e9c1607273904ea26c3512e4c0750f54

### Other Info

#### purpose

> Source-verification corrections and additions after reviewing the direct-literature agent output; bibliographic identity is primary-source verified, while abstract-only method summaries may also use scholarly indexes.

#### verified_as_of

2026-08-19

#### records

- title: Personalized Interest Graphs for Theme-Driven User Behavior | venue_year: ACM RecSys 2025 | doi: 10.1145/3705328.3748133 | verified_claim: The eBay system builds multi-granularity LLM-generated user-interest graphs for theme-centric cross-category recommendation. Its abstract reports online gains for engagement with previously unseen categories, purchases, and buyer count. | crossinterest_role: Direct industrial method precedent for cross-category interest expansion; not a public history-to-future benchmark because logs, graph, taxonomy, and evaluation data are private. | sources: https://doi.org/10.1145/3705328.3748133, https://dblp.org/rec/conf/recsys/ZinmanCFBFE25, https://www.semanticscholar.org/paper/085c2a95e9c1607273904ea26c3512e4c0750f54
- title: When Relevance Meets Novelty: Dual-Stable Periodic Optimization for Serendipitous Recommendation | venue_year: arXiv preprint 2025, version 3 dated 2026-03-04 | identifier: arXiv:2508.00450 | verified_claim: CoEA jointly models long-term group identity and short-term individual interests, and periodically co-optimizes novelty and relevance LLMs using incremental feedback; its abstract reports offline and online experiments. | crossinterest_role: Method inspiration for separating stable history from emerging interests and for periodic alignment; it does not provide a verified public future-novel-interest benchmark. | sources: https://arxiv.org/abs/2508.00450
- title: Serendipitous Recommendation with Multimodal LLM | venue_year: RecSys 2025 EARL Workshop preprint | identifier: arXiv:2506.08283 | verified_claim: The hierarchical MLLM framework identifies relevant but unexplored interest clusters and grounds them through conventional recommenders; its abstract reports live experiments on a private short-video platform. | crossinterest_role: Direct method precedent for unexplored semantic clusters and multimodal grounding; private production evidence, not a reusable dataset. | sources: https://arxiv.org/abs/2506.08283
- title: Breaking the Relevance-Diversity Seesaw: Hierarchical LLM Reasoning with RL for Industrial Novelty Recommendation | venue_year: SIGIR 2026 | doi: 10.1145/3805712.3808399 | verified_claim: BALANCE decomposes novelty recommendation into Novelty Tag, Interest Topic, and Recommendation List stages, adds self-reflection supervision, and optimizes sequence- and item-level rewards with SIPO. The abstract reports offline and JD.com online A/B tests. | crossinterest_role: Closest 2026 industrial method precedent for hierarchical novelty direction and topic generation; its data and production evaluation remain private. | sources: https://doi.org/10.1145/3805712.3808399, https://dblp.org/rec/conf/sigir/SunZWXYXQL26, https://www.semanticscholar.org/paper/d1d23a8d371ea378217787a7e29f32c9b3243bf8


<a id="media-and-social-content-datasets"></a>
## Media and social content datasets

### Category: identity

#### name

Media and social content datasets

#### kind

Comparative dataset audit

#### venue_year

> MovieLens 32M and Tag Genome 2021; HetRec 2011; LFM-1b ICMR 2016; Spotify MPD RecSys 2018; KuaiRec and KuaiRand CIKM 2022; MicroLens 2023

#### official_sources

- **MovieLens 32M and Tag Genome**
  https://grouplens.org/datasets/movielens/, https://files.grouplens.org/datasets/movielens/ml-32m-README.html, https://grouplens.org/datasets/movielens/tag-genome-2021
- **Last.fm HetRec 2011**
  https://grouplens.org/datasets/hetrec-2011/, https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
- **LFM-1b**
  https://www.cp.jku.at/datasets/LFM-1b/, https://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_icmr_2016.pdf
- **Spotify Million Playlist Dataset**
  https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
- **Spotify Music Streaming Sessions**
  https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge
- **KuaiRec**
  https://github.com/chongminggao/KuaiRec, https://zenodo.org/records/18164998
- **KuaiRand**
  https://github.com/chongminggao/KuaiRand, https://zenodo.org/records/10439422, https://doi.org/10.1145/3511808.3557624
- **MicroLens**
  https://github.com/westlake-repl/MicroLens, https://recsys.westlake.edu.cn/, https://arxiv.org/abs/2309.15379

### Category: task_semantics

#### input_history

> MovieLens supplies earlier timestamped ratings/tags per anonymized user. LFM-1b historically supplied timestamped scrobbles; HetRec only aggregates artist listening counts, while its tag assignments have timestamps. KuaiRec and KuaiRand supply timestamped video exposures/views with watch-time feedback and stable user IDs. MicroLens supplies timestamp-ordered public user-comment interactions. Spotify MPD supplies ordered tracks inside playlists but no user ID; Spotify skip data supplies sessions but no durable user identity.

#### prediction_target

> Predict a future semantic genre, tag, artist cluster or video category absent from the frozen user history. Prefer long-view, like, follow or repeated engagement over a raw impression as positive onset. MovieLens ratings and MicroLens comments are consumption/strong-engagement proxies, not verified watching. MPD can only predict withheld playlist tracks; Spotify skip can only predict skip within a session, so neither directly targets a user's future interest.

#### novel_interest_definition

> A target is a future semantic concept absent from H(u), not merely a new movie, song or video ID. For MovieLens, distinguish new genre from new Tag Genome concept; for music, distinguish new track, artist and genre; for KuaiRec/KuaiRand, use the released hierarchical category names or caption-derived clusters. State that novelty is relative to observed data, never lifetime novelty.

#### novelty_granularity_and_reference_window

> Use a frozen pre-cutoff user history and report item, creator/artist, leaf category/tag and ancestor category separately. MovieLens genres are coarse while Tag Genome relevance is fine-grained and computed post hoc. KuaiRand provides up to four semantic category levels in its supplement; KuaiRec now provides caption/category files and hierarchical raw categories. Playlist position is not a temporal user-history window.

#### interest_onset_definition

> KuaiRand onset should be first long-view/like/follow in a new category after exposure; report click-only separately. KuaiRec can use first sufficiently high watch ratio in a new category. MovieLens onset is first positive rating in a new genre/tag concept, with rating-time caveats. MicroLens onset is first public comment on a new video concept. MPD track inclusion is playlist composition, not user-interest onset.

#### is_direct_match

- **KuaiRand:** Strong short-window direct construction plus randomized-exposure validation.
- **KuaiRec:** Strong short-window direct construction with nearly fully observed user-item feedback in the small matrix.
- **MovieLens:** Dataset-only adjacent match: long chronology and stable users, but coarse semantics, no exposure and rating timestamps need not equal watch times.
- **LFM-1b:** Conceptually strong historical match, operationally rejected because the owner withdrew download access.
- **HetRec Last.fm:** Not a temporal listening match because user-artist listens are aggregated; timestamps apply to tag assignments, not listens.
- **Spotify MPD and skip sessions:** Not a longitudinal user match; playlist/session identifiers must not be promoted to user IDs.
- **MicroLens:** Dataset-only adjacent match with rich content and timestamps but positive-only public-comment sampling and release constraints.

#### persistence_signal

> KuaiRand has thousands of sequential interactions per user and multiple feedback strengths, enabling later category re-engagement. KuaiRec supports repeated category engagement but the nearly complete small matrix is mostly one exposure per user-video. MovieLens can measure repeated later ratings in a category, not repeat consumption of the same movie. LFM-1b could measure repeated artist/genre listening but is unavailable. HetRec aggregates counts without sequence. MPD and skip-session data cannot establish cross-session persistence. MicroLens can measure repeated comments across related videos, not passive watching.

### Category: method_and_evaluation

#### method

> Create deterministic temporal user splits, map items to frozen semantic concepts, and predict first future concept onset plus later persistence. Compare popularity, creator/category transitions, sequential recommenders, multi-interest encoders and text/multimodal encoders. For KuaiRand, estimate metrics separately on standard-policy and randomized exposures; for KuaiRec, use the fully observed small matrix for candidate-complete counterfactual checks.

#### evaluation_protocol

> Primary user-level protocol: burn-in/history, onset and persistence windows; no random event split. Evaluate category recall/NDCG, first-onset time, calibration and sustained-adoption recall, with repeat-item, new-item-known-category and new-category strata. MovieLens should use chronological ratings and user-disjoint secondary evaluation. KuaiRand should report standard logs and randomized pool separately. MPD's playlist-continuation protocol and Spotify's session skip protocol must not be interpreted as user-level new-interest evaluation.

#### generalization_axis

> Seen-user temporal future, held-out users, cold items within known categories and new-to-user categories. MovieLens spans 1995-2023 but has platform and catalog drift. KuaiRec/KuaiRand test dense or randomized exposure over short windows. MicroLens mainly tests content-driven cold-item generalization. Playlist/session data tests context continuation, not durable-user generalization.

#### exposure_handling

> KuaiRand is the strongest exposure resource: about 0.37% of standard-feed recommendations were replaced by an item sampled uniformly from a 7,583-video random pool, and is_rand marks interventions. KuaiRec's small matrix is 99.6% observed because almost every included user viewed every included video, though it is not a randomized full-platform catalog. MovieLens, Last.fm, MPD and MicroLens lack feed impression slates. Spotify skip rows are exposed tracks inside sessions but lack stable users and randomized propensities.

#### main_findings_and_limits

> Use KuaiRand as the highest-priority causal/short-horizon benchmark and KuaiRec as a candidate-complete robustness benchmark. Use frozen MovieLens 32M plus Tag Genome as a long-horizon semantic robustness set, with strong warnings about ratings, post-hoc semantics and missing exposure. LFM-1b and Spotify MPD are not currently reproducible public choices. MPD has playlist IDs, not user IDs. MicroLens is valuable for multimodal cold-item experiments but its comments are selected strong positives and current license boundaries require care.

### Category: dataset_schema

#### stable_user_identity

> MovieLens, Last.fm, KuaiRec, KuaiRand and MicroLens provide anonymized IDs consistent across their own files/windows. MovieLens IDs do not transfer across dataset releases. MicroLens users are public commenters, not necessarily viewers. Spotify MPD pid identifies a playlist, not a person; collaborative playlists further invalidate a one-playlist-one-user assumption. Spotify skip data is explicitly session based.

#### left_censoring_and_history_coverage

> MovieLens observes only ratings after user enrollment and requires at least 20 ratings, so first observed genre is not first watched genre; rapid backfilled ratings may distort chronology. KuaiRand covers about one month and KuaiRec a short platform window, making prior interests invisible. MicroLens captures only public comments and caps comments per video, creating severe positive and popularity selection. MPD samples filtered public playlists with dithering/fictitious tracks and cannot reconstruct a user's other playlists or listening history.

#### item_semantics

> MovieLens has title, release year, coarse genres, user tags and external IMDb/TMDb links; Tag Genome supplies dense relevance to human-readable tags but is a computed snapshot. HetRec Last.fm has artist names and user tags; LFM-1b genre profiles use Last.fm tags plus external vocabularies. MPD has playlist title, track/artist/album names and URIs but no released genre taxonomy. KuaiRec now has captions, category text, tags and hierarchical raw categories. KuaiRand supplements provide captions/cover text and up to four category levels. MicroLens contains titles, tags, covers, audio and full videos.

#### feedback_and_exposure

> MovieLens: ratings and tag applications, no impressions/watch. HetRec: aggregate play count and social/tag relations. LFM-1b: historical scrobbles. MPD: playlist membership/order, not consumption feedback. Spotify skip: exposed session tracks with skip labels. KuaiRec: view duration, duration and watch ratio plus dense exposure; no native like label. KuaiRand: 12 signals including click, like, follow, comment, forward, hate, long-view and play time, plus is_rand. MicroLens: public comments as strong positive proxies, plus item-level likes/views metadata.

#### candidate_space

> KuaiRec small matrix has an almost complete 1,411-by-3,327 observed candidate block. KuaiRand's randomized pool of 7,583 videos has known randomized opportunities, but its 32-million-video standard catalog is not exhaustively exposed. MovieLens lists rated/tagged movies, not the full historically available catalog. MPD lists tracks appearing in sampled playlists, not recommendations shown. LFM/HetRec and MicroLens cannot reconstruct full contemporaneous candidate catalogs. Spotify skip supplies the already-served session sequence, not alternative candidates.

### Category: crossinterest_fit

#### leakage_and_missingness_risks

> Risks include missing impressions, selected raters/commenters, left censoring, rating backfill, random event splits, creator popularity, future upload/daily-statistic leakage, post-hoc Tag Genome or category leakage, playlist-to-user identity error, session-to-user identity error, and sampled-negative inflation. Freeze dataset versions, user splits, item semantics and first-seen times; never infer dislike from unobserved media.

#### recommended_role

- **KuaiRand:** High-priority causal and short-window new-category benchmark.
- **KuaiRec:** High-priority fully observed candidate-space robustness benchmark.
- **MovieLens 32M plus Tag Genome:** Medium-priority long-horizon semantic robustness benchmark.
- **MicroLens:** Medium-priority multimodal/cold-item auxiliary after release and license audit.
- **HetRec Last.fm:** Low-priority aggregate preference/tag resource, not temporal prediction.
- **LFM-1b:** Reject for new experiments unless owner access is restored.
- **Spotify MPD and skip:** Method inspiration/pretraining only; reject as longitudinal user benchmarks.

#### priority

> High: KuaiRand and KuaiRec. Medium: frozen MovieLens 32M plus Tag Genome; MicroLens after exact-file and terms verification. Low: HetRec Last.fm. Reject/blocked: LFM-1b, Spotify MPD and Spotify skip sessions for reproducible longitudinal new-interest evaluation.

#### repeat_explore_and_persistence_audit

> Partition future media events into repeat item, new item-known creator/category, new creator-known category, and new category. Then split new-category targets into one-off versus later repeated positive engagement. KuaiRand supports exposure-aware persistence; KuaiRec supports category persistence; MovieLens supports later distinct-movie ratings in the category. MPD/session data and aggregate HetRec listens fail the durable persistence audit.

### Category: evidence

#### evidence_quotes_or_schema_facts

- MovieLens 32M README: 32,000,204 ratings, 2,000,072 tag applications, 87,585 movies and 200,948 anonymized users between 1995-01-09 and 2023-10-12; user IDs are consistent across ratings and tags.
- GroupLens says ml-latest changes over time and is not appropriate for reporting research results; use the stable 32M release.
- HetRec README: 92,834 user-artist relations contain listeningCount, while timestamps are provided for tag assignments, not listening events.
- LFM-1b official page: the dataset is not available for download anymore due to license issues.
- AIcrowd MPD page: the dataset is no longer downloadable; each entry is a playlist with pid, title, modified_at and ordered tracks, not a user record.
- KuaiRec README: the small matrix has 4,676,570 interactions at 99.6% density and each included user viewed almost every included video.
- KuaiRand README: about 0.37% of interactions were replaced by randomly sampled pool items; is_rand identifies intervention, and the random-policy log contains 1,186,059 interactions.
- MicroLens paper: interactions are publicly observable comments with timestamps, not platform clicks or likes.

#### sources

- https://files.grouplens.org/datasets/movielens/ml-32m-README.html — exact MovieLens schema, counts, dates and terms.
- https://grouplens.org/datasets/movielens/ — stable-release guidance and Tag Genome statistics.
- https://grouplens.org/datasets/hetrec-2011/ — live Last.fm HetRec download and headline counts.
- https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip — embedded primary README with aggregate-listening schema and non-commercial license.
- https://www.cp.jku.at/datasets/LFM-1b/ — owner statement that LFM-1b was withdrawn for license issues.
- https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge — MPD schema, sampling, terms and current unavailability.
- https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge — session semantics, scale and withdrawal notice.
- https://github.com/chongminggao/KuaiRec — official schema, statistics, semantics and Zenodo links.
- https://github.com/chongminggao/KuaiRand — official randomization design, schema, statistics, license and Zenodo links.
- https://arxiv.org/abs/2309.15379 — MicroLens construction, exact scale and comment-feedback semantics.
- https://recsys.westlake.edu.cn/ — current MicroLens files and academic-use/no-redistribution declaration.

### Uncertain fields

- construct_h_f_ynew
- release_and_license
- scale
- taxonomy_and_opportunity_drift
- timestamps_and_order
- verification_status

<a id="multi-interest-modeling-and-generative-retrieval"></a>
## Multi-interest modeling and generative retrieval

### Category: identity

#### name

Multi-interest modeling and generative retrieval

#### kind

Comparative method audit

#### venue_year

> MIND CIKM 2019; ComiRec KDD 2020; SUM 2021; P5 RecSys 2022; TIGER NeurIPS 2023; LETTER CIKM 2024; OneRec 2025

#### official_sources

- **MIND**
  https://arxiv.org/abs/1904.08030, https://doi.org/10.1145/3357384.3357924
- **ComiRec**
  https://arxiv.org/abs/2005.09347, https://github.com/THUDM/ComiRec, https://doi.org/10.1145/3394486.3403344
- **SUM**
  https://arxiv.org/abs/2102.09211
- **P5**
  https://arxiv.org/abs/2203.13366, https://github.com/jeykigung/P5
- **TIGER**
  https://arxiv.org/abs/2305.05065, https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html
- **LETTER**
  https://arxiv.org/abs/2405.07314
- **OneRec**
  https://arxiv.org/abs/2502.18965

### Category: task_semantics

#### input_history

> MIND, ComiRec and SUM encode a sequence of previously interacted items into multiple latent interest vectors or channels. TIGER and LETTER convert catalog items into semantic-code sequences and condition an autoregressive model on earlier item codes. P5 converts histories, item metadata and tasks into natural-language templates. OneRec encodes historical video behavior and generates a session-sized recommendation sequence.

#### prediction_target

> The original targets are mostly the next interacted item, a withheld item, or a generated recommendation session. MIND/ComiRec retrieve candidate items; SUM predicts sequential behavior; TIGER/LETTER generate the next item's identifier; P5 includes sequential recommendation among several text-to-text tasks; OneRec generates videos and optimizes aggregate/session preference. None of these primary task definitions explicitly targets the set of future semantic interests absent from the user's frozen history.

#### novel_interest_definition

> The audited works generally do not define novel interest. Multiple interest vectors mean multiple aspects already inferred from H(u), diversity means a varied recommendation list, cold item means limited item-side interaction history, and a new semantic-ID sequence means a newly generated item identifier. None is equivalent to Y_new(u)=F(u)-H(u) at a declared semantic granularity.

#### novelty_granularity_and_reference_window

> Original evaluations usually score item IDs relative to the preceding interaction sequence. They do not stratify whether the target is a new item in a known category, a new leaf category, a new ancestor category, or merely semantically distant. For CrossInterest, attach a frozen taxonomy to every event and compute novelty against the entire declared H window, not only the last item or active-interest vector.

#### interest_onset_definition

> Original models usually treat the held-out interaction as the label, regardless of whether it repeats a known interest. CrossInterest should define onset independently of the architecture: first future purchase/strong engagement in a previously absent semantic category, followed by an optional persistence window. A model-generated interest phrase or semantic code is a prediction, not onset evidence.

#### is_direct_match

- **MIND, ComiRec, SUM:** Transferable method-only matches for representing several historical interests; their evaluations do not isolate history-absent future interests.
- **TIGER and LETTER:** Transferable method-only matches for semantic item tokenization, constrained generation and cold-item generalization; semantic IDs are latent item identifiers, not novel-interest labels.
- **P5:** Transferable prompting/text-generation framework; its unified tasks do not by themselves create a valid new-interest target.
- **OneRec:** Transferable session-wise generation and preference-alignment method; online watch-time lift is not an isolated new-interest result.
- **query and hierarchical category generation:** A promising CrossInterest adaptation, but not directly validated by these works: generate explicit taxonomy paths or retrieval queries and score only history-absent future concepts.

#### persistence_signal

> The representative papers evaluate one held-out next item, a recommendation list/session, or overall online engagement. They generally do not require the newly predicted semantic interest to recur later. OneRec's session-wise generation captures within-request coherence, not longitudinal adoption. CrossInterest must reserve a later window and label one-off versus repeated new-category engagement.

### Category: method_and_evaluation

#### method

- **multi-interest extraction:** MIND uses capsule dynamic routing over behaviors plus label-aware attention; ComiRec uses multi-interest extraction and a controllable accuracy-diversity aggregation; SUM maintains multiple incremental interest channels with erase-add updates and attention.
- **semantic-ID retrieval:** TIGER learns residual-quantized semantic IDs from item content, then autoregressively predicts the next item's code sequence. LETTER jointly regularizes tokenization with hierarchical semantics, collaborative alignment and code diversity, and adds ranking-guided generation.
- **language generation:** P5 casts user/item/review tasks as personalized text-to-text prompts. A CrossInterest variant could generate a human-readable query or taxonomy path, then retrieve items under that concept.
- **unified/session generation:** OneRec replaces retrieve-then-rank with encoder-decoder session generation, sparse MoE capacity and iterative reward-model/DPO preference alignment.
- **recommended composition:** Use a multi-interest encoder for H, an explicit exploration head for concepts absent from H, constrained hierarchical decoding, and a separate item retriever. Do not force the exploration head to reconstruct only observed-interest vectors.

#### evaluation_protocol

> The audited literature mainly uses chronological leave-one-out/next-item evaluation with Recall/Hit Rate/NDCG and sometimes list diversity, cold-item buckets or online aggregate engagement. This does not answer the CrossInterest question because repeat targets dominate and negatives may be sampled. Re-evaluate every method on a frozen H/F split with explicit Y_new, full-catalog or contemporaneous-candidate ranking, separate repeat/explore strata, category-level exactness, user-level bootstrap intervals and a later persistence window.

#### generalization_axis

> MIND/ComiRec/SUM emphasize user-interest capacity and next-item retrieval; TIGER reports improved retrieval for items with no prior interaction history, which is item cold-start rather than user-new interest; LETTER improves item tokenization/ranking; P5 emphasizes cross-task zero/few-shot transfer; OneRec emphasizes industrial scale and preference alignment. Add seen-user future-new-category and held-out-user future-new-category axes explicitly.

#### exposure_handling

> The representative offline benchmarks generally use logged positives and sampled or catalog negatives without randomized exposure or impression propensities. Their models therefore learn from MNAR interactions. OneRec's reward model simulates preference pairs because only one list is displayed, but this is not randomized exposure identification. CrossInterest should use exposed/randomized subsets where available and report unlabeled-catalog evaluation separately.

#### main_findings_and_limits

> The methods provide strong building blocks but no direct evidence that they predict future interests absent from history. Multi-interest encoders can improve recall by covering established tastes; diversity controls can add variety without accuracy on true new interests; semantic IDs help item retrieval and cold items but can encode popularity/collaboration; LLM prompting can verbalize plausible interests without behavioral grounding. The key research contribution should therefore be the target construction and repeat-versus-explore evaluation, not merely swapping in a larger generator.

### Category: dataset_schema

#### stable_user_identity

> All sequential evaluations assume a stable user or session history within their processed datasets, but papers often split each user's sequence rather than test identity durability across long real-world windows. OneRec's session generation and some industrial logs may emphasize request/session continuity. Dataset-level stable identity must be audited independently; a session or playlist key cannot substitute for a user key.

#### left_censoring_and_history_coverage

> Multi-interest models treat observed history as sufficient input but do not establish that it covers the user's prior lifetime interests. Short or truncated sequences can make an old interest appear novel. Semantic-ID and LLM models do not solve this measurement problem. Apply minimum-history and burn-in gates, vary H length, and describe targets as newly observed within the dataset.

#### item_semantics

> MIND/ComiRec/SUM can run on ID/category embeddings and do not require human-readable semantics. TIGER creates content-based residual-quantized codes; LETTER adds hierarchical semantic, collaborative and diversity regularization. P5 uses textual metadata and reviews. OneRec generates video identifiers in an industrial system. CrossInterest should retain human-readable taxonomy labels alongside latent codes so predictions and novelty checks remain auditable.

#### feedback_and_exposure

> The methods are usually trained on positive interactions such as clicks, purchases, ratings or watches; explicit impressions, positions and propensities depend on the underlying dataset and are generally absent from public evaluations. OneRec aligns to a learned reward and reports watch time; that outcome remains policy-dependent. Use feedback strength to define onset before model training rather than letting each model redefine the label.

#### candidate_space

> MIND/ComiRec retrieve by nearest interests from an item pool; TIGER/LETTER decode semantic IDs under a catalog trie or identifier vocabulary; P5 may generate item tokens/text; OneRec generates a session directly. These mechanisms can cover large catalogs, but the papers do not reconstruct historical availability or actual exposures. For category generation, constrain decoding to the frozen taxonomy and then retrieve only contemporaneously eligible items.

### Category: crossinterest_fit

#### construct_h_f_ynew

> Use the same leakage-safe dataset builder for every architecture: H(u) is the frozen sequence before cutoff; F(u) is strong future behavior; Y_new(u) contains future taxonomy concepts absent from all of H. Feed only H to MIND/ComiRec/SUM, TIGER/LETTER, P5 or OneRec adaptations. Add two heads: repeat-item/category retrieval and exploration-category generation. For the exploration head, mask concepts already present in H during constrained decoding, then retrieve items under generated concepts. Score against Y_new before scoring item ranking.

#### leakage_and_missingness_risks

> Risks include target-aware interest selection during training, future category/text features in semantic tokenizers, collaborative codebooks trained on dev/test interactions, popularity leakage, random rather than temporal splits, sampled-negative inflation, session/user conflation, post-cutoff taxonomy drift, LLM memorization and plausible-but-ungrounded generated queries. Fit tokenizers, concept vocabularies, popularity and retrieval indices on training-time data only and preserve hashes.

#### recommended_role

- **MIND/ComiRec/SUM:** Primary encoder baselines and ablations for whether multiple historical-interest slots improve new-category forecasting.
- **TIGER/LETTER:** Primary generative-retrieval baselines for semantic item codes and constrained catalog decoding.
- **P5:** Natural-language query/category generation baseline, with strict grounding and closed-vocabulary evaluation.
- **OneRec:** Method inspiration for multi-item/session decoding and preference alignment; reproduction is limited by industrial data/code availability.
- **new CrossInterest model:** Hybrid: explicit repeat/explore separation, interpretable taxonomy-path generation, candidate retrieval and persistence-aware training.

#### priority

> High: re-evaluate ComiRec or MIND and TIGER or LETTER under the identical Y_new protocol. Medium: P5-style text query generation and SUM's incremental interests. Method inspiration only: OneRec until reproducible artifacts exist. The first gate is a strong popularity/category-transition baseline; do not scale an LLM before proving user-conditioned signal.

#### repeat_explore_and_persistence_audit

> Re-run every baseline with four disjoint labels: repeat item, new item within known concept, first observed concept and sustained first observed concept. Report the fraction of test positives in each group before metrics. Measure whether each latent interest slot maps to known-history concepts or contributes to Y_new. A diverse list containing unseen categories counts only if future behavior confirms the category, and persistence requires another later positive after onset.

### Category: evidence

#### evidence_quotes_or_schema_facts

- MIND explicitly represents a user with multiple vectors and learns them by clustering historical behaviors; the paper frames evaluation as matching/recommendation, not history-absent interest forecasting.
- ComiRec's abstract formalizes the task as predicting the next items and controls the balance between recommendation accuracy and diversity.
- SUM maintains multiple interest channels with erase-and-add updates for large-scale sequential recommendation.
- TIGER trains a sequence-to-sequence model to predict the Semantic ID of the next item and separately reports better retrieval for items with no prior interaction history.
- LETTER integrates hierarchical semantics, collaborative signals and code-assignment diversity into a learnable tokenizer; its target remains generative recommendation/ranking.
- P5 converts interactions, descriptions, metadata and reviews into text-to-text tasks and supports personalized prompts, zero-shot and few-shot transfer.
- OneRec generates a session of videos and uses a reward model plus iterative DPO; its reported 1.6% watch-time lift is an aggregate online result, not a new-interest stratum.

#### sources

- https://arxiv.org/abs/1904.08030 — MIND architecture, historical-behavior routing and matching task.
- https://arxiv.org/abs/2005.09347 — ComiRec next-item formulation, multi-interest extraction and diversity control.
- https://github.com/THUDM/ComiRec — live author code repository; no license surfaced in the repository metadata check.
- https://arxiv.org/abs/2102.09211 — SUM multi-channel incremental interest model.
- https://arxiv.org/abs/2305.05065 — TIGER semantic IDs, autoregressive next-item retrieval and cold-item result.
- https://arxiv.org/abs/2405.07314 — LETTER learnable tokenization and ranking-guided generation.
- https://arxiv.org/abs/2203.13366 — P5 text-to-text personalized prompting framework.
- https://github.com/jeykigung/P5 — live MIT-licensed P5 implementation.
- https://arxiv.org/abs/2502.18965 — OneRec session generation, MoE, reward modeling/DPO and online watch-time result.

### Uncertain fields

- release_and_license
- scale
- taxonomy_and_opportunity_drift
- timestamps_and_order
- verification_status

<a id="public-news-and-feed-recommendation-logs-for-future-novel-interest-prediction-mind-eb-nerd-adressa-globo-com-and-auxiliary-exposure-logs"></a>
## Public news and feed recommendation logs for future novel-interest prediction: MIND, EB-NeRD, Adressa, Globo.com, and auxiliary exposure logs

### Category: identity

#### name

> Public news and feed recommendation logs for future novel-interest prediction: MIND, EB-NeRD, Adressa, Globo.com, and auxiliary exposure logs

#### kind

Comparative dataset audit

#### official_sources

- **MIND**
  - https://msnews.github.io/
  - https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news
  - https://aclanthology.org/2020.acl-main.331/
  - https://github.com/msnews/MIND
- **EB-NeRD**
  - https://recsys.eb.dk/
  - https://recsys.eb.dk/dataset/
  - https://arxiv.org/abs/2410.03432
  - https://github.com/ebanalyse/ebnerd-benchmark
- **Adressa**
  https://reclab.idi.ntnu.no/dataset/, https://reclab.idi.ntnu.no/dataset/cxdataset_v5.pdf, https://doi.org/10.1145/3106426.3109436
- **Globo.com**
  - https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom
  - https://github.com/gabrielspmoreira/chameleon_recsys
  - https://arxiv.org/abs/1808.00076
  - https://arxiv.org/abs/1904.10367
- **auxiliary**
  https://webscope.sandbox.yahoo.com/catalog.php?datatype=r, http://www.newsreelchallenge.org/dataset/

### Category: task_semantics

#### prediction_target

> None of the four primary datasets natively labels 'future new interest.' Their native targets are candidate click/ranking for an impression (MIND and EB-NeRD) or next clicked article/page view (Adressa and Globo.com). For CrossInterest, define a user-level temporal target: categories, subcategories, topics, entities, or semantic clusters positively consumed during a fixed future window but absent from a fixed past window. Keep the observed target as positive-unlabeled adoption, not a complete latent-interest set.

#### novel_interest_definition

> Recommended operational definition: a future positive engagement in a semantic unit not positively consumed in the user's observed history, with item novelty reported separately from category/topic novelty. A single future click is 'observed exploration,' not necessarily an interest. EB-NeRD uniquely permits a stricter onset threshold using click plus read-time/scroll and a separate later-repeat criterion. MIND has only clicks; Adressa has page views plus sometimes activeTime; Globo.com has clicks only in the public release.

#### novelty_granularity_and_reference_window

- **primary_definition:** Use leaf category/subcategory novelty relative to every observed positive in a fixed pre-cutoff history window. Also report ancestor-category novelty and semantic-distance novelty as separate outcomes; do not mix new-item clicks with new-interest labels.
- **MIND:** Category and subcategory novelty relative to the first four or five weeks; the official taxonomy has 20 top-level categories. A new article in an old category is item novelty only.
- **EB-NeRD:** Category, one of 263 subcategories, publisher-provided topic, or embedding-cluster novelty relative to the 21-day history. The seven-day behavior period is the future window.
- **Adressa:** Category, keyword/entity, or semantic-text novelty relative to an earlier window within the ten-week log. Use a minimum history duration and a disjoint future duration.
- **Globo.com:** Category or 250-dimensional content-embedding cluster novelty relative to an earlier portion of the 16-day log; the window is short and original text is absent.

#### interest_onset_definition

- **MIND:** First clicked exposure in a previously unseen semantic unit; no dwell signal is available, so onset remains weak.
- **EB-NeRD:** Prefer the first clicked in-view article in an unseen semantic unit that also passes a pre-registered next_read_time or next_scroll_percentage threshold. Retain click-only onset as a sensitivity analysis.
- **Adressa:** First page view in an unseen unit, optionally requiring activeTime above a threshold; missing activeTime must not be treated as zero.
- **Globo.com:** First click in an unseen category/cluster; version 2 removed repeated clicks inside sessions, so persistence must be measured across later events or sessions rather than within-session duplicate rows.

#### persistence_signal

- **MIND:** Possible only as repeated later clicks in the remaining short future window; no dwell or explicit satisfaction signal.
- **EB-NeRD:** Strongest among the candidates: later category/topic clicks can be separated from onset, with read-time and scroll used as engagement quality. Seven future days still limit long-term persistence claims.
- **Adressa:** Repeated later views across sessions and activeTime can indicate persistence over several weeks, subject to missing dwell and absent exposure.
- **Globo.com:** Cross-session repeats can be counted, but version 2 deliberately removes same-article repeats within sessions and only 16 days are observed.

### Category: method_and_evaluation

#### method

> Build a common temporal benchmark rather than reuse native next-click labels unchanged. For each user, freeze an observation cutoff, derive semantic histories only from pre-cutoff positive events, enumerate exposed future positives where impressions exist, and label first-time semantic units. Compare global popularity, recent popularity, last-category transition, conditional co-occurrence, content-similarity, sequence models, and language-model rankers. Add an engagement-qualified EB-NeRD target and a persistence head predicting whether onset is followed by another positive in the same semantic unit.

#### evaluation_protocol

> Use chronological windows and user-level aggregation. Report seen-user temporal generalization as the main axis, and create an optional fully user-disjoint test to measure transfer without calling it personalized prediction. For MIND use the official first-four/fifth/sixth-week chronology but dynamically append earlier positive impressions when forming later targets. For EB-NeRD use 21-day history, early future days for onset, and remaining future days for persistence; the exact split within seven days must be pre-registered. For Adressa and Globo.com choose non-overlapping windows after inspecting event counts. Evaluate full taxonomy ranking and, where present, actual in-view candidate ranking; report Recall/NDCG/MRR, macro label recall, onset precision among exposed candidates, coverage, calibration, and paired user-clustered bootstrap intervals. Include popularity and last-category transition baselines.

#### generalization_axis

> The intended main axis is temporal future for stable, previously observed users. Secondary axes are new items within known categories, newly published articles, leaf-category onset, and user-disjoint transfer. Do not conflate unseen-user evaluation with predicting a known user's interest change. EB-NeRD and MIND can explicitly test in-impression item ranking; Adressa and Globo.com primarily test positive-sequence forecasting under unknown exposure.

#### main_findings_and_limits

> EB-NeRD materially improves on MIND for CrossInterest because it exposes aligned history timestamps, in-view slates, article publication times, body text, category/subcategory/topic metadata, and engagement depth. It still observes only six weeks and supplies fixed 21-day histories, so dynamic reconstruction and explicit persistence auditing are required. Adressa provides ten weeks and rich Norwegian content/activeTime, making it attractive for temporal robustness, but the absence of impressions creates severe exposure MNAR. Globo.com is useful for short-session and cold-article robustness but is weak for durable new-interest claims. MIND remains a useful English benchmark with exact impression candidates, but released archives omit full bodies despite the paper's dataset-level description, histories are split-level snapshots in the official construction, and clicks alone do not establish durable interest.

### Category: dataset_schema

#### timestamps_and_order

- **MIND:** Impression timestamp to the second in MM/DD/YYYY HH:MM:SS AM/PM; histories are sorted by click time according to the paper, but the public history column does not include per-click timestamps.
- **EB-NeRD:** Microsecond-typed impression timestamps and aligned timestamp lists for history clicks; article publication and modification timestamps are also present. In-view ID order is deliberately shuffled.
- **Adressa:** Event time is Unix time; publishTime is provided for articles; session boundaries and chronological event order can be reconstructed. activeTime is seconds when known.
- **Globo.com:** Click timestamp, session start, article creation time, and hourly CSV partitioning support exact temporal order across the 16-day collection.

#### left_censoring_and_history_coverage

- **MIND:** At most four weeks precede the training target week and five weeks precede test. Users were selected for at least five clicks in the six-week period, so pre-window interests are unobserved and apparent novelty can be rediscovery.
- **EB-NeRD:** Each target split exposes only 21 days of pre-period clicks. Active users were selected using five to 1,000 clicks during 2023-05-18 to 2023-06-08, which induces activity selection. Earlier interests and off-platform consumption are unobserved.
- **Adressa:** The ten-week archive provides the longest observable history here, but any cutoff remains left-censored at 2017-01-01 and user IDs may be device/browser scoped. Require a minimum history duration and run sensitivity analyses with longer burn-in.
- **Globo.com:** Only 16 days are observed and version 2 keeps sessions of at least two clicks, producing severe left censoring and selection toward active sessions. Treat novelty as 'not seen in the observed window,' never lifetime-new interest.

#### item_semantics

- **MIND:** Released news.tsv contains article ID, category, subcategory, title, abstract, URL, title entities, and abstract entities plus Wikidata-derived embeddings. The paper describes article bodies, but Microsoft Learn explicitly states that full bodies are not distributed due to licensing and some URLs are expired.
- **EB-NeRD:** articles.parquet includes Danish title, subtitle/abstract, full body, category and subcategories, article type, URL, premium flag, publication/modification time, image IDs, NER/entities, topics, sentiment, and aggregate early-consumption fields. Text and image embeddings are separately offered.
- **Adressa:** The light release includes canonical URL/document ID, title, category, keywords, named entities, author, publish time, and generated profiles. Full requested data adds body text and richer raw/article/content attributes.
- **Globo.com:** Public metadata provides article ID, category ID, creation time, publisher ID, and word count, plus a 250-dimensional learned article embedding for 364,047 articles. Original Portuguese article text is not shared for licensing reasons.

#### taxonomy_and_opportunity_drift

- **MIND:** Twenty top-level editorial categories plus subcategories are observed over six weeks, but taxonomy versioning and article publication times are not explicit in the released file. News popularity and opportunity change rapidly; future news metadata must be joined only after its first observed availability.
- **EB-NeRD:** Thirty-two categories, 263 subcategories, publication/modification time, premium status, page context, device and subscription help model opportunity. Topics and entities come from proprietary models and may encode a later extraction snapshot; aggregate total_inviews/pageviews/read_time are measured over the first seven days after publication and must never be used to predict an earlier onset.
- **Adressa:** Category/profile fields and publish time allow partial drift auditing, but light and full schemas differ, some articles are paywalled, and user/device/location constraints affect availability. Freeze taxonomy mappings and candidate eligibility at cutoff.
- **Globo.com:** Category IDs and creation time allow cold-item and catalog-age slices. Original taxonomy meanings and actual serving inventory are limited, and the 16-day period can be dominated by transient news cycles.

### Category: crossinterest_fit

#### construct_h_f_ynew

- **MIND:** For each user, H(u) is the set or ordered multiset of category/subcategory labels attached to all history clicks before cutoff. Reconstruct F(u) from clicked articles in later timestamped impressions, dynamically adding earlier future clicks when scoring later ones. Y_new(u)=set(F)-set(H). Preserve impression IDs and in-slate non-clicks for exposure-aware ranking. Do not use the same fixed history for every later event without an explicit static-history ablation.
- **EB-NeRD:** H(u) comes from all timestamped clicks in history.parquet before the behavior period. Sort behaviors by impression time and obtain F(u) from clicked in-view articles in an onset window; dynamically update history. Define Y_new at category, subcategory, or pre-frozen topic/embedding-cluster level. Reserve later behavior days to label persistence and use next_read_time/scroll to create engagement-qualified onsets. This is the cleanest construction among the audited datasets.
- **Adressa:** Sort events by released userId and Unix time. After a burn-in period, derive H(u) from positive page views and F(u) from a disjoint future interval; compute new category/keyword/entity/semantic-cluster units. activeTime can qualify positives. Because exposures are absent, evaluate positive forecasting or sampled ranking with explicit positive-unlabeled assumptions.
- **Globo.com:** Aggregate all sessions per user_id, sort clicks, split the 16 days chronologically, map clicks to category or frozen embedding clusters, and compute set difference. Use article created_at_ts to exclude unavailable candidates. Persistence is a later cross-session click in the same new unit. The construction is feasible but fragile because the window is short and exposure is absent.

#### leakage_and_missingness_risks

> Shared risks are left censoring, exposure MNAR, active-user selection, popularity/seasonality, transient breaking-news effects, user-ID fragmentation, semantic taxonomy drift, and treating unexposed items as negatives. MIND risks include fixed split-level histories, missing per-history-click timestamps, expired URLs, and accidentally assuming paper-described bodies are in the archive. EB-NeRD risks include future leakage from total_inviews/total_pageviews/total_read_time and article modification time, fixed histories not updated within behavior splits, and shuffled positions. Adressa risks include missing activeTime, light/full schema mismatch, paywall-mediated opportunity, and no impressions. Globo.com risks include short-window rediscovery, removal of within-session repeats, missing article text, and a 364k catalog whose actual exposure is unknown. All splits and semantic mappings must be frozen before looking at test outcomes.

#### recommended_role

- **EB-NeRD:** Primary new benchmark after a local schema/count preflight and license review. It is the best source for exposure-aware onset and engagement/persistence sensitivity analyses.
- **MIND:** Retain as the English baseline and continuity benchmark, but describe it as observed future novel-click prediction and keep dynamic-history versus fixed-history variants separate.
- **Adressa:** High-value temporal robustness benchmark for longer histories, rich text, and dwell-qualified positives; not the main exposure-aware benchmark.
- **Globo.com:** Low-cost session/cold-article robustness benchmark or method sanity check; not evidence of durable new-interest prediction.
- **auxiliary:** Use Yahoo! R6 only to test exposure/off-policy methodology, and inspect NewsREEL/Plista archives before considering them.

#### priority

> 1) EB-NeRD: high, because its schema most directly fills MIND's missing timestamps, engagement, publication time, and exposure detail. 2) Adressa: medium-high for long-window positive-sequence robustness, subject to storage/access and exposure limitations. 3) MIND: medium-high as the existing English continuity benchmark but not sufficient alone for durable-interest claims. 4) Globo.com: low-medium because 16 days, no exposures, and no raw text sharply limit the estimand. 5) Yahoo!/NewsREEL auxiliaries: low until access/schema are re-verified.

#### repeat_explore_and_persistence_audit

> For every dataset, partition future positives into repeat semantic units already in H(u), first observed exploration units absent from H(u), and item-only novelty inside an old unit. For each exploration onset, label one-off versus repeated adoption using later disjoint events; never count the onset event itself as persistence. EB-NeRD should additionally report click-only versus dwell/scroll-qualified onset and persistence. Adressa should report activeTime-observed and activeTime-missing strata. MIND should report static official history versus dynamically appended history. Globo.com should measure persistence only across later events/sessions because version 2 removed repeated clicks within a session. Report per-user rates, macro semantic-unit recall, opportunity-conditioned rates where impressions exist, and confidence intervals clustered by user.

### Category: evidence

#### evidence_quotes_or_schema_facts

- MIND official documentation: behaviors.tsv has Impression ID, User ID, Time, History, and Impressions with click labels; news.tsv has ID, category, subcategory, title, abstract, URL, and entities.
- MIND paper: one million users with at least five clicks were sampled over six weeks from October 12 to November 22, 2019; training history uses the first four weeks and test history the first five weeks.
- MIND release caveat: Microsoft Learn states that full article bodies are not made available for download because of licensing, even though the paper describes bodies in the constructed dataset.
- EB-NeRD official schema: every split contains seven days of behaviors and 21 days of fixed prior click history; history includes article IDs, timestamps, read times, and scroll percentages.
- EB-NeRD paper: 1,103,602 users, 125,541 articles, 37,966,985 impressions, 32 categories, and 263 subcategories over April 27 to June 8, 2023.
- EB-NeRD behavior schema: in-view candidates are registered as seen, clicked candidates are listed, impression time is recorded, and the in-view order has been shuffled.
- Adressa official documentation: light archives cover one week and ten weeks; light event fields include userId, Unix time, activeTime, session boundaries, category, document ID, title, keywords, entities, author, and publishTime; full article bodies require request access.
- Adressa documentation: userId can differentiate devices/browsers or identify subscription users, so it is not guaranteed to equal one human across devices.
- Globo.com data card: version 2 covers October 1-16, 2017 with about three million clicks, over one million sessions, 314,000 users, and more than 46,000 interacted articles; metadata and embeddings cover 364,047 published articles.
- Globo.com release caveat: original article text could not be shared for licensing reasons, and version 2 removes repeated clicks within sessions.

#### sources

- https://msnews.github.io/ — official MIND overview, downloads, scale, and dataset terms.
- https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news — released file schema and explicit body-text limitation.
- https://aclanthology.org/2020.acl-main.331.pdf — MIND dates, split construction, counts, and evaluation design.
- https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf — MIND custom research license terms.
- https://recsys.eb.dk/ — EB-NeRD scale, access form, and license link.
- https://recsys.eb.dk/dataset/ — EB-NeRD behavior, history, article, and artifact schemas.
- https://recsys.eb.dk/assets/pdf/general_license_terms.pdf — EB-NeRD dataset-use restrictions.
- https://arxiv.org/pdf/2410.03432 — EB-NeRD construction, counts, native ranking protocol, and comparison table.
- https://reclab.idi.ntnu.no/dataset/ — Adressa releases, sizes, access boundary, and CC BY-NC-SA 4.0 statement.
- https://reclab.idi.ntnu.no/dataset/cxdataset_v5.pdf — Adressa fields, dates, counts, missingness, and user-ID semantics.
- https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom — Globo.com v2 content, dates, counts, version changes, and license.
- https://github.com/gabrielspmoreira/chameleon_recsys — Globo/Adressa preprocessing, native session task, and no-text limitation.
- https://arxiv.org/abs/1904.10367 — Globo.com contextual session-recommendation protocol.

### Uncertain fields

- candidate_space
- exposure_handling
- feedback_and_exposure
- input_history
- is_direct_match
- release_and_license
- scale
- stable_user_identity
- venue_year
- verification_status

<a id="novelty-unexpectedness-serendipity-exploration-and-cross-domain-recommendation-definitions-evidence-and-reusable-datasets"></a>
## Novelty, unexpectedness, serendipity, exploration, and cross-domain recommendation: definitions, evidence, and reusable datasets

### Category: identity

#### name

> Novelty, unexpectedness, serendipity, exploration, and cross-domain recommendation: definitions, evidence, and reusable datasets

#### kind

Multi-paper adjacent-objective and dataset audit

#### official_sources

> https://doi.org/10.1145/2043932.2043955 ; https://doi.org/10.1145/2559952 ; https://doi.org/10.1016/j.knosys.2016.08.014 ; https://doi.org/10.24963/ijcai.2017/343 ; https://www.ijcai.org/proceedings/2017/0343.pdf ; https://doi.org/10.1145/3308558.3313469 ; https://github.com/greenblue96/Taobao-Serendipity-Dataset ; https://arxiv.org/abs/2008.07146 ; https://github.com/st-tech/zr-obp ; https://doi.org/10.1145/3511808.3557220 ; https://github.com/chongminggao/KuaiRec ; https://doi.org/10.1145/3511808.3557624 ; https://github.com/chongminggao/KuaiRand ; https://doi.org/10.1145/3511808.3557262 ; https://github.com/cjx96/C2DSR ; https://arxiv.org/abs/2210.10629 ; https://github.com/yuangh-x/2022-NIPS-Tenrec ; https://doi.org/10.1145/3604915.3608851 ; https://github.com/Ton-moy/surprise-and-serendipity ; https://amazon-reviews-2023.github.io/ ; https://arxiv.org/abs/2403.03952 ; https://doi.org/10.1145/3699682.3728325 ; https://arxiv.org/abs/2505.15440 ; https://arxiv.org/abs/2502.13539

### Category: task_semantics

#### input_history

> Offline novelty and unexpectedness metrics usually consume a user profile or prior consumed-item set plus a recommendation list. Topic-Level Bayesian Surprise consumes an ordered Goodreads history of books, ratings, and topic distributions. SERAL consumes recent behavior plus static, short-term, and long-term cognitive profiles. KuaiRand provides persistent-user standard-feed history for 2022-04-08 through 2022-04-21 followed by standard and randomly intervened feed events for 2022-04-22 through 2022-05-08. KuaiRec provides timestamped video-view sequences. Tenrec provides time-ordered interactions from four article/video scenarios, although absolute timestamps were removed. C2DSR merges two domain-specific item sequences in chronological order. EMCDR instead consumes source- and target-domain rating matrices rather than a temporal pre-transition history.

#### prediction_target

> The literature contains several distinct targets. Popularity novelty and profile-distance unexpectedness score an item or list, not a future behavior. Taobao Serendipity records immediate perceptions of one already exposed product. Topic-Level Bayesian Surprise detects whether a consumed book is topic-surprising and recommends a positively rated surprising book through similar-user histories. SERAL is the closest direct target: it generates the next serendipitous product and labels an item positive when it is clicked and its atomic category was absent from the user's valid exposures during the last 10 active days. KuaiRand, KuaiRec, Tenrec, Amazon Reviews'23, and Open Bandit Dataset do not ship a canonical future-new-interest label. C2DSR predicts the next item separately in a known target domain; it does not predict which domain the user enters next. EMCDR predicts ratings/rankings for artificially cold target-domain entities.

#### novel_interest_definition

> Novelty is often mere unfamiliarity or low popularity; unexpectedness is deviation from the user's expected or consideration set; serendipity normally adds usefulness, relevance, or positive affect to unexpectedness. The 2025 interview study defines experienced serendipity as an unintentional encounter that feels fortuitous, refreshing, and enriching. None of these concepts alone guarantees a persistent new interest. SERAL operationalizes a particularly relevant behavioral proxy: a click in an atomic category absent from recent valid exposures. Topic-Level Bayesian Surprise uses a manual label when a consumed book contains topics not seen, or only tangentially seen, earlier in that reader's history. Cross-domain recommendation normally defines a source/target transfer problem, not a first-ever interest in the target domain.

#### novelty_granularity_and_reference_window

> The audit distinguishes: global item novelty, such as inverse popularity; user-item novelty relative to consumed items; semantic unexpectedness relative to a profile or expected set; category-level surprise relative to an ordered history; and domain-level transition. SERAL uses atomic-category absence in valid exposures over the last 10 active days. The Topic-Level dataset compares each book from position 16 onward against all previous annotated books for that reader. Taobao asks whether one exposed item feels novel, unexpected, or pleasantly surprising, without releasing the history used by the participant. KuaiRand and KuaiRec permit item, tag, first-level category, or hierarchical-category definitions, but the supplementary taxonomy version must be frozen. C2DSR's Food-Kitchen, Movie-Book, and Entertainment-Education pairs are fixed domains, not novel-category labels.

#### interest_onset_definition

> SERAL uses a click as the relevance/onset event after recent category non-exposure. Topic-Level Bayesian Surprise combines post-factum surprise with the consumed book's rating; positive utility means a raw Goodreads rating above 3. Taobao separately records click, same-day purchase-or-cart, purchase intention, and perceived serendipity, but only for one exposure. KuaiRand can distinguish click/valid play, long view, like, follow, comment, forward, dislike, and profile entry; a defensible onset should require at least long view or another strong positive signal, not exposure alone. KuaiRec offers watch ratio but no native like label. Amazon reviews, Tenrec clicks/likes/shares/follows, and C2DSR interactions are observed actions, not proof that the underlying interest began at that event.

#### is_direct_match

> Direct one-step semantic proxy but not public: SERAL. Direct manual topic-surprise calibration but extremely small: Topic-Level Bayesian Surprise. Best public opportunity-aware candidate to construct the task: KuaiRand; useful near-full-exposure auxiliary: KuaiRec. Large temporal cross-domain candidate with severe exposure missingness: Amazon Reviews'23. Partial-support dataset: Tenrec, because it has true exposed negatives and scenario/category fields but no absolute timestamps. Subjective auxiliary only: Taobao Serendipity. Exposure/OPE method-only: Open Bandit Dataset. Cross-domain next-item method-only: C2DSR. Static cold-start transfer, not onset forecasting: EMCDR/Douban. Popularity novelty, distance unexpectedness, and list diversity metrics are evaluation components, not future-interest ground truth.

### Category: method_and_evaluation

#### method

> Representative proxy methods include inverse-popularity/self-information novelty, profile or consideration-set distance for unexpectedness, and serendipity as unexpectedness combined with relevance/utility. Topic-Level Bayesian Surprise measures KL divergence between posterior and prior preference distributions after a rating, with serendipity equal to actual centered rating times surprise. SERAL generates titles with an aligned LLM, retrieves items, and mixes a nearline serendipity channel into an industrial ranker. KuaiRand and Open Bandit support propensity-aware or randomized-exposure evaluation. EMCDR maps source latent factors into a target domain for cold-start entities. C2DSR jointly models single-domain and merged cross-domain sequences with graph, attention, next-item, and contrastive objectives.

#### evaluation_protocol

> Offline novelty and serendipity metrics must be validated against user judgments or realized relevance, not accepted by construction. Topic-Level Bayesian Surprise labels 449 books from four readers, leaves one reader out for tuning on the other three, and reports surprise-detection and serendipitous-recommendation precision/recall/F1. SERAL uses 19,141 private positive samples, a 9:1 split, and 99 sampled non-serendipitous candidates per test positive; it also reports a two-week online traffic-bucket A/B test and a longer small-traffic baseline comparison. KuaiRand enables a stronger protocol: train on standard history, test only on later randomized exposures within the 7,583-item random pool, score click/long-view/strong feedback, and estimate policy values with known randomization. KuaiRec can evaluate within its 99.6%-dense 1,411-by-3,327 small matrix. C2DSR corrected an earlier temporal leak by placing users' latest sequences into validation/test, but still ranks one positive against 999 negatives and preselects users active in both domains. CrossInterest should report category-level recall/calibration plus repeat, new-item-known-category, new-category-one-off, and persistent-new-category strata.

#### generalization_axis

> SERAL evaluates later events for existing industrial users and categories, not unseen categories or users. Topic-Level Bayesian Surprise tests four existing readers with a common topic model. KuaiRand supports temporal future and randomized-exposure generalization for the same users and a fixed random candidate pool. KuaiRec supports dense user-item evaluation but has a short, platform-specific video domain. Amazon Reviews'23 can support time-forward, cross-category, user-disjoint, item-cold, and category-cold variants if explicitly constructed. Tenrec supports scenario transfer with overlapping users/items, but not calendar-time extrapolation because timestamps are removed. C2DSR evaluates later dual-domain sequences for users already active in both domains. EMCDR randomly hides target-domain ratings, a simulated cold-start axis rather than observed domain entry.

#### main_findings_and_limits

> The closest operational precedent is SERAL, whose 10-active-day atomic-category non-exposure plus click label is almost a short-window future-new-category click. Its two-week online test reports +5.70 percentage points in serendipity exposure ratio, +29.56% serendipity clicks, and +27.60% serendipity transaction volume, but the label is private, sampled-candidate offline evaluation is optimistic, and neither one click nor aggregate transactions establish per-user persistent interest. Topic-Level Bayesian Surprise demonstrates that user-specific temporal topic surprise can be manually labeled, but only four users are annotated. KuaiRand uniquely joins history, semantics, multiple outcomes, and randomized exposure, making it the highest-value public auxiliary benchmark. KuaiRec reduces missing exposure but is short-horizon and watch-ratio-only. Amazon Reviews'23 is the best large-scale cross-domain raw corpus, yet its reviews are MNAR positives. Tenrec offers exposed negatives and multi-feedback, but no timestamps. C2DSR and EMCDR show that cross-domain accuracy gains do not imply domain-onset prediction: they condition on users already present in both domains or synthetically hide target data.

### Category: dataset_schema

#### timestamps_and_order

> KuaiRand supplies millisecond timestamps and separate standard-history and later standard/random log files. KuaiRec supplies human-readable time, date, Unix timestamp, and watch duration. Amazon Reviews'23 supplies second-or-finer Unix timestamps across 1996-2023. Topic-Level Goodreads histories contain time strings and timestamps. C2DSR merges source-domain and target-domain interactions chronologically and publishes a corrected latest-sequence split. Tenrec removes timestamps but preserves interactions in time order over 2021-09-17 through 2021-12-07, preventing exact horizon and inactivity definitions. Taobao Serendipity is one survey exposure per user; Open Bandit spans seven days with timestamps but no stable user ID.

#### left_censoring_and_history_coverage

> Every first observed category or domain is only new relative to the recorded window. KuaiRand contributes two weeks of pre-period standard history before its 17-day intervention period, still far from lifetime history. KuaiRec, Tenrec, and C2DSR have short or filtered histories; C2DSR explicitly retains only users with at least three events in each domain, eliminating true first-domain entrants. Amazon's long span helps but reviews omit most browsing and purchases and users may predate 1996. Topic-Level selects readers with at least 100 ratings to reduce missing-reading bias, yet unreported books remain possible. SERAL deliberately uses a short 10-active-day exposure window, so returning interests are labeled serendipitous after inactivity.

### Category: crossinterest_fit

#### construct_h_f_ynew

> Recommended primary construction: KuaiRand H(u) equals all standard-feed positive behavior before 2022-04-22; F(u) equals later randomized-exposure outcomes, optionally followed by later standard-feed confirmation; Y_new(u) contains category nodes with a strong positive randomized outcome that were absent from H(u). Restrict candidates to the random pool, freeze the category map, and create separate click, long-view, and later-persistence labels. KuaiRec can provide a dense auxiliary version using timestamped watch ratios, with H before a per-user cutoff and F after it. Amazon Reviews'23 can create large temporal cross-domain H/F sets from verified reviews, but Y_new is positive-unlabeled and cannot treat unreviewed domains as negatives. Tenrec can form order-based H/F and category novelty but not calendar windows. Topic-Level annotations can calibrate semantic surprise; Taobao can calibrate perceived novelty/serendipity. SERAL provides the closest rule template but no public rows. Do not use C2DSR's prefiltered dual-domain users to estimate first domain entry, and do not use Open Bandit without user identity.

#### leakage_and_missingness_risks

> Major risks are equating low popularity with user novelty; using predicted relevance inside both the label and metric; treating non-exposure as dislike; calling the first recorded action lifetime onset; allowing future taxonomy, captions, reviews, or corpus-wide LDA topics into H; sampling negatives that were unavailable or already known; letting the same user cross splits; preselecting users who later enter the target domain; and measuring only the easy conditional task after revealing the target domain. SERAL's click-based label can include accidental clicks and returning interests after 10 inactive days. Amazon/Goodreads reviews are MNAR. KuaiRand's random pool is not the full catalog, and only about 0.37% of feed events are randomized. KuaiRec's near-full viewing intervention may alter natural preference and fatigue. C2DSR's 999-negative evaluation and both-domain user filter cannot establish open-world domain discovery.

#### recommended_role

> Use SERAL as the closest label-definition and industrial-evaluation precedent, not as a reusable dataset. Audit KuaiRand first as the primary public opportunity-aware robustness benchmark; use KuaiRec as a dense-exposure auxiliary. Use Amazon Reviews'23 for a large cross-domain positive-unlabeled temporal benchmark with conservative claims. Use Tenrec as a multi-feedback/exposed-negative scenario-transfer benchmark when order-only time is acceptable. Use Topic-Level Bayesian Surprise and Taobao Serendipity for small human-grounded calibration of semantic surprise and perceived serendipity. Use Open Bandit only to validate propensity/OPE machinery. Use C2DSR and EMCDR as method baselines and counterexamples showing that cross-domain transfer is not new-domain onset prediction.

#### priority

> Highest dataset-audit priority: KuaiRand, followed by KuaiRec and Amazon Reviews'23. High conceptual priority: SERAL's 10-active-day atomic-category label and Topic-Level temporal surprise annotation. Medium: Tenrec and Taobao Serendipity. Method-only medium: C2DSR. Low/direct-task reject: EMCDR/Douban and Open Bandit as standalone future-interest datasets, although both remain useful for transfer and exposure methodology respectively.

#### repeat_explore_and_persistence_audit

> For every dataset, split outcomes into repeated item, new item in known leaf category, new leaf category under a known ancestor, new top-level category/domain, return after the reference window, and persistent new category. A serendipity metric must also separate unexpected-but-irrelevant exposures from unexpected positive outcomes. On KuaiRand, report randomized exposed negatives and positives separately and require a later same-category strong event for persistence. On KuaiRec, report first strong watch and later same-category strong watches. On Amazon and Tenrec, never interpret missing events as exposed negatives. SERAL already identifies a 10-day category return/new event, but its published per-item label needs a later-adoption audit. Topic-Level and Taobao data cannot measure persistence.

### Category: evidence

#### evidence_quotes_or_schema_facts

> SERAL Definition 1 states that a serendipitous item 'must be clicked' and its atomic category must be absent from valid exposures over the last N active days; N=10 in the deployed scenario. KuaiRand's official README reports random replacement from a fixed pool, an is_rand field, millisecond timestamps, 12 feedback signals, 27,285 users, and 1,186,059 random events. KuaiRec's official README defines fully observed as almost every selected user viewing every selected video and reports 99.6% density for 1,411 by 3,327. Tenrec's paper states that timestamps were removed but interaction order retained, and gives true exposed negatives plus overlapping users/items across four scenarios. The Topic-Level paper manually labels a book surprising when its topics did not appear, or appeared only tangentially, in prior books; only 449 books from four users are labeled. C2DSR reports that about 60% of older processed validation/test samples preceded training samples, then corrects the split using latest sequences; it still filters users to both domains and at least three events per domain. EMCDR randomly removes target-domain ratings to simulate cold start. Open Bandit records exact propensities under uniform-random and Thompson-sampling policies but omits a stable user ID. Taobao's CSV contains one exposed item per user, not a history.

#### sources

> Definitions: https://doi.org/10.1145/2043932.2043955 ; https://doi.org/10.1145/2559952 ; https://doi.org/10.1016/j.knosys.2016.08.014 ; https://doi.org/10.1145/3699682.3728325 ; https://arxiv.org/abs/2505.15440 . Direct industrial proxy: https://arxiv.org/abs/2502.13539 . Human-grounded datasets: https://doi.org/10.1145/3308558.3313469 ; https://github.com/greenblue96/Taobao-Serendipity-Dataset ; https://doi.org/10.1145/3604915.3608851 ; https://github.com/Ton-moy/surprise-and-serendipity . Random/near-full exposure datasets: https://doi.org/10.1145/3511808.3557624 ; https://github.com/chongminggao/KuaiRand ; https://doi.org/10.1145/3511808.3557220 ; https://github.com/chongminggao/KuaiRec ; https://arxiv.org/abs/2008.07146 ; https://github.com/st-tech/zr-obp . Cross-domain data/methods: https://amazon-reviews-2023.github.io/ ; https://arxiv.org/abs/2403.03952 ; https://arxiv.org/abs/2210.10629 ; https://github.com/yuangh-x/2022-NIPS-Tenrec ; https://doi.org/10.1145/3511808.3557262 ; https://github.com/cjx96/C2DSR ; https://doi.org/10.24963/ijcai.2017/343 ; https://www.ijcai.org/proceedings/2017/0343.pdf .

### Uncertain fields

- candidate_space
- exposure_handling
- feedback_and_exposure
- item_semantics
- persistence_signal
- release_and_license
- scale
- stable_user_identity
- taxonomy_and_opportunity_drift
- venue_year
- verification_status

<a id="randomized-exposure-impression-slate-and-explicit-serendipity-datasets-yahoo-r6-open-bandit-dataset-coat-kuairand-finn-no-slates-contentwise-impressions-and-taobao-serendipity"></a>
## Randomized-exposure, impression-slate, and explicit serendipity datasets: Yahoo! R6, Open Bandit Dataset, Coat, KuaiRand, FINN.no Slates, ContentWise Impressions, and Taobao Serendipity

### Category: identity

#### name

> Randomized-exposure, impression-slate, and explicit serendipity datasets: Yahoo! R6, Open Bandit Dataset, Coat, KuaiRand, FINN.no Slates, ContentWise Impressions, and Taobao Serendipity

#### kind

Comparative audit of causal and subjective auxiliary datasets

#### venue_year

> Yahoo! R6: Yahoo! Webscope logged-bandit release associated with 2010 contextual-bandit evaluation; Open Bandit Dataset: NeurIPS 2020 workshop/arXiv 2020; Coat: ICML 2016; KuaiRand: CIKM 2022, current Zenodo record 10439422; FINN.no Slates: arXiv 2021 and Data Mining and Knowledge Discovery 2022; ContentWise Impressions: CIKM 2020; Taobao Serendipity: WWW 2019 and UMAP 2020.

#### official_sources

- **Yahoo_R6**
  https://webscope.sandbox.yahoo.com/catalog.php?datatype=r, https://arxiv.org/abs/1003.0146
- **Open_Bandit_Dataset**
  https://github.com/st-tech/zr-obp/tree/master/obd, https://research.zozo.com/data.html, https://arxiv.org/abs/2008.07146
- **Coat**
  https://arxiv.org/abs/1602.05352
- **KuaiRand**
  https://github.com/chongminggao/KuaiRand, https://zenodo.org/records/10439422, https://doi.org/10.1145/3511808.3557624
- **FINN_no_Slates**
  https://github.com/finn-no/recsys-slates-dataset, https://arxiv.org/abs/2104.15046
- **ContentWise_Impressions**
  https://github.com/ContentWise/contentwise-impressions, https://arxiv.org/abs/2008.01212
- **Taobao_Serendipity**
  https://github.com/greenblue96/Taobao-Serendipity-Dataset, https://www.comp.hkbu.edu.hk/~lichen/download/TaoBao_Serendipity_Dataset.html, https://doi.org/10.1145/3308558.3313469

### Category: task_semantics

#### prediction_target

> The native outcomes are click under a logged action (Yahoo R6 and Open Bandit), multi-signal response to an intervened video (KuaiRand), click/no-click within a slate (FINN.no), media interaction after an impression (ContentWise), a 1-5 rating under random coat assignment (Coat), or one-shot perceived novelty/serendipity and same-day click/cart-or-purchase (Taobao). None natively labels sustained future new interest. CrossInterest must derive a semantic onset and a separate later-persistence outcome.

#### novel_interest_definition

> Only Taobao directly asks whether the recommendation is novel, unexpected, diverse relative to prior purchases/recommendations, and a pleasant surprise; those are perceptions, not longitudinal adoption. For KuaiRand, FINN.no, and ContentWise, define observed exploration as positive response to a category absent from the pre-cutoff observed history. Random assignment or exposure alone is opportunity, never evidence of interest. OBD, Yahoo R6, and Coat are primarily causal-calibration data rather than direct novel-interest corpora.

#### novelty_granularity_and_reference_window

- **KuaiRand:** Prefer the released video category hierarchy, report novelty separately at leaf and ancestor levels, and use 2022-04-08 to 04-21 as the frozen reference history before 04-22 to 05-08 outcomes. Caption-embedding clusters are a sensitivity analysis.
- **FINN_no_Slates:** Item-group novelty among 290 category-plus-geography groups relative to earlier retained interactions; item novelty must not be called interest novelty.
- **ContentWise_Impressions:** Series or coarse item-type novelty relative to earlier interactions; the public semantic taxonomy is too coarse for strong topic-level claims.
- **Coat:** Novel item rating relative to the participant's 24 self-selected coats, not a temporal new category; only four item covariates support coarse grouping.
- **Taobao_Serendipity:** Self-reported item novelty and category difference relative to remembered prior purchases/recommendations in a single survey. No public reference-window events are available.
- **OBD_and_Yahoo:** No longitudinal reference window can be constructed without stable user identity.

#### interest_onset_definition

- **KuaiRand:** First positive response to a history-absent category under either standard or random exposure; pre-register a strong-positive rule such as long_view, like, follow, or sufficient play fraction rather than treating exposure as onset.
- **FINN_no_Slates:** First click on a visible item from a history-absent group. Search and recommendation onsets must be stratified because search expresses prior intent.
- **ContentWise_Impressions:** First post-cutoff interaction with a history-absent series/type after a logged recommendation impression; direct-link and non-direct-link impressions should remain separate.
- **Coat:** A favorable rating on a randomly assigned coat is preference evidence, not longitudinal onset.
- **Taobao_Serendipity:** Clicked or same-day cart/purchase is an immediate outcome; perceived novelty or purchase intention alone is not behavioral onset.
- **OBD_and_Yahoo:** A click is an immediate contextual response and cannot be tied to a user's prior semantic history.

#### is_direct_match

- **KuaiRand:** Closest dataset-only match: stable sequences, a pre-period, randomized interventions, timestamps, semantic supplements, and later rich feedback make a derived H-to-F benchmark feasible.
- **FINN_no_Slates:** Adjacent exposure-aware sequence dataset. It has complete visible slates and stable short histories but production exposure is not randomized and the public reduction keeps only up to 20 interactions.
- **ContentWise_Impressions:** Adjacent impression-aware sequence dataset with account history and full recommendation lists, but non-random exposure and coarse semantics limit causal novelty claims.
- **Open_Bandit_Dataset:** Method-only match for off-policy and propensity validation; no persistent user histories.
- **Yahoo_R6:** Method-only logged-bandit match; no stable longitudinal user identity.
- **Coat:** Method-only selection-bias calibration with randomized ratings, not a sequential future-interest task.
- **Taobao_Serendipity:** Auxiliary subjective validation only; no public history, exposure slate, or repeated outcome.
- **newer_user_studies:** No release-level verified downloadable direct match was found.

#### persistence_signal

- **KuaiRand:** Yes: later timestamped clicks, long views, likes, follows, comments, or repeated category consumption can be measured after onset within the remaining future window.
- **FINN_no_Slates:** Limited: later clicks in the same item group among the remaining maximum 20 interactions; no dwell or long-term follow-up.
- **ContentWise_Impressions:** Possible as later interaction with the same newly entered series/type during roughly three months, but account sharing weakens person-level interpretation.
- **Yahoo_R6_Open_Bandit_Coat_Taobao:** No defensible longitudinal persistence signal in the public unit of observation.

### Category: method_and_evaluation

#### method

> For CrossInterest, use KuaiRand as a two-stage benchmark: estimate exposure/opportunity with the intervention flag and predict engagement-qualified semantic onset from frozen history, then predict later same-category adoption. Use inverse-propensity or doubly robust estimators only when propensities are actually known; do not manufacture propensities for KuaiRand, FINN.no, or ContentWise. Use Open Bandit and Yahoo R6 to unit-test OPE code, Coat to test MNAR-rating corrections, and Taobao to test whether model novelty scores align with explicit perceptions.

#### evaluation_protocol

> Primary KuaiRand protocol: H is 2022-04-08 to 04-21 standard history; 04-22 to 05-08 is split chronologically into onset and persistence subwindows. Report full-taxonomy and exposed-opportunity ranking, random-versus-standard strata, strong-positive thresholds, macro category recall, calibration, and user-clustered uncertainty. FINN.no and ContentWise are temporal exposure-conditioned robustness tests with search/recommendation or direct/non-direct strata. OBD/Yahoo use replay/OPE metrics, Coat uses random-test rating error or ranking, and Taobao uses correlation/calibration against nine Likert variables plus click and purchase/cart, never as primary predictive accuracy.

#### generalization_axis

> KuaiRand supports seen-user temporal future, randomized-versus-standard policy shift, item/category onset, and scenario-policy shift. FINN.no and ContentWise support seen-identity temporal evolution under production exposure. OBD/Yahoo test policy generalization without user-history personalization. Coat tests observational-to-random assignment shift. Taobao tests cross-user agreement with explicit subjective judgments.

#### main_findings_and_limits

> KuaiRand is the only audited release that jointly supplies persistent IDs, a clean pre-period, timestamped future responses, randomized interventions, and usable semantic metadata. Its intervention is sparse, event propensities and complete slates are absent, and a random long view still only proves response to an opportunity, not durable interest. FINN.no and ContentWise add valuable real impression context but cannot identify causal exposure effects without stronger assumptions. OBD and Yahoo are excellent OPE checks but structurally cannot model history-driven new interest. Coat separates MNAR from randomized ratings but is static. Taobao supplies unusually direct subjective novelty and serendipity labels, yet its public CSV is one-shot and has only a tiny purchased/cart-positive subset.

### Category: dataset_schema

#### stable_user_identity

- **KuaiRand:** Yes, explicit re-indexed user_id joins all pre-period, standard-future, random-future, and user-feature files.
- **FINN_no_Slates:** Yes within the released first 20 interactions, via anonymized userId.
- **ContentWise_Impressions:** Yes within the collection, but the paper states users are accounts and may represent multiple household members.
- **Coat:** Participant identity joins the two static rating matrices, but there is no behavioral timeline.
- **Taobao_Serendipity:** One stable row identifier per surveyed user, but no repeated public events.
- **Open_Bandit_and_Yahoo_R6:** No stable user identifier suitable for longitudinal history.

#### left_censoring_and_history_coverage

> KuaiRand observes only two pre-period weeks and seventeen future days, so an apparently new category may be rediscovery of an older/off-platform interest. FINN.no keeps only first 20 logged interactions and drops users with fewer than 10, while 90% of no-click slates are removed. ContentWise begins at a fixed date and uses accounts, so prior consumption and person identity are incomplete. Coat's 24 self-selected coats are a tiny preference sample. Taobao withholds the histories used during survey selection. OBD/Yahoo have no longitudinal coverage at all.

#### feedback_and_exposure

> KuaiRand has is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, play time, duration, profile/comment dwell, profile entry, and is_rand. FINN.no has visible slate, scroll length, clicked item/index, no-click, and interaction type. ContentWise has ordered recommended series lists, row position/list length, and timestamped item/series interactions with recommendation_id. OBD has click, position, and action_prob. Yahoo has shown article and click [uncertain current archive]. Coat has 1-5 ratings. Taobao has nine 1-5 perception variables, click, and same-day purchase/cart.

#### candidate_space

- **KuaiRand:** The 7,583-video random pool is known at release level, but the complete candidate slate and exact time-varying eligibility of standard recommendations cannot be reconstructed.
- **FINN_no_Slates:** The actually visible slate up to scroll length is known; the full contemporaneous marketplace inventory and pre-scroll slate tail beyond 25 are not.
- **ContentWise_Impressions:** Actual recommendation lists are recorded, but not a complete time-varying OTT catalog with all eligibility rules.
- **Open_Bandit_and_Yahoo:** Logged action sets/candidate pools support bandit replay, but not longitudinal semantic candidates.
- **Coat:** All 300 coats define the experiment universe; each user randomly rates 16 in the unbiased phase.
- **Taobao:** Only the selected recommendation is public; the candidate set is absent.

#### taxonomy_and_opportunity_drift

> KuaiRand's caption/category supplements may postdate the logs and must be version-frozen; video availability and random-pool membership are only partially represented. FINN.no groups combine category and geography, while classifieds inventory expires rapidly and the 25-item cap changes opportunity. ContentWise content availability, account entitlement, placement, and catalog changes are incompletely observed. Coat has a fixed 300-item experiment. Taobao's anonymous categories have no version or inventory timeline. OBD/Yahoo action sets vary by campaign/event but have no longitudinal semantic taxonomy.

### Category: crossinterest_fit

#### construct_h_f_ynew

- **KuaiRand:** Feasible. Define H(u) from all standard interactions on 04-08 to 04-21 that satisfy a frozen positive rule; define F(u) from engagement-qualified interactions on 04-22 to 05-08; compute Y_new(u)=semantic_units(F)-semantic_units(H). Preserve is_rand, scenario, exposure time, and feedback strength. Reserve later days after first onset for persistence.
- **FINN_no_Slates:** Partially feasible. Use early ordered slate clicks for H and later clicks for F, then group set difference. Condition evaluation on actual visible opportunities and separate search from recommendation; lack of timestamps, sparse 20-step histories, and observational policy make this a robustness task, not a causal benchmark.
- **ContentWise_Impressions:** Partially feasible. Sort account interactions, use an early burn-in H and later F, and define new series/type units. Link to preceding recommendation impressions where possible; keep unknown exposure positive-unlabeled and avoid person-level claims.
- **Coat:** Not a strict temporal construction. The 24 chosen ratings can proxy H and favorable random ratings can proxy unseen-item response, but F-H is neither future nor category adoption.
- **Taobao_Serendipity:** Not feasible from the public CSV because H, a future sequence F, and repeated outcomes are missing.
- **Open_Bandit_and_Yahoo_R6:** Not feasible because stable user identity and longitudinal H/F are absent.

#### leakage_and_missingness_risks

> Major risks are treating randomized exposure as interest, treating a one-off click as adoption, using future aggregate video features, leaking post-period captions/categories, ignoring sparse intervention, reconstructing false propensities, and labeling unexposed items negative. KuaiRand also has two weeks of left censoring and policy/scenario confounding. FINN.no has no-click downsampling, search-intent confounding, a 20-step cap, and fast inventory turnover. ContentWise has household accounts, observational placement, and coarse semantics. Coat has small static rating matrices. Taobao has selection by an undisclosed recommender, no released history, and only 10 purchase/cart positives.

#### recommended_role

- **KuaiRand:** Primary causal/exposure-aware auxiliary benchmark and the only high-priority candidate for a derived future novel-interest task.
- **FINN_no_Slates:** Exposure-aware sequential robustness benchmark, especially for repeat-versus-explore under visible slates.
- **ContentWise_Impressions:** Secondary impression-aware temporal robustness benchmark at series/type level.
- **Open_Bandit_and_Yahoo_R6:** OPE implementation validation only.
- **Coat:** MNAR and randomized-preference calibration only.
- **Taobao_Serendipity:** Auxiliary subjective novelty/serendipity validation only.
- **newer_user_studies:** Reject until a downloadable, schema-verifiable release is identified.

#### priority

> 1) KuaiRand: high. 2) FINN.no Slates: medium-high for exposure-aware robustness, not causal novelty. 3) ContentWise Impressions: medium. 4) Open Bandit Dataset: high only for OPE code validation. 5) Coat: medium only for selection-bias calibration. 6) Taobao Serendipity: medium only for subjective validation. 7) Yahoo R6: low until current access is re-verified, and method-only even if obtained.

#### repeat_explore_and_persistence_audit

> Report three disjoint classes at each future event: repeat semantic unit already in H, first observed explore unit absent from H, and new item within an old unit. An exposure with no positive response is opportunity, not explore; the first positive is onset, not persistence. For KuaiRand, require a later disjoint-window positive in the same semantic unit and stratify onset by random versus standard exposure and by click-only versus strong engagement. For FINN.no, count persistence only on later slates and separate search from recommendation. For ContentWise, require later series/type interaction and report account-level limitations. OBD, Yahoo, Coat, and Taobao cannot support this audit longitudinally and should be reported as such.

### Category: evidence

#### evidence_quotes_or_schema_facts

- OBD official README: about 26M rows; each row contains a user impression, selected action, true propensity score, and click; policies are random and Bernoulli Thompson Sampling across all/men/women campaigns.
- KuaiRand official README: each recommended video had equal probability of replacement by a random-pool video; about 0.37% of final interactions were replaced; explicit user IDs, timestamps, 12 feedback signals, and thousands of interactions per user are provided.
- KuaiRand files separate standard logs for 2022-04-08 to 04-21 from random and standard logs for 04-22 to 05-08 for the same users.
- FINN.no paper/repository: 37.4M interactions, approximately 2.3M users, 1.3M items, 290 groups, and all visible items up to scroll length; approximately 80% of slates are search and 20% recommendation.
- FINN.no reduction: retain 10% of no-click slates, first 20 interactions per user, users with at least 10 interactions, and at most 25 visible items.
- ContentWise release separates 307,453 impressions linked directly to interactions from 23,342,617 impressions without a direct link, alongside 10,457,810 interactions.
- Coat experiment: each of 290 users rated 24 self-selected coats and then 16 uniformly random coats from a 300-coat universe.
- Taobao author CSV: 11,383 one-row users and 34 columns, including nine perception ratings, Clicked, Purchased, CEI-II, and TIPI; Purchased includes same-day purchase or cart addition.

### Uncertain fields

- exposure_handling
- input_history
- item_semantics
- release_and_license
- scale
- sources
- timestamps_and_order
- verification_status
