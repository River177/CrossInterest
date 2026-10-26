# Future Novel-Interest Baseline Results (Draft)

This draft is generated directly from the baseline manifest. It contains no invented examples, translations, or numerical claims.

- Reference baseline: `global_popularity`
- Rule version: `future-interest-baselines-v1`
- Manifest: `artifacts/future-novel-interest/baselines-v1/manifest.json`
- Output directory: `docs/experiments/2026-08-12/future_novel_interest_baselines_artifacts`

## Track results

### `exposure_restricted`

| Condition | N | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@20 | Macro Recall@1 | Macro Recall@5 | Macro Recall@10 | Macro Recall@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global_popularity | 62,687 | 0.1197 | 0.4003 | 0.5883 | 0.7957 | 0.2281 | 0.3201 | 0.3920 | 0.4604 | 0.2281 | 0.6304 | 0.8098 | 0.9360 | 0.4082 | 0.0201 | 0.0994 | 0.1901 | 0.3627 |
| conditional_frequency | 62,687 | 0.1202 | 0.3879 | 0.5838 | 0.7986 | 0.2271 | 0.3081 | 0.3835 | 0.4544 | 0.2271 | 0.6131 | 0.8060 | 0.9378 | 0.3973 | 0.0205 | 0.0992 | 0.1934 | 0.3694 |
| conditional_lift | 62,687 | 0.0605 | 0.2289 | 0.3738 | 0.5889 | 0.0929 | 0.1663 | 0.2210 | 0.2890 | 0.0929 | 0.3516 | 0.5499 | 0.7769 | 0.2236 | 0.1403 | 0.3558 | 0.5049 | 0.7029 |
| embedding_cosine | 62,687 | 0.0987 | 0.3350 | 0.5123 | 0.7344 | 0.1753 | 0.2589 | 0.3266 | 0.3992 | 0.1753 | 0.5300 | 0.7354 | 0.9055 | 0.3395 | 0.0545 | 0.2205 | 0.3688 | 0.5984 |
| multilabel_topic_logistic | 62,687 | 0.1048 | 0.3515 | 0.5300 | 0.7504 | 0.1872 | 0.2735 | 0.3416 | 0.4139 | 0.1872 | 0.5534 | 0.7499 | 0.9137 | 0.3548 | 0.0734 | 0.2766 | 0.4483 | 0.6834 |
| multilabel_embedding_logistic | 62,687 | 0.1211 | 0.3935 | 0.5846 | 0.7997 | 0.2220 | 0.3118 | 0.3850 | 0.4566 | 0.2220 | 0.6142 | 0.8027 | 0.9397 | 0.3973 | 0.0696 | 0.2562 | 0.4043 | 0.6312 |

Bootstrap deltas versus the reference (observed delta with percentile interval):

| Condition | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conditional_frequency | +0.0005 [-0.0011, +0.0020] | -0.0123 [-0.0141, -0.0106] | -0.0046 [-0.0061, -0.0030] | +0.0030 [+0.0021, +0.0039] | -0.0011 [-0.0038, +0.0017] | -0.0120 [-0.0133, -0.0108] | -0.0085 [-0.0095, -0.0075] | -0.0060 [-0.0069, -0.0052] | -0.0011 [-0.0038, +0.0017] | -0.0173 [-0.0200, -0.0149] | -0.0038 [-0.0057, -0.0019] | +0.0018 [+0.0008, +0.0027] | -0.0109 [-0.0125, -0.0093] |
| conditional_lift | -0.0592 [-0.0617, -0.0567] | -0.1713 [-0.1751, -0.1676] | -0.2145 [-0.2188, -0.2107] | -0.2068 [-0.2106, -0.2033] | -0.1352 [-0.1391, -0.1312] | -0.1538 [-0.1569, -0.1508] | -0.1710 [-0.1739, -0.1681] | -0.1714 [-0.1741, -0.1687] | -0.1352 [-0.1391, -0.1312] | -0.2788 [-0.2840, -0.2736] | -0.2598 [-0.2650, -0.2552] | -0.1591 [-0.1631, -0.1555] | -0.1846 [-0.1880, -0.1811] |
| embedding_cosine | -0.0210 [-0.0235, -0.0184] | -0.0652 [-0.0687, -0.0616] | -0.0760 [-0.0794, -0.0727] | -0.0613 [-0.0640, -0.0584] | -0.0528 [-0.0570, -0.0486] | -0.0612 [-0.0643, -0.0582] | -0.0654 [-0.0680, -0.0627] | -0.0612 [-0.0637, -0.0587] | -0.0528 [-0.0570, -0.0486] | -0.1004 [-0.1053, -0.0957] | -0.0743 [-0.0782, -0.0703] | -0.0305 [-0.0332, -0.0278] | -0.0688 [-0.0723, -0.0652] |
| multilabel_topic_logistic | -0.0149 [-0.0174, -0.0124] | -0.0487 [-0.0523, -0.0452] | -0.0584 [-0.0621, -0.0548] | -0.0452 [-0.0482, -0.0424] | -0.0409 [-0.0448, -0.0368] | -0.0466 [-0.0496, -0.0438] | -0.0504 [-0.0531, -0.0477] | -0.0466 [-0.0490, -0.0441] | -0.0409 [-0.0448, -0.0368] | -0.0770 [-0.0820, -0.0723] | -0.0598 [-0.0639, -0.0559] | -0.0223 [-0.0250, -0.0195] | -0.0534 [-0.0568, -0.0501] |
| multilabel_embedding_logistic | +0.0014 [-0.0013, +0.0041] | -0.0068 [-0.0105, -0.0033] | -0.0037 [-0.0070, -0.0003] | +0.0041 [+0.0015, +0.0066] | -0.0061 [-0.0105, -0.0017] | -0.0083 [-0.0114, -0.0053] | -0.0070 [-0.0097, -0.0042] | -0.0038 [-0.0062, -0.0014] | -0.0061 [-0.0105, -0.0017] | -0.0162 [-0.0209, -0.0114] | -0.0071 [-0.0108, -0.0033] | +0.0037 [+0.0013, +0.0061] | -0.0109 [-0.0145, -0.0073] |

### `full_taxonomy`

| Condition | N | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@20 | Macro Recall@1 | Macro Recall@5 | Macro Recall@10 | Macro Recall@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global_popularity | 62,687 | 0.0668 | 0.2727 | 0.4377 | 0.6358 | 0.1486 | 0.2155 | 0.2785 | 0.3426 | 0.1486 | 0.4933 | 0.6778 | 0.8311 | 0.3043 | 0.0116 | 0.0636 | 0.1246 | 0.2310 |
| conditional_frequency | 62,687 | 0.0432 | 0.2363 | 0.4118 | 0.6197 | 0.0989 | 0.1804 | 0.2471 | 0.3143 | 0.0989 | 0.4438 | 0.6520 | 0.8198 | 0.2619 | 0.0082 | 0.0583 | 0.1212 | 0.2314 |
| conditional_lift | 62,687 | 0.0000 | 0.0012 | 0.0070 | 0.0311 | 0.0000 | 0.0007 | 0.0030 | 0.0109 | 0.0000 | 0.0032 | 0.0185 | 0.0806 | 0.0067 | 0.0000 | 0.0334 | 0.0910 | 0.2080 |
| embedding_cosine | 62,687 | 0.0266 | 0.1020 | 0.1746 | 0.3026 | 0.0625 | 0.0830 | 0.1107 | 0.1516 | 0.0625 | 0.2222 | 0.3514 | 0.5293 | 0.1456 | 0.0239 | 0.0995 | 0.1748 | 0.3035 |
| multilabel_topic_logistic | 62,687 | 0.0250 | 0.1012 | 0.1708 | 0.2813 | 0.0594 | 0.0815 | 0.1079 | 0.1432 | 0.0594 | 0.2194 | 0.3399 | 0.4960 | 0.1404 | 0.0322 | 0.1170 | 0.2011 | 0.3301 |
| multilabel_embedding_logistic | 62,687 | 0.0392 | 0.1503 | 0.2585 | 0.4307 | 0.0964 | 0.1236 | 0.1645 | 0.2196 | 0.0964 | 0.3100 | 0.4671 | 0.6539 | 0.2034 | 0.0358 | 0.1404 | 0.2312 | 0.3736 |

Bootstrap deltas versus the reference (observed delta with percentile interval):

| Condition | Recall@1 | Recall@5 | Recall@10 | Recall@20 | NDCG@1 | NDCG@5 | NDCG@10 | NDCG@20 | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conditional_frequency | -0.0236 [-0.0252, -0.0219] | -0.0364 [-0.0383, -0.0346] | -0.0260 [-0.0278, -0.0242] | -0.0161 [-0.0179, -0.0143] | -0.0496 [-0.0525, -0.0465] | -0.0351 [-0.0363, -0.0339] | -0.0314 [-0.0325, -0.0303] | -0.0283 [-0.0293, -0.0273] | -0.0496 [-0.0525, -0.0465] | -0.0495 [-0.0522, -0.0468] | -0.0258 [-0.0282, -0.0236] | -0.0113 [-0.0132, -0.0095] | -0.0424 [-0.0440, -0.0408] |
| conditional_lift | -0.0668 [-0.0683, -0.0653] | -0.2715 [-0.2742, -0.2688] | -0.4307 [-0.4339, -0.4277] | -0.6047 [-0.6079, -0.6016] | -0.1486 [-0.1513, -0.1458] | -0.2148 [-0.2169, -0.2126] | -0.2755 [-0.2775, -0.2734] | -0.3316 [-0.3337, -0.3297] | -0.1486 [-0.1513, -0.1458] | -0.4900 [-0.4939, -0.4861] | -0.6593 [-0.6632, -0.6555] | -0.7505 [-0.7541, -0.7469] | -0.2975 [-0.3001, -0.2951] |
| embedding_cosine | -0.0402 [-0.0421, -0.0384] | -0.1707 [-0.1739, -0.1675] | -0.2632 [-0.2670, -0.2594] | -0.3332 [-0.3371, -0.3292] | -0.0860 [-0.0894, -0.0827] | -0.1325 [-0.1350, -0.1300] | -0.1678 [-0.1703, -0.1653] | -0.1910 [-0.1933, -0.1885] | -0.0860 [-0.0894, -0.0827] | -0.2711 [-0.2758, -0.2664] | -0.3264 [-0.3313, -0.3213] | -0.3018 [-0.3063, -0.2972] | -0.1587 [-0.1617, -0.1556] |
| multilabel_topic_logistic | -0.0418 [-0.0434, -0.0400] | -0.1715 [-0.1747, -0.1683] | -0.2669 [-0.2709, -0.2632] | -0.3545 [-0.3585, -0.3507] | -0.0892 [-0.0923, -0.0860] | -0.1339 [-0.1362, -0.1315] | -0.1706 [-0.1731, -0.1683] | -0.1994 [-0.2018, -0.1972] | -0.0892 [-0.0923, -0.0860] | -0.2738 [-0.2786, -0.2691] | -0.3379 [-0.3429, -0.3331] | -0.3351 [-0.3395, -0.3306] | -0.1639 [-0.1668, -0.1611] |
| multilabel_embedding_logistic | -0.0276 [-0.0295, -0.0258] | -0.1224 [-0.1258, -0.1191] | -0.1793 [-0.1832, -0.1753] | -0.2051 [-0.2088, -0.2010] | -0.0522 [-0.0557, -0.0487] | -0.0919 [-0.0946, -0.0893] | -0.1140 [-0.1167, -0.1113] | -0.1230 [-0.1255, -0.1205] | -0.0522 [-0.0557, -0.0487] | -0.1833 [-0.1882, -0.1784] | -0.2107 [-0.2156, -0.2059] | -0.1772 [-0.1812, -0.1729] | -0.1009 [-0.1041, -0.0978] |

## Label support

### `exposure_restricted`

Supported labels: 123; unsupported labels: 2.

| Label | Positive-target support |
|---|---:|
| Accidents & Emergency Response | 5006 |
| Books & Literature | 963 |
| Celebrity & Creator Culture | 5520 |
| Comics & Graphic Narratives | 223 |
| Film | 2083 |
| Folklore & Legends | 20 |
| Music | 4503 |
| Television & Streaming | 3047 |
| Theater & Performing Arts | 6141 |
| Trivia & Puzzles | 1113 |
| Visual Arts & Design | 1232 |
| Agriculture & Food Production | 553 |
| Companies & Corporate Strategy | 4267 |
| Economy & Economic Policy | 873 |
| Entrepreneurship & Small Business | 140 |
| Labor & Employment | 2296 |
| Markets & Investing | 988 |
| Personal Finance | 2497 |
| Real Estate | 2457 |
| Constitutional & Civil Law | 61 |
| Courts & Legal Cases | 4210 |
| Crime & Public Safety | 3890 |
| Policing & Corrections | 3721 |
| Careers & Professional Development | 1166 |
| Higher Education | 1441 |
| Schools & Childhood Education | 161 |
| Skills & Lifelong Learning | 4 |
| Workplace Life | 211 |
| Baking & Desserts | 1278 |
| Cooking & Recipes | 2173 |
| Food Culture & Trends | 2225 |
| Restaurants & Dining | 1916 |
| Wine, Beer & Spirits | 631 |
| Civic Life & Public Commemorations | 2357 |
| Civil Rights & Social Justice | 2558 |
| Elections & Political Campaigns | 2726 |
| Government & Public Policy | 2988 |
| Immigration & Citizenship | 2281 |
| Infrastructure & Transportation Policy | 1610 |
| Population & Demographic Change | 404 |
| Public Services & Social Welfare | 683 |
| Aging & Longevity | 1242 |
| Fitness & Exercise | 1576 |
| Medicine & Medical Research | 3433 |
| Mental Health | 176 |
| Nutrition & Healthy Eating | 2031 |
| Public Health | 832 |
| Sexual & Reproductive Health | 951 |
| Sleep Health | 309 |
| Historical Events & Eras | 2347 |
| Architecture & Residential Design | 779 |
| Gardening & Landscaping | 83 |
| Home Appliances & Technology | 96 |
| Home Improvement & Repair | 532 |
| Household Cleaning & Laundry | 239 |
| Interior Design & Decorating | 1844 |
| Beauty & Grooming | 1421 |
| Cannabis Culture & Use | 5 |
| Charity & Volunteering | 786 |
| Crafts & Collecting | 546 |
| Fashion & Personal Style | 3683 |
| Gambling & Lotteries | 195 |
| Genealogy & Family History | 182 |
| Parenting & Family Life | 2215 |
| Personal Growth | 852 |
| Pets & Animal Companions | 2558 |
| Relationships & Dating | 3497 |
| Shopping & Gift Ideas | 4928 |
| Journalism & News Media | 1221 |
| Astrology & Divination | 158 |
| Ethics & Moral Debate | 19 |
| Faith Communities & Practice | 369 |
| Religion in Public Life | 144 |
| Archaeology & Human Origins | 1026 |
| Biology & Genetics | 104 |
| Climate Change | 738 |
| Energy & Sustainability | 611 |
| Physics & Chemistry | 12 |
| Space & Astronomy | 625 |
| Wildlife & Conservation | 2875 |
| American Football | 6684 |
| Baseball & Softball | 863 |
| Basketball | 2666 |
| Combat Sports | 594 |
| General & Multi-Sport Coverage | 278 |
| Golf | 492 |
| Horse Racing & Equestrian Sports | 25 |
| Hunting & Fishing | 9 |
| Ice Hockey | 485 |
| Motorsports | 101 |
| Olympic & Athletics Sports | 237 |
| Outdoor & Endurance Sports | 63 |
| Rugby | 1 |
| Soccer | 305 |
| Sports Business & Administration | 362 |
| Tennis & Racket Sports | 8 |
| Artificial Intelligence & Robotics | 58 |
| Consumer Technology | 121 |
| Cybersecurity & Digital Privacy | 492 |
| Internet & Social Platforms | 219 |
| Software & Computing | 30 |
| Telecommunications | 140 |
| Video Games & Interactive Media | 19 |
| Adventure & Nature Travel | 115 |
| Air & Rail Travel | 794 |
| Cruises | 709 |
| Culture & Heritage Travel | 96 |
| Destinations & Trip Ideas | 1823 |
| Hotels & Accommodation | 691 |
| Travel Planning & Deals | 708 |
| Boating & Watercraft | 648 |
| Cars & Driving | 1676 |
| Electric & Alternative Vehicles | 471 |
| Motorcycles | 866 |
| Public & Shared Mobility | 5 |
| Vehicle Buying & Ownership | 635 |
| Earthquakes & Geological Hazards | 90 |
| Floods, Droughts & Wildfires | 957 |
| Severe Storms | 2917 |
| Weather Forecasts & Seasonal Conditions | 1515 |
| Diplomacy & Geopolitics | 1649 |
| Humanitarian Crises & Aid | 3 |
| War & Armed Conflict | 1159 |

Unsupported labels: `Spirituality & Meaning`, `Global Institutions & Cooperation`

### `full_taxonomy`

Supported labels: 123; unsupported labels: 2.

| Label | Positive-target support |
|---|---:|
| Accidents & Emergency Response | 5006 |
| Books & Literature | 963 |
| Celebrity & Creator Culture | 5520 |
| Comics & Graphic Narratives | 223 |
| Film | 2083 |
| Folklore & Legends | 20 |
| Music | 4503 |
| Television & Streaming | 3047 |
| Theater & Performing Arts | 6141 |
| Trivia & Puzzles | 1113 |
| Visual Arts & Design | 1232 |
| Agriculture & Food Production | 553 |
| Companies & Corporate Strategy | 4267 |
| Economy & Economic Policy | 873 |
| Entrepreneurship & Small Business | 140 |
| Labor & Employment | 2296 |
| Markets & Investing | 988 |
| Personal Finance | 2497 |
| Real Estate | 2457 |
| Constitutional & Civil Law | 61 |
| Courts & Legal Cases | 4210 |
| Crime & Public Safety | 3890 |
| Policing & Corrections | 3721 |
| Careers & Professional Development | 1166 |
| Higher Education | 1441 |
| Schools & Childhood Education | 161 |
| Skills & Lifelong Learning | 4 |
| Workplace Life | 211 |
| Baking & Desserts | 1278 |
| Cooking & Recipes | 2173 |
| Food Culture & Trends | 2225 |
| Restaurants & Dining | 1916 |
| Wine, Beer & Spirits | 631 |
| Civic Life & Public Commemorations | 2357 |
| Civil Rights & Social Justice | 2558 |
| Elections & Political Campaigns | 2726 |
| Government & Public Policy | 2988 |
| Immigration & Citizenship | 2281 |
| Infrastructure & Transportation Policy | 1610 |
| Population & Demographic Change | 404 |
| Public Services & Social Welfare | 683 |
| Aging & Longevity | 1242 |
| Fitness & Exercise | 1576 |
| Medicine & Medical Research | 3433 |
| Mental Health | 176 |
| Nutrition & Healthy Eating | 2031 |
| Public Health | 832 |
| Sexual & Reproductive Health | 951 |
| Sleep Health | 309 |
| Historical Events & Eras | 2347 |
| Architecture & Residential Design | 779 |
| Gardening & Landscaping | 83 |
| Home Appliances & Technology | 96 |
| Home Improvement & Repair | 532 |
| Household Cleaning & Laundry | 239 |
| Interior Design & Decorating | 1844 |
| Beauty & Grooming | 1421 |
| Cannabis Culture & Use | 5 |
| Charity & Volunteering | 786 |
| Crafts & Collecting | 546 |
| Fashion & Personal Style | 3683 |
| Gambling & Lotteries | 195 |
| Genealogy & Family History | 182 |
| Parenting & Family Life | 2215 |
| Personal Growth | 852 |
| Pets & Animal Companions | 2558 |
| Relationships & Dating | 3497 |
| Shopping & Gift Ideas | 4928 |
| Journalism & News Media | 1221 |
| Astrology & Divination | 158 |
| Ethics & Moral Debate | 19 |
| Faith Communities & Practice | 369 |
| Religion in Public Life | 144 |
| Archaeology & Human Origins | 1026 |
| Biology & Genetics | 104 |
| Climate Change | 738 |
| Energy & Sustainability | 611 |
| Physics & Chemistry | 12 |
| Space & Astronomy | 625 |
| Wildlife & Conservation | 2875 |
| American Football | 6684 |
| Baseball & Softball | 863 |
| Basketball | 2666 |
| Combat Sports | 594 |
| General & Multi-Sport Coverage | 278 |
| Golf | 492 |
| Horse Racing & Equestrian Sports | 25 |
| Hunting & Fishing | 9 |
| Ice Hockey | 485 |
| Motorsports | 101 |
| Olympic & Athletics Sports | 237 |
| Outdoor & Endurance Sports | 63 |
| Rugby | 1 |
| Soccer | 305 |
| Sports Business & Administration | 362 |
| Tennis & Racket Sports | 8 |
| Artificial Intelligence & Robotics | 58 |
| Consumer Technology | 121 |
| Cybersecurity & Digital Privacy | 492 |
| Internet & Social Platforms | 219 |
| Software & Computing | 30 |
| Telecommunications | 140 |
| Video Games & Interactive Media | 19 |
| Adventure & Nature Travel | 115 |
| Air & Rail Travel | 794 |
| Cruises | 709 |
| Culture & Heritage Travel | 96 |
| Destinations & Trip Ideas | 1823 |
| Hotels & Accommodation | 691 |
| Travel Planning & Deals | 708 |
| Boating & Watercraft | 648 |
| Cars & Driving | 1676 |
| Electric & Alternative Vehicles | 471 |
| Motorcycles | 866 |
| Public & Shared Mobility | 5 |
| Vehicle Buying & Ownership | 635 |
| Earthquakes & Geological Hazards | 90 |
| Floods, Droughts & Wildfires | 957 |
| Severe Storms | 2917 |
| Weather Forecasts & Seasonal Conditions | 1515 |
| Diplomacy & Geopolitics | 1649 |
| Humanitarian Crises & Aid | 3 |
| War & Armed Conflict | 1159 |

Unsupported labels: `Spirituality & Meaning`, `Global Institutions & Cooperation`

## Data and run paths

- Benchmark directory: `data/processed/future_novel_interest_users_v1`
- Artifact directory: `artifacts/future-novel-interest/baselines-v1`
- `global_popularity/exposure_restricted` metrics: `artifacts/future-novel-interest/baselines-v1/global_popularity/exposure_restricted/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/global_popularity/exposure_restricted/details.parquet`
- `global_popularity/full_taxonomy` metrics: `artifacts/future-novel-interest/baselines-v1/global_popularity/full_taxonomy/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/global_popularity/full_taxonomy/details.parquet`
- `conditional_frequency/exposure_restricted` metrics: `artifacts/future-novel-interest/baselines-v1/conditional_frequency/exposure_restricted/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/conditional_frequency/exposure_restricted/details.parquet`
- `conditional_frequency/full_taxonomy` metrics: `artifacts/future-novel-interest/baselines-v1/conditional_frequency/full_taxonomy/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/conditional_frequency/full_taxonomy/details.parquet`
- `conditional_lift/exposure_restricted` metrics: `artifacts/future-novel-interest/baselines-v1/conditional_lift/exposure_restricted/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/conditional_lift/exposure_restricted/details.parquet`
- `conditional_lift/full_taxonomy` metrics: `artifacts/future-novel-interest/baselines-v1/conditional_lift/full_taxonomy/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/conditional_lift/full_taxonomy/details.parquet`
- `embedding_cosine/exposure_restricted` metrics: `artifacts/future-novel-interest/baselines-v1/embedding_cosine/exposure_restricted/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/embedding_cosine/exposure_restricted/details.parquet`
- `embedding_cosine/full_taxonomy` metrics: `artifacts/future-novel-interest/baselines-v1/embedding_cosine/full_taxonomy/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/embedding_cosine/full_taxonomy/details.parquet`
- `multilabel_topic_logistic/exposure_restricted` metrics: `artifacts/future-novel-interest/baselines-v1/multilabel_topic_logistic/exposure_restricted/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/multilabel_topic_logistic/exposure_restricted/details.parquet`
- `multilabel_topic_logistic/full_taxonomy` metrics: `artifacts/future-novel-interest/baselines-v1/multilabel_topic_logistic/full_taxonomy/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/multilabel_topic_logistic/full_taxonomy/details.parquet`
- `multilabel_embedding_logistic/exposure_restricted` metrics: `artifacts/future-novel-interest/baselines-v1/multilabel_embedding_logistic/exposure_restricted/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/multilabel_embedding_logistic/exposure_restricted/details.parquet`
- `multilabel_embedding_logistic/full_taxonomy` metrics: `artifacts/future-novel-interest/baselines-v1/multilabel_embedding_logistic/full_taxonomy/metrics.json`; details: `artifacts/future-novel-interest/baselines-v1/multilabel_embedding_logistic/full_taxonomy/details.parquet`

## Caveats

- Never-exposed unknown: labels that were never exposed or otherwise absent from the evaluated candidate set are not evidence of recoverability.
- Future positives are observed benchmark events, not randomized outcomes; this report makes no causal claim.
- This report uses deterministic ranking metrics and paired user bootstrap intervals; it uses no LLM judge.
