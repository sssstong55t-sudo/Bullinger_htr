# Official-Only Results

Actual CPU inference completed on 39 eligible official letters, 357 paragraphs and 327 documented body-place spans. Personal review labels were not read.

| Branch | Matched / official | Agreement P | Agreement R | Agreement F1 |
|---|---|---|---|---|
|single_04|168/327|60.43%|51.38%|55.54%|
|competition_04|161/327|62.16%|49.24%|54.95%|
|competition_08|109/327|80.74%|33.33%|47.19%|
|competition_04_official_aliases|230/327|68.05%|70.34%|69.17%|
|competition_08_official_aliases|200/327|82.99%|61.16%|70.42%|

The primary report is `competition_08_official_aliases`. Other branches are controls, not a retrospective best-score selection. The annotation-agreement interpretation and unknown completeness limit apply to every branch. No new HTR test or model training was performed.

The 127 official misses consist of 30 exact matches found only by the lower-threshold control, one overlapping-boundary case, and 96 cases without an exact or overlapping primary candidate. See `results/official_misses.csv`. These errors should not be used to improve the system and then treated as independent confirmation on the same letters.