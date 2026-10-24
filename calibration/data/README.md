# Task-duration data for the tail-index fit

Both files are small extracts of public METR releases; `tail_fit.py` reads them.

- `metr_th11_baselined_tasks.csv` — the 96 HCAST and RE-Bench tasks in METR's
  Time Horizon 1.1 suite whose length comes from human baselines (SWAA tasks and
  author-estimated lengths excluded). `human_minutes` is METR's task length, the
  geometric mean of successful baseline run times. Source:
  `reports/time-horizon-1-1/data/raw/runs.jsonl` in
  https://github.com/METR/eval-analysis-public.
- `metr_early2025_rct_issues.csv` — the 246 real open-source issues in METR's
  early-2025 developer-productivity RCT, with the developer's forecast without AI
  and the actual initial implementation time (minutes). Source: `data_complete.csv`
  in https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs.

