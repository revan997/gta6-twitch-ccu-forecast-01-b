[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/revan997/gta6-twitch-ccu-forecast-01-b/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/revan997/gta6-twitch-ccu-forecast-01-b/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers, hour by hour — empirical-quantile forecast

Entry for [make.fun competition `gta6-twitch-ccu-forecast-01`](https://make.fun/c/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, 2026-11-19 (UTC), after the first hour's print,
as five quantiles per hour (5%, 25%, 50%, 75%, 95%).

## Program

`main.py` (Python 3.13, standard library only, offline, deterministic) reads
every case file in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`/output/<id>.out` for each.

## Model — hour-by-hour empirical quantiles

For each hour h = 2..24, the model forecasts the log-ratio
`ln(viewers_h / viewers_hour1)` from the case's own comparable past
launches, but not as raw empirical quantiles (38 curves per hour is noisy,
and the ratio depends strongly on how big hour 1 already was — across the
comparables, corr(ln hour-1, ln ratio at hour 19) ≈ −0.93):

- per hour, a **weighted local-linear regression** of the log-ratio on
  `ln(hour-1 viewers)`, weighting each comparable by a Gaussian kernel in
  that predictor (bandwidth 1.5 log units, hard trim beyond 3 bandwidths);
- **weighted empirical quantiles of the regression residuals** at the five
  scored levels, shrunk halfway toward a normal with the same residual
  scale and narrowed by a factor 0.85;
- the median curve is lightly **smoothed across neighbouring hours**
  (0.25/0.5/0.25), the same shift applied to all five quantiles of an hour;
- comparables are **tier-filtered** by the tag in their unlock note: weak
  analogues and early-access-contaminated launches are dropped (their
  hour-1 print does not mean the same thing); untagged comparables are kept,
  and if the filter would leave too few curves, all are used.

## Validation (local scorer: mean pinball loss on the log scale)

Practice answers were taken from SullyGnome (the kit's documented source);
the scorer was verified by reproducing the house baseline's stated
practice loss of 0.387 exactly.

| Check | Result |
|---|---|
| Practice loss (4 cases, true answers) | **0.157** (p00 0.175, p01 0.156, p02 0.116, p03 0.182) vs house baseline 0.387 |
| 90% coverage on practice | 86/92 hours = 0.935 |
| Leave-one-out, GTA-like curves (n=20) | 0.221 |
| Leave-one-out, all curves (n=41) | 0.385 — dominated by simultaneous-unlock / tiny-category launches GTA VI will not resemble |
| Format | 23 rows × 5 quantiles, monotone, within bounds, ≤ 65,536 bytes |
| Determinism | two runs byte-identical |
| Runtime | ~0.2 s per case (limit 8 s) |

Honest note: 0.157 on practice is well below the house baseline but above
the live board's top-20 bar of ~0.135, so on practice-like cases this would
not reach the paid zone. Its best case is a very large hour-1 print (the
biggest practice case, p02, scores 0.116): if GTA VI's hour 1 is huge, the
model's neighbours are the flat-curve mega-launches and the final-case loss
could plausibly land near or below the bar. See `REPORT.md` in the workdir
for the full validation report (not part of this repo).

## License

MIT — see [LICENSE](LICENSE).
