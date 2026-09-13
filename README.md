# Pill Scanner (ground-up rebuild)

A from-scratch rebuild of pill identification, replacing the ePillID-only,
single ~2-image-per-class approach in the `pill-id` repo. That repo's v1/v2
apps are untouched by this work — this is a separate project.

## Why a rebuild

The previous model hit 95%/89% top-5/top-10 on ePillID's own reference-image
test split, but performed poorly on real phone photos in the deployed app.
Root cause: every number came from the same studio-photo distribution the
model trained on (see `dinov2_projection_head/*_test_metrics.json` in
`pill-id`) — there was no real-world ("consumer-quality") test set, no
imprint/shape/color signal used at inference time (only for display), and
only ~2 reference images per class (ePillID's front+back pair).

## What's different here

1. **Cascading attribute filter** (`backend/app/filter_cascade.py`) —
   shape/color narrow the candidate field, imprint text (OCR'd from the
   photo) is near-decisive within that shortlist, embedding similarity is
   the fallback/tiebreaker. Soft-scored, not hard elimination, since
   automatic shape/color detection from a phone photo is imperfect.
2. **Tiered class coverage**: top ~500 RX (`data/seed_lists/top_500_seed.csv`)
   + top ~500 OTC (`data/seed_lists/top_otc_seed.csv`) get priority for data
   density (aiming for 4-5 images/side where the source data allows it), with
   a long tail of a few thousand more classes best-effort beyond that.
3. **Mandatory accuracy gate before deploy** (`eval/evaluate.py`) — reports
   top-1/5/10/20/50 split by **reference vs. consumer-quality domain**,
   by images-per-class depth, plus a confidence-calibration curve and a
   worst-performing-classes list. A model only gets exported to the app if
   its consumer-domain (real-world) numbers clear a bar you set — see
   `--min-top5-consumer`.

## Status

Scaffold + two Kaggle notebooks, both validated against synthetic data (this
authoring environment has RxNav/openFDA/DailyMed/Kaggle itself all blocked
by network policy, so nothing here has been executed against real, live
data yet — every piece is unit- or integration-tested with synthetic
fixtures standing in for the real thing). A live run on Kaggle has
confirmed real structural facts along the way (DailyMed's zip-of-zips
layout, its actual current part count, Kaggle's dataset mount-path
conventions) and the code was corrected against those, not assumptions.

**US-only policy**: every image source is US-sourced (ePillID, NIH C3PI,
DailyMed). Several real, per-class-labeled Kaggle datasets (Hungarian,
Philippines, Vietnamese-derived) were tried for extra volume/diversity and
then removed once this was made explicit — being real and labeled isn't
enough if it isn't a US product.

## Layout

| Path | What |
|---|---|
| `data/seed_lists/` | Top-500 RX + top-500 OTC generic-name seed lists (compiled from training-time knowledge — cross-check against a live ClinCalc Top 300 pull before treating ranks as current) |
| `data/DATA_SOURCES.md` | Which datasets to pull, what each contributes, and known coverage gaps |
| `scripts/build_drug_metadata.py` | Seed name -> RxCUI -> NDCs -> imprint/color/shape/score marks, via RxNav |
| `scripts/parse_dailymed_spl.py` | DailyMed SPL XML -> per-NDC imprint/color/shape/score + image filenames (also inlined into the acquisition notebook so it has no repo-attachment dependency) |
| `backend/app/filter_cascade.py` | The cascading shape/color/imprint/embedding fusion scorer |
| `eval/evaluate.py` | The accuracy gate: multi-axis metrics (including RX/OTC and reference/consumer splits) + calibration + worst-class report |
| `notebooks/dailymed_acquisition.ipynb` | **Run this first**, CPU-only, via Save & Run All (commit) — downloads + parses DailyMed's bulk US RX+OTC SPL data unattended, writes a manifest + images as this notebook's Output |
| `notebooks/pill_scanner_training.ipynb` | Attaches `dailymed_acquisition.ipynb`'s output (Add Input -> Notebook) alongside the other Kaggle-hosted datasets, builds the unified manifest, trains, and runs the accuracy gate |

## Next steps

1. Run `notebooks/dailymed_acquisition.ipynb` on Kaggle (Accelerator: None,
   Save & Run All) — its default scope is 2 OTC parts + 1 RX part, widen
   `SPL_PARTS_TO_FETCH` once that scope is confirmed working.
2. In `notebooks/pill_scanner_training.ipynb`, attach that notebook's output
   plus `tommyngx/epillid-data-v1` and (optionally) C3PI, then run Sections
   1-4 and check the printed image/label counts before spending GPU time.
3. Train with the domain-randomization augmentation already wired in
   (Section 5) — closing the studio-to-phone gap is the single
   highest-leverage fix identified from the pill-id numbers.
4. Run the accuracy gate (Section 10 / `eval/evaluate.py`) — do not export
   to the app unless the consumer-domain top-5 bar is met, and check the
   by-category (RX/OTC) split specifically, not just the overall number.
