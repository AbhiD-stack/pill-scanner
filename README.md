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

Scaffold only — no trained model yet. Data acquisition and training must run
somewhere with real internet access (a Kaggle notebook); the authoring
environment for this code had all external data-source hosts blocked by
network policy (RxNav, openFDA, DailyMed, even clincalc.com and wikipedia.org
returned 403 from the egress proxy), so nothing here has been executed against
live data yet — only unit-tested against synthetic inputs.

## Layout

| Path | What |
|---|---|
| `data/seed_lists/` | Top-500 RX + top-500 OTC generic-name seed lists (compiled from training-time knowledge — cross-check against a live ClinCalc Top 300 pull before treating ranks as current) |
| `data/DATA_SOURCES.md` | Which datasets to pull, what each contributes, and known coverage gaps |
| `scripts/build_drug_metadata.py` | Seed name -> RxCUI -> NDCs -> imprint/color/shape/score marks, via RxNav |
| `backend/app/filter_cascade.py` | The cascading shape/color/imprint/embedding fusion scorer |
| `eval/evaluate.py` | The accuracy gate: multi-axis metrics + calibration + worst-class report |
| `notebooks/` | Kaggle-run data acquisition + training + eval pipeline |

## Next steps

1. Run `scripts/build_drug_metadata.py` against both seed lists on Kaggle
   (or any machine with internet) to resolve real NDCs/metadata.
2. Pull images per `data/DATA_SOURCES.md` (C3PI reference + consumer tiers,
   CURE, Kaggle mirrors), prioritizing the two seed lists.
3. Train the embedding model with domain-randomization augmentation
   (background/lighting/perspective/blur — closing the studio-to-phone gap
   is the single highest-leverage fix identified from the pill-id numbers).
4. Run `eval/evaluate.py` — do not export to the app unless the
   consumer-domain top-5 bar is met.
