# Data sources

This repo's data acquisition runs in the Kaggle notebook (`notebooks/`), not in
whatever environment is editing this code — image/metadata downloads need real
internet access.

**US-only policy**: every image source in this pipeline must be US-sourced.
Three Kaggle datasets tried early on (OGYEIv2/Hungarian, a Philippines
drugstore photo set, and an unconfirmed-origin drug/vitamin set) were removed
once this was made explicit, even though they were real, per-class-labeled
photos — being real and labeled isn't enough if the product/imprint/label
conventions aren't US ones. Two more (a Vietnamese VAIPE-based pill-detection
set and a couple of unconfirmed-origin ones) were dropped too even though
they were only ever used as unlabeled bounding-box/detector-training data
with no drug-identity claim attached — a stricter "no non-US imagery
anywhere in the pipeline" bar, not just "no non-US drug labels." Current
image sources are exactly three: ePillID, NIH C3PI, and DailyMed — all
NIH/FDA-derived, all US.

## Images

| Source | What it gives us | Images/side/class | Priority |
|---|---|---|---|
| ePillID (already have, in `pill-id`) | 4,902 RX classes, studio reference photos | 1 | baseline, carry forward |
| NIH C3PI — reference tier | Broader RX imprint coverage than ePillID alone | 1-2 | fill RX class gaps |
| NIH C3PI — consumer-quality tier | Phone/scanner photos, varied lighting/background | 3-10+ (uneven across NDCs) | closes the reference-vs-wild domain gap; prioritize for the top-500 RX tier first |
| DailyMed bulk SPL (RX + OTC) | Manufacturer-submitted photos + FDA-structured imprint/color/shape/score, straight from the drug label | 1 per NDC typically | primary OTC source — see its own section below |

CURE was never actually wired in (no confirmed Kaggle mirror was found), and
a handful of other Kaggle-hosted datasets tried for extra volume were
removed once the US-only policy was made explicit (see above) — don't
re-add them without confirming US origin first.

Expect uneven depth: the top-500 RX + top-500 OTC tier is where we can
realistically hit 4-5 images/side; the long-tail thousands will often land at
1-2. The acquisition notebook records actual per-class image counts rather
than assuming a uniform number — `eval/` reports accuracy split by this count
so a thin class's low score isn't confused with a model bug.

## DailyMed bulk SPL — the primary US RX+OTC image+metadata source

Found after the original 5 Kaggle-hosted datasets turned out to have zero
OTC-labeled images (ePillID and the VAIPE-based detection set are RX-only;
the rest are unlabeled-by-drug). DailyMed's bulk Structured Product Labeling
(SPL) releases, confirmed live at
https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm,
split into `dm_spl_release_human_rx_part{1,2}.zip` and
`dm_spl_release_human_otc_part{1,2,3}.zip` — each SPL document is the actual
FDA-submitted drug label XML, which for oral solid dosage forms includes
`SPLCOLOR`/`SPLIMPRINT`/`SPLSHAPE`/`SPLSCORE`/`SPLSIZE` characteristics and a
`SPLIMAGE` reference to a manufacturer-submitted product photo bundled in the
same package. This is the only source in the whole pipeline that is US-only,
covers OTC as well as RX, and has both real images and FDA-structured
physical-characteristic data (no OCR guessing needed for these records).

`scripts/parse_dailymed_spl.py` is a modernized (Python 3) port of the
parsing logic in HHS's archived `pillbox-data-process` repo (the actual
source of this imprint/color/shape/score extraction approach — a newer-
looking alternative, `pharmaDB/dailymed_data_processor`, was checked and
turned out to only handle label text/history, not images or physical
characteristics, so it wasn't a useful base). It's unit-tested against a
synthetic SPL XML fixture, not yet run against a real bulk zip.

Known limitation: the parser only handles the common single-part
`<manufacturedProduct>` case, not the nested `<part>`/`<partProduct>`
structure some multi-part kits use — those get silently skipped rather than
mis-parsed. Revisit if the yield looks low relative to a zip's XML count.

## Metadata (imprint / color / shape / score marks)

RxNav's `ndcproperties` endpoint (already used in `pill-id/backend/scripts/build_ndc_names.py`)
returns `IMPRINT_CODE`, `COLORTEXT`, and also `SPLSHAPE` / `SPLSCORE`, which the
original script didn't capture. `scripts/build_drug_metadata.py` here pulls all
four, so the cascading filter (see `backend/app/filter_cascade.py`) has shape
and score-mark signal, not just imprint + color.

OTC has no equivalent federal imprint mandate, so OTC metadata is thinner by
nature — DailyMed/openFDA structured product listings cover many OTC NDCs but
skew toward packaging photos rather than loose-pill photos with legible
imprints. Manual verification will matter more for the OTC seed list than RX.

## Seed lists

- `seed_lists/top_500_seed.csv` — RX tier-1/2 core, compiled from stable
  multi-year prescribing patterns. **Needs a final cross-check against the
  live ClinCalc Top 300 (clincalc.com/DrugStats/Top300Drugs.aspx) for exact
  current ranking** — this list was compiled without live internet access in
  the authoring environment, so treat rank/inclusion as "very likely right,"
  not verified against the current-year source.
- `seed_lists/top_otc_seed.csv` — common OTC actives + major brand names.
  Store-brand variants (CVS/Walgreens/Kirkland-equivalent) multiply the
  effective NDC/class count per active ingredient and should be pulled in by
  the acquisition notebook once each active ingredient's canonical RxCUI is
  resolved.
