
# ============================================================================
# 0. Setup
# ============================================================================

!pip install -q transformers timm albumentations peft pytesseract
!apt-get -qq install -y tesseract-ocr > /dev/null

import os, json, glob, re, random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

WORK_DIR = Path("/kaggle/working")
MANIFEST_PATH = WORK_DIR / "image_manifest.csv"
METADATA_DIR = WORK_DIR / "metadata"
EXPORT_DIR = WORK_DIR / "export"
for d in (METADATA_DIR, EXPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)



# ============================================================================
# 1. Attach datasets
# ============================================================================

# 1.1 Kaggle-hosted datasets. Confirmed-US: epillid (NIH C3PI-derived). The
# rest are non-US or unconfirmed origin, kept for volume/diversity per an
# explicit "primarily US, maximize images per class" call — DailyMed (its
# own acquisition notebook) + epillid + C3PI are the confirmed-US backbone;
# these add bulk on top of that, not instead of it.
KAGGLE_DATASETS = {
    "epillid": ("tommyngx", "epillid-data-v1"),              # US (NIH C3PI-derived)
    "pills_detect_a": ("alexanderyyy", "pills-detection-dataset"),   # Vietnam (VAIPE-based)
    "pills_detect_b": ("perfect9015", "pillsdetectiondataset"),      # unconfirmed origin
    "trumed_1k": ("trumedicines", "1k-pharmaceutical-pill-image-dataset"),  # unconfirmed origin
    "trumed_tablets": ("trumedicines", "pharmaceutical-tablets-dataset"),   # unconfirmed origin
    "ogyeiv2": ("richardradli", "ogyeiv2"),                    # Hungary
    "phvitamins_v2": ("vencerlanz09", "pharmaceutical-drugs-and-vitamins-dataset-v2"),  # Philippines
    "drugs_vitamins_cls": ("utkarshsaxenadn", "drugs-and-vitamins-classification"),     # unconfirmed origin
}

def find_dataset_root(owner, slug):
    candidates = [
        f"/kaggle/input/datasets/{owner}/{slug}",
        f"/kaggle/input/{slug}",
    ]
    candidates += glob.glob(f"/kaggle/input/datasets/{owner}/{slug}*")
    candidates += glob.glob(f"/kaggle/input/{slug}*")
    for c in candidates:
        if Path(c).exists():
            return c
    return None

dataset_roots = {}
for key, (owner, slug) in KAGGLE_DATASETS.items():
    root = find_dataset_root(owner, slug)
    if root:
        dataset_roots[key] = root
        print(f"[{key}] found at {root}")
    else:
        print(f"[{key}] NOT attached — add it via Add Input if you want this source included")

for key, root in dataset_roots.items():
    print(f"\n--- {key} ({root}) — first 20 entries, up to 3 levels deep ---")
    n = 0
    for p in Path(root).rglob("*"):
        if n >= 20:
            break
        print(" ", p.relative_to(root))
        n += 1


# 1.2 NIH C3PI / RxIMAGE — reads notebooks/c3pi_acquisition.ipynb's committed
# output the same way Section 1.3 reads DailyMed's, instead of the earlier
# live-discovery-and-download version of this cell (that version predates
# c3pi_acquisition.ipynb actually being run — its real, confirmed URLs are
# https://data.lhncbc.nlm.nih.gov/public/Pills/rximage.zip (reference,
# confirmed: 48,312 images / 4,236 NDCs) and the consumer-quality index at
# https://data.lhncbc.nlm.nih.gov/public/Pills/index.html (per-disc folders,
# crawler deferred -- reference set only, for now).
#
# Add Input > Notebook > c3pi_acquisition (after committing it via Save
# Version > Save & Run All) to attach its c3pi_manifest.csv here.
import glob as _glob_c3pi
import zipfile as _zipfile_c3pi

def _find_or_unzip_manifest(filename, unzip_dir_name):
    """Kaggle bundles a notebook's output into a single `_output_.zip`
    instead of exposing files individually once there are too many output
    files (confirmed: this is exactly what happened to the DailyMed
    acquisition notebook's ~133k images — its manifest csv wasn't found
    loose under /kaggle/input at all, only `_output_.zip` was). Try the
    direct glob first, then fall back to finding and extracting any
    `_output_.zip` under /kaggle/input and searching again."""
    hits = _glob_c3pi.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if hits:
        return hits[0]
    zips = _glob_c3pi.glob("/kaggle/input/**/_output_.zip", recursive=True)
    for zip_path in zips:
        extract_dir = WORK_DIR / unzip_dir_name / Path(zip_path).parent.name
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            print(f"  extracting {zip_path} -> {extract_dir} (Kaggle zipped this "
                  f"notebook's output because it had too many files)")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with _zipfile_c3pi.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        hits = list(extract_dir.rglob(filename))
        if hits:
            return str(hits[0])
    return None

C3PI_MANIFEST_PATH = _find_or_unzip_manifest("c3pi_manifest.csv", "c3pi_unzipped")
if C3PI_MANIFEST_PATH:
    print(f"Found C3PI acquisition output at: {C3PI_MANIFEST_PATH}")
else:
    print("C3PI acquisition output not found. Run notebooks/c3pi_acquisition.ipynb "
          "separately (Save & Run All / commit), then Add Input > Notebook > that "
          "notebook here, and re-run this cell. Proceeding without it for now -- "
          "this is the reference-tier RX source, ~48k images / 4,236 NDCs.")



# ============================================================================
# 1.3 DailyMed bulk SPL (US RX+OTC) — from a separate acquisition notebook
# ============================================================================

import glob as _glob
import zipfile as _zipfile_dm

def _find_or_unzip_dailymed_manifest(filename, unzip_dir_name):
    """Same Kaggle output-bundling behavior as C3PI's helper in Section 1.2 —
    duplicated here (not shared) so each acquisition section stays
    self-contained if pasted/run independently. Confirmed live: DailyMed's
    ~133k-image output got bundled into `_output_.zip` rather than exposed
    as individual files, so a direct glob for dailymed_manifest.csv alone
    misses it entirely."""
    hits = _glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if hits:
        return hits[0]
    zips = _glob.glob("/kaggle/input/**/_output_.zip", recursive=True)
    for zip_path in zips:
        extract_dir = WORK_DIR / unzip_dir_name / Path(zip_path).parent.name
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            print(f"  extracting {zip_path} -> {extract_dir} (Kaggle zipped this "
                  f"notebook's output because it had too many files)")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with _zipfile_dm.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        hits = list(extract_dir.rglob(filename))
        if hits:
            return str(hits[0])
    return None

DAILYMED_MANIFEST_PATH = _find_or_unzip_dailymed_manifest("dailymed_manifest.csv", "dailymed_unzipped")
if DAILYMED_MANIFEST_PATH:
    print(f"Found DailyMed acquisition output at: {DAILYMED_MANIFEST_PATH}")
    DAILYMED_METADATA_PATH = Path(DAILYMED_MANIFEST_PATH).parent / "dailymed_metadata.json"
    print(f"  metadata present: {DAILYMED_METADATA_PATH.exists()}")
else:
    print("DailyMed acquisition output not found. Run notebooks/dailymed_acquisition.ipynb "
          "separately (Save & Run All / commit), then Add Input > Notebook > that notebook "
          "here, and re-run this cell.")



# ============================================================================
# 2. Drug metadata (imprint / color / shape / score marks)
# ============================================================================

import sys
import time
import urllib.parse, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"

class RateLimited(Exception):
    pass

def normalize_ndc9(ndc):
    """5-4 digit labeler-product prefix (ignoring package size), same
    zero-padding convention pill-id's build_ndc_names.py uses. Defined here
    (rather than after the resolution loop, where it's also used to build
    PRIORITY_NDC9_SET) so resolve_one can dedupe by it too -- package-size
    variants of the same drug (e.g. "0069-2587-01" vs "0069-2587-30") share
    an NDC9 prefix, so there's no point spending an HTTP call on more than
    one of them."""
    parts = str(ndc).split("-")
    if len(parts) < 2:
        return str(ndc)
    return f"{parts[0].zfill(5)}-{parts[1].zfill(4)}"

# Diagnostic counters — the previous version of this cell silently swallowed
# every exception (network error, HTTP error, bad JSON, unexpected response
# shape all looked identical: "0 NDC entries, no error shown"). A real run
# hit that exact wall: 190/199 names "resolved" but 0 NDC entries anywhere,
# with zero visibility into why. These counters and the printed samples below
# make the actual failure mode show up in the log instead of being guessed at.
_diag = {"http_error": 0, "url_error": 0, "json_error": 0, "other_error": 0, "ok": 0}
_diag_lock = __import__("threading").Lock()
_first_errors = []  # first few (url, exception) pairs, for a human to read

def _get(url, _label=""):
    last_exc = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pill-scanner/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.load(resp)
            with _diag_lock:
                _diag["ok"] += 1
            return data
        except urllib.error.HTTPError as e:
            last_exc = e
            with _diag_lock:
                _diag["http_error"] += 1
        except urllib.error.URLError as e:
            last_exc = e
            with _diag_lock:
                _diag["url_error"] += 1
        except Exception as e:
            last_exc = e
            with _diag_lock:
                _diag["json_error" if "JSON" in type(e).__name__ else "other_error"] += 1
        time.sleep(2 ** attempt)
    with _diag_lock:
        if len(_first_errors) < 5:
            _first_errors.append((url, f"{type(last_exc).__name__}: {last_exc}"))
    raise RateLimited(f"{url} -> {type(last_exc).__name__}: {last_exc}")

def rxcuis_for_name(name):
    data = _get(f"{RXNAV_BASE}/rxcui.json?name={urllib.parse.quote(name)}&search=2", name)
    return data.get("idGroup", {}).get("rxnormId", []) or []

def ndcs_for_rxcui(rxcui):
    data = _get(f"{RXNAV_BASE}/rxcui/{rxcui}/ndcs.json", rxcui)
    return data.get("ndcGroup", {}).get("ndcList", {}).get("ndc", []) or []

# Root cause of the "177 rxcuis resolved / 0 NDCs" bug: rxcui.json?search=2
# on a bare generic name (e.g. "atorvastatin") resolves to RxNorm's
# ingredient-level concept (TTY=IN), and /ndcs.json only returns NDCs for
# concrete dispensable-drug concepts (SCD/SBD/GPCK/BPCK) -- an ingredient
# concept has no NDCs directly attached, by design of the RxNorm concept
# graph, not a bug in this code. Confirmed by the diagnostics: 377/377 HTTP
# calls succeeded (ok=377, 0 errors of any kind), so this was never a
# network/rate-limit issue -- it was querying the wrong concept level.
CONCRETE_DRUG_TTYS = "SCD+SBD+GPCK+BPCK"
MAX_CONCRETE_RXCUIS_PER_NAME = 25   # cap fan-out per ingredient name
MAX_NDCS_PER_CONCRETE_RXCUI = 10    # cap package-size variants per concept

def related_concrete_rxcuis(rxcui):
    data = _get(f"{RXNAV_BASE}/rxcui/{rxcui}/related.json?tty={CONCRETE_DRUG_TTYS}", rxcui)
    groups = data.get("relatedGroup", {}).get("conceptGroup", []) or []
    out = []
    for g in groups:
        for cp in g.get("conceptProperties", []) or []:
            if cp.get("rxcui"):
                out.append(cp["rxcui"])
    return out

def properties_for_ndc(ndc):
    data = _get(f"{RXNAV_BASE}/ndcproperties.json?id={ndc}&ndcstatus=ALL", ndc)
    pl = data.get("ndcPropertyList", {}).get("ndcProperty", [])
    if not pl:
        return None
    p = pl[0]
    props = {x["propName"]: x["propValue"] for x in p.get("propertyConceptList", {}).get("propertyConcept", [])}
    return {
        "ndc": ndc, "rxcui": p.get("rxcui"),
        "imprint": props.get("IMPRINT_CODE"), "color": props.get("COLORTEXT"),
        "shape": props.get("SPLSHAPE"), "score_marks": props.get("SPLSCORE"),
        "status": props.get("NDC_STATUS"),
    }

_diag_names = {"rxcui_hits": 0, "rxcui_misses": 0, "ndc_hits": 0, "ndc_misses": 0}

def resolve_one(name):
    entries = []
    try:
        rxcuis = rxcuis_for_name(name)
        with _diag_lock:
            _diag_names["rxcui_hits" if rxcuis else "rxcui_misses"] += 1
        for rxcui in rxcuis:
            # Expand each resolved rxcui to its concrete dispensable-drug
            # relatives first -- an ingredient-level rxcui itself will
            # always return an empty NDC list, so querying it directly (the
            # old behavior) silently produced zero NDCs for every name.
            try:
                concrete_rxcuis = related_concrete_rxcuis(rxcui)
            except RateLimited:
                concrete_rxcuis = []
            # Fall back to the original rxcui too, in case rxcui.json already
            # resolved straight to a concrete concept for some names (brand
            # names in particular sometimes do) -- cheap and harmless if
            # ndcs_for_rxcui just returns [] for it.
            # Bounded, not exhaustive: an ingredient can expand to dozens of
            # strength/brand/generic concepts, each with many package-size
            # NDCs -- we only need enough real NDC9 prefixes per name to
            # populate the priority tier, not every package size ever sold.
            targets = list(dict.fromkeys(concrete_rxcuis + [rxcui]))[:MAX_CONCRETE_RXCUIS_PER_NAME]
            seen_ndc9 = set()
            for target_rxcui in targets:
                try:
                    ndcs = ndcs_for_rxcui(target_rxcui)
                except RateLimited:
                    continue
                with _diag_lock:
                    _diag_names["ndc_hits" if ndcs else "ndc_misses"] += 1
                fetched_for_this_rxcui = 0
                for ndc in ndcs:
                    prefix = normalize_ndc9(ndc)
                    if prefix in seen_ndc9:
                        continue  # same drug/strength, just a different package size
                    if fetched_for_this_rxcui >= MAX_NDCS_PER_CONCRETE_RXCUI:
                        break
                    seen_ndc9.add(prefix)
                    fetched_for_this_rxcui += 1
                    try:
                        props = properties_for_ndc(ndc)
                    except RateLimited:
                        continue
                    if props:
                        entries.append(props)
    except RateLimited:
        pass
    return name, entries

# Inlined from data/seed_lists/{top_500_seed,top_otc_seed}.csv (kept in
# sync manually) — GitHub-repo attachment has been unreliable in practice
# this session, and this list is now load-bearing (it defines the
# "priority tier" the accuracy gate reports separately, per the actual
# goal: 80/90 on the top-500 RX+OTC specifically, not the long tail).
seed_names = ['atorvastatin', 'levothyroxine', 'metformin', 'lisinopril', 'amlodipine', 'metoprolol tartrate', 'metoprolol succinate ER', 'albuterol', 'omeprazole', 'losartan', 'gabapentin', 'hydrochlorothiazide', 'sertraline', 'simvastatin', 'escitalopram', 'rosuvastatin', 'bupropion', 'furosemide', 'pantoprazole', 'trazodone', 'fluticasone propionate nasal', 'duloxetine', 'tamsulosin', 'prednisone', 'amoxicillin', 'alprazolam', 'citalopram', 'meloxicam', 'clopidogrel', 'azithromycin', 'warfarin', 'cyclobenzaprine', 'carvedilol', 'venlafaxine ER', 'insulin glargine', 'montelukast', 'pravastatin', 'hydrocodone/acetaminophen', 'tramadol', 'potassium chloride ER', 'lorazepam', 'clonazepam', 'cholecalciferol (vitamin D3) high-dose', 'spironolactone', 'allopurinol', 'amitriptyline', 'buspirone', 'famotidine', 'zolpidem', 'atenolol', 'diazepam', 'glipizide', 'ezetimibe', 'doxycycline hyclate', 'cephalexin', 'ciprofloxacin', 'levofloxacin', 'clindamycin', 'fluoxetine', 'paroxetine', 'quetiapine', 'risperidone', 'aripiprazole', 'olanzapine', 'lamotrigine', 'levetiracetam', 'topiramate', 'divalproex sodium ER', 'phenytoin', 'carbamazepine', 'oxycodone', 'oxycodone/acetaminophen', 'morphine sulfate ER', 'methylphenidate', 'amphetamine/dextroamphetamine', 'lisdexamfetamine', 'atomoxetine', 'sildenafil', 'tadalafil', 'finasteride', 'metronidazole', 'sulfamethoxazole/trimethoprim', 'nitrofurantoin', 'valacyclovir', 'acyclovir', 'prednisolone', 'methylprednisolone dose pack', 'budesonide', 'tiotropium', 'cetirizine', 'loratadine', 'fexofenadine', 'diphenhydramine', 'promethazine', 'ondansetron', 'metoclopramide', 'docusate sodium', 'polyethylene glycol 3350', 'lactulose', 'hydralazine', 'clonidine', 'diltiazem ER', 'verapamil ER', 'nifedipine ER', 'digoxin', 'apixaban', 'rivaroxaban', 'dabigatran', 'cyanocobalamin (vitamin B12)', 'folic acid', 'ferrous sulfate', 'calcium acetate', 'sevelamer', 'sitagliptin', 'glimepiride', 'pioglitazone', 'empagliflozin', 'dapagliflozin', 'liraglutide', 'semaglutide oral', 'tirzepatide', 'mirtazapine', 'desvenlafaxine', 'vortioxetine', 'buprenorphine/naloxone', 'naltrexone', 'disulfiram', 'propranolol', 'labetalol', 'sotalol', 'amiodarone', 'isosorbide mononitrate', 'nitroglycerin SL', 'colchicine', 'febuxostat', 'methotrexate', 'hydroxychloroquine', 'prednisone dose pack', 'dexamethasone', 'triamcinolone', 'azelastine', 'mometasone', 'benzonatate', 'guaifenesin with codeine', 'lidocaine patch', 'gabapentin ER', 'pregabalin', 'baclofen', 'tizanidine', 'methocarbamol', 'acetaminophen', 'ibuprofen', 'naproxen sodium', 'aspirin', 'acetaminophen/diphenhydramine', 'loratadine', 'cetirizine', 'fexofenadine', 'diphenhydramine', 'chlorpheniramine', 'loratadine/pseudoephedrine', 'omeprazole', 'esomeprazole', 'lansoprazole', 'famotidine', 'calcium carbonate', 'calcium carbonate/magnesium hydroxide', 'simethicone', 'loperamide', 'bismuth subsalicylate', 'docusate sodium', 'polyethylene glycol 3350', 'bisacodyl', 'sennosides', 'dextromethorphan', 'guaifenesin', 'guaifenesin/dextromethorphan', 'phenylephrine', 'pseudoephedrine', 'acetaminophen/dextromethorphan/phenylephrine', 'doxylamine/acetaminophen/dextromethorphan', 'diphenhydramine (sleep aid)', 'doxylamine succinate', 'melatonin', 'meclizine', 'dimenhydrinate', 'multivitamin adult', 'vitamin D3', 'vitamin C', 'biotin', 'fish oil / omega-3', 'folic acid', 'iron (ferrous sulfate)', 'calcium + vitamin D', 'glucosamine/chondroitin', 'probiotic', 'nicotine lozenge', 'ibuprofen/famotidine', 'naproxen/esomeprazole']

print(f"{len(seed_names)} seed drug names to resolve")

drug_metadata = {}
if seed_names:
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(resolve_one, n) for n in seed_names]
        for i, fut in enumerate(as_completed(futures), 1):
            name, entries = fut.result()
            drug_metadata[name] = entries
            if i % 25 == 0:
                print(f"  {i}/{len(seed_names)} resolved")

json.dump(drug_metadata, open(METADATA_DIR / "drug_metadata.json", "w"), indent=0)
total_ndcs = sum(len(v) for v in drug_metadata.values())
print(f"Resolved {len(drug_metadata)} names / {total_ndcs} NDC entries")

# Diagnostics: if total_ndcs is 0, this tells you exactly where the pipeline
# broke instead of leaving it a silent mystery.
print(f"\nHTTP diagnostics: {_diag}")
print(f"Name-resolution diagnostics: {_diag_names}")
if _first_errors:
    print("\nFirst few underlying errors (url -> exception):")
    for url, err in _first_errors:
        print(f"  {url}\n    -> {err}")
if total_ndcs == 0:
    if _diag["ok"] == 0:
        print("\n^ EVERY RxNav HTTP call failed (0 successful responses) — this "
              "is a network/connectivity problem (Kaggle's internet access for "
              "this notebook, a firewall, or RxNav being down), not a data problem. "
              "Check Settings > Internet is On, and try opening "
              "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=aspirin&search=2 "
              "in a browser to confirm the API itself is reachable.")
    elif _diag_names["rxcui_hits"] == 0:
        print("\n^ RxNav responded successfully but returned ZERO rxcuis for "
              "every single seed name — that's implausible for common drugs "
              "like atorvastatin/metformin/ibuprofen, so the query itself is "
              "probably malformed (check the exact response shape against "
              "the printed sample above) rather than a per-name miss.")
    else:
        print("\n^ Names resolved to rxcuis but NDC lookups came back empty — "
              "check the /ndcs.json response shape against what ndcs_for_rxcui "
              "expects (ndcGroup.ndcList.ndc).")

# normalize_ndc9 is defined earlier in this cell (next to RXNAV_BASE), so
# resolve_one can also use it for early NDC9-based dedup.
PRIORITY_NDC9_SET = {
    normalize_ndc9(entry["ndc"])
    for entries in drug_metadata.values()
    for entry in entries
    if entry.get("ndc")
}
print(f"Priority tier: {len(PRIORITY_NDC9_SET)} distinct NDC9 prefixes across the seed lists")



# ============================================================================
# 3. Build the unified image manifest
# ============================================================================

def default_label_from_filename(path):
    return Path(path).stem.split("_", 1)[0]

def guess_side(path):
    name = Path(path).stem.lower()
    if any(t in name for t in ("_sf", "front", "_f_", "top")):
        return "front"
    if any(t in name for t in ("_sb", "back", "_b_", "bottom")):
        return "back"
    return "unknown"

manifest_rows = []  # dicts: path, label, side, domain, source, category ("RX"/"OTC"/"unknown")

def add_epillid():
    """Uses ePillID's own all_labels.csv (confirmed columns: images,
    pilltype_id, label_code_id, prod_code_id, is_ref, is_front, is_new,
    image_path, label) rather than guessing from filenames — it has real
    front/back and reference/pool flags. All of ePillID is RX (it's built
    from NIH's C3PI, which is RX-only by construction)."""
    root = dataset_roots.get("epillid")
    if not root:
        return
    csv_path = Path(root) / "ePillID_data" / "all_labels.csv"
    base = Path(root) / "ePillID_data" / "classification_data"
    if not csv_path.exists():
        print("epillid: all_labels.csv missing, skipping (no safe fallback)")
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    missing = 0
    for _, row in df.iterrows():
        img_path = base / row["image_path"]
        if not img_path.exists():
            missing += 1
            continue
        manifest_rows.append({
            "path": str(img_path),
            "label": row["label"],
            "side": "front" if bool(row["is_front"]) else "back",
            # is_ref=False is STILL studio-pipeline photography (ePillID's
            # larger "training pool" tier), not a real-world/phone photo.
            # Tagging it "consumer" would recreate the exact false-confidence
            # problem this whole eval redesign exists to catch.
            "domain": "reference" if bool(row["is_ref"]) else "reference_pool",
            "source": "epillid",
            "category": "RX",
        })
    if missing:
        print(f"epillid: {missing} rows in all_labels.csv had no matching file on disk")

def add_c3pi():
    """Reads c3pi_acquisition.ipynb's committed output (c3pi_manifest.csv) —
    see Section 1.2 — rather than scanning a raw C3PI_ROOT directory. That
    notebook already built the manifest (label/side/domain heuristic applied
    once, at acquisition time, and confirmed against the real rximage.zip
    structure), so this just re-roots paths the same way add_dailymed() does.

    PERFORMANCE: the acquisition notebook recorded absolute paths under its
    own /kaggle/working, which don't exist once mounted here as a notebook
    input -- so practically every row misses the direct-path check. A
    previous version of this function fell back to `alt2.rglob(filename)`
    PER ROW on a miss, i.e. a full directory-tree walk of the ~48k-file
    extracted C3PI tree for each of ~48k rows. Confirmed live: this is what
    silently ate 6+ hours of a run between "Resolved... NDC entries" and the
    next printed line, with training not starting until deep into the
    session and getting killed before a single checkpoint saved. Fixed by
    building one filename->path index via a single os.walk, then doing O(1)
    dict lookups per row instead."""
    if not C3PI_MANIFEST_PATH:
        return
    import csv as _csv3
    import os as _os3
    manifest_dir = Path(C3PI_MANIFEST_PATH).parent
    filename_index = {}
    for search_root in (manifest_dir / "c3pi_images", manifest_dir / "rximage_extracted"):
        if not search_root.exists():
            continue
        for dirpath, _dirnames, filenames in _os3.walk(search_root):
            for fn in filenames:
                filename_index.setdefault(fn, str(Path(dirpath) / fn))
    n_missing = 0
    with open(C3PI_MANIFEST_PATH) as f:
        for row in _csv3.DictReader(f):
            path = row["path"]
            if not Path(path).exists():
                path = filename_index.get(Path(path).name)
                if path is None:
                    n_missing += 1
                    continue
            manifest_rows.append({**row, "path": path})
    if n_missing:
        print(f"add_c3pi: {n_missing} manifest rows had no resolvable image file")

def add_dailymed():
    """Reads the separate dailymed_acquisition.ipynb notebook's output
    (dailymed_manifest.csv) rather than downloading live in this notebook —
    see Section 1.3. This is the only source with real US-NDC OTC coverage.

    Same indexed re-rooting as add_c3pi() (one directory walk, then O(1)
    dict lookups) rather than a single guessed sibling path per row -- more
    robust if the real extracted layout doesn't put every image directly
    under dailymed_images/, and avoids reintroducing a per-row filesystem
    search like the one that stalled add_c3pi() for hours."""
    if not DAILYMED_MANIFEST_PATH:
        return
    import csv as _csv2
    import os as _os2
    manifest_dir = Path(DAILYMED_MANIFEST_PATH).parent
    filename_index = {}
    for search_root in (manifest_dir / "dailymed_images", manifest_dir):
        if not search_root.exists():
            continue
        for dirpath, _dirnames, filenames in _os2.walk(search_root):
            for fn in filenames:
                filename_index.setdefault(fn, str(Path(dirpath) / fn))
    n_missing = 0
    with open(DAILYMED_MANIFEST_PATH) as f:
        for row in _csv2.DictReader(f):
            path = row["path"]
            if not Path(path).exists():
                path = filename_index.get(Path(path).name)
                if path is None:
                    n_missing += 1
                    continue
            manifest_rows.append({**row, "path": path})
    if n_missing:
        print(f"add_dailymed: {n_missing} manifest rows had no resolvable image file")

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

def _find_images(root):
    for pattern in IMAGE_EXTENSIONS:
        yield from Path(root).rglob(pattern)

def add_generic_detection_dataset(key):
    root = dataset_roots.get(key)
    if not root:
        return
    # These are bounding-box detection datasets (single class "pill"), not
    # per-drug labeled. They're valuable for training/improving a pill
    # cropper, not for classification labels — tag label=None so downstream
    # training code can route them to the detector step instead of the
    # classifier, rather than silently mislabeling them.
    n = 0
    for p in _find_images(root):
        manifest_rows.append({
            "path": str(p), "label": None, "side": "unknown",
            "domain": "consumer", "source": key, "category": "unknown",
        })
        n += 1
    if n == 0:
        print(f"{key}: no image files found (checked .jpg/.jpeg/.png) — "
              f"the attached dataset may only contain a non-image artifact "
              f"(e.g. a packaged/opaque file or a pretrained model weights "
              f"file), not raw images to train on.")

# Split/wrapper folder names that are never themselves a class name —
# confirmed necessary via a live run: ogyeiv2 and phvitamins_v2 both nest
# images under train/valid/test + images/labels wrapper folders rather than
# having class folders directly at the dataset root, which the original
# root.iterdir()-only version of this function missed entirely (0 images
# from either). phvitamins_v2's real class folders (e.g. "Biogesic",
# "Bonamine") sit two levels deeper, under "Capsure Dataset/Train Image/".
_SPLIT_WRAPPER_NAMES = {
    "train", "val", "valid", "validation", "test", "images", "image",
    "labels", "label", "train image", "val image", "test image",
    "annotations",
}

def add_labeled_folder_dataset(key, label_prefix=None):
    """For classification-style datasets where each image's *immediate
    parent folder* is a real class/product name (ogyeiv2, phvitamins_v2,
    drugs_vitamins_cls) — searched recursively rather than assuming class
    folders sit directly at the dataset root, since real layouts vary (see
    _SPLIT_WRAPPER_NAMES above). Labels are raw folder names (prefixed by
    source), NOT US-NDC-mapped — visual/appearance diversity, not exact
    NDC-level classes. A dataset whose images all sit under split/wrapper
    folders with no real class-name level (e.g. a pure YOLO detection
    layout with only train/images, train/labels — ogyeiv2's actual layout)
    legitimately contributes 0 labeled rows here; that's correct, not a bug,
    short of writing a YOLO-label parser to recover class names from
    per-image .txt files instead of folder names.
    """
    root = dataset_roots.get(key)
    if not root:
        return
    prefix = label_prefix or key
    n_added = 0
    for p in _find_images(root):
        class_name = p.parent.name
        if class_name.strip().lower() in _SPLIT_WRAPPER_NAMES:
            continue  # parent is a split/wrapper folder, not a real class name
        manifest_rows.append({
            "path": str(p),
            "label": f"{prefix}:{class_name}",
            "side": "unknown",
            "domain": "reference_pool",
            "source": key,
            "category": "unknown",
        })
        n_added += 1
    if n_added == 0:
        print(f"{key}: no images found with a real class-name parent folder "
              f"(only split/wrapper-named parents, or no images at all) — "
              f"this source contributes 0 rows.")

add_epillid()
add_c3pi()
add_dailymed()
# Non-US / unconfirmed-origin sources, restored for volume per an explicit
# "primarily US, maximize images per class" call. None of these are
# US-NDC-mapped, so they widen visual/appearance diversity and (for the
# detection sets) crop/detector training data — they don't substitute for
# epillid/C3PI/DailyMed's real US NDC coverage.
add_generic_detection_dataset("pills_detect_a")
add_generic_detection_dataset("pills_detect_b")
add_generic_detection_dataset("trumed_1k")
add_generic_detection_dataset("trumed_tablets")
add_labeled_folder_dataset("ogyeiv2")
add_labeled_folder_dataset("phvitamins_v2")
add_labeled_folder_dataset("drugs_vitamins_cls")

# Tag each row's tier — "priority" if it's one of the top-500 RX/OTC seed
# drugs (Section 2's PRIORITY_NDC9_SET), "long_tail" for other real NDC
# rows, "unknown" for non-NDC raw-name rows (the diversity-only datasets).
# This is the split that actually matters for the doctor-testing goal —
# reported separately, never blended into one overall number.
for row in manifest_rows:
    if row["category"] in ("RX", "OTC") and row["label"]:
        row["tier"] = "priority" if normalize_ndc9(row["label"]) in PRIORITY_NDC9_SET else "long_tail"
    else:
        row["tier"] = "unknown"

print(f"Total images collected: {len(manifest_rows)}")
by_source = Counter(r["source"] for r in manifest_rows)
print("By source:", dict(by_source))
by_cat = Counter(r["category"] for r in manifest_rows)
print("By category (RX/OTC/unknown):", dict(by_cat))
by_tier = Counter(r["tier"] for r in manifest_rows)
print("By tier (priority/long_tail/unknown):", dict(by_tier))

labeled_rows = [r for r in manifest_rows if r["label"]]

# Validate every labeled image opens before committing hours of training to
# it -- confirmed necessary via a live run: a single corrupt/unreadable
# DailyMed image (UnidentifiedImageError) crashed a 7-HOUR run at the very
# first checkpoint opportunity, with nothing saved. A bulk pipeline pulling
# ~250k files from zips-of-zips will have some bad ones; filtering them out
# up front (fast header-only check, not a full decode) means training can't
# be brought down by one of them, and PillDataset/embed_rows/quick_val_top5
# below also catch+skip per-item as a second line of defense.
def _is_openable_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

print(f"Validating {len(labeled_rows)} labeled image files before training...")
with ThreadPoolExecutor(max_workers=32) as _pool:
    _valid_flags = list(_pool.map(_is_openable_image, [r["path"] for r in labeled_rows]))
_n_before_validation = len(labeled_rows)
labeled_rows = [r for r, ok in zip(labeled_rows, _valid_flags) if ok]
_n_corrupt = _n_before_validation - len(labeled_rows)
if _n_corrupt:
    print(f"Filtered out {_n_corrupt} unreadable/corrupt image files "
          f"(of {_n_before_validation} labeled rows) before training")

by_label_count = Counter(r["label"] for r in labeled_rows)
print(f"Labeled images: {len(labeled_rows)} across {len(by_label_count)} distinct labels")
depth_dist = Counter(by_label_count.values())
print("Images-per-label distribution (count -> how many labels have that many images):",
      dict(sorted(depth_dist.items())[:15]))

import csv as _csv
with open(MANIFEST_PATH, "w", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=["path", "label", "side", "domain", "source", "category", "tier"])
    w.writeheader()
    w.writerows(manifest_rows)
print(f"Wrote manifest to {MANIFEST_PATH}")



# ============================================================================
# 4. Train / val / test split
# ============================================================================

from collections import defaultdict as _dd

rows_by_label = _dd(list)
for r in labeled_rows:
    rows_by_label[r["label"]].append(r)

train_rows, val_rows, test_rows = [], [], []
rng = random.Random(SEED)

for label, rows in rows_by_label.items():
    rng.shuffle(rows)
    consumer = [r for r in rows if r["domain"] == "consumer"]
    reference = [r for r in rows if r["domain"] != "consumer"]

    # Guarantee at least one train image; everything else splits 70/15/15,
    # with consumer images preferentially routed to val/test so that split
    # isn't reference-only.
    pool = reference[1:] + reference[:1] if len(reference) == 1 else reference
    if len(rows) == 1:
        train_rows.extend(rows)
        continue

    n_test = max(1, round(0.15 * len(rows))) if len(rows) >= 4 else (1 if consumer else 0)
    n_val = max(1, round(0.15 * len(rows))) if len(rows) >= 4 else 0

    test_pick = (consumer[:n_test] + reference)[:n_test] if consumer else reference[:n_test]
    remaining = [r for r in rows if r not in test_pick]
    val_pick = remaining[:n_val]
    train_pick = [r for r in remaining if r not in val_pick]

    train_rows.extend(train_pick)
    val_rows.extend(val_pick)
    test_rows.extend(test_pick)

print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
print("test domain mix:", Counter(r["domain"] for r in test_rows))



# ============================================================================
# 5. Domain-randomization augmentation
# ============================================================================

import albumentations as A

reference_domain_aug = A.Compose([
    A.RandomRotate90(p=0.3),
    A.Rotate(limit=25, p=0.7),
    A.Perspective(scale=(0.02, 0.08), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=20, p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=(3, 9)),
    ], p=0.4),
    A.ImageCompression(quality_range=(35, 90), p=0.6),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.05, 0.15),
                     hole_width_range=(0.05, 0.15), p=0.3),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
])

consumer_domain_aug = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.ImageCompression(quality_range=(60, 95), p=0.3),
])

def augment_for_domain(pil_image, domain):
    arr = np.array(pil_image.convert("RGB"))
    aug = reference_domain_aug if domain != "consumer" else consumer_domain_aug
    return Image.fromarray(aug(image=arr)["image"])



# ============================================================================
# 6. Dataset + P-K batch sampler
# ============================================================================

from torch.utils.data import Dataset, Sampler

class PillDataset(Dataset):
    def __init__(self, rows, processor, training):
        self.rows = rows
        self.processor = processor
        self.training = training
        self.labels = sorted({r["label"] for r in rows})
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx, _retries_left=3):
        row = self.rows[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception as e:
            # Belt-and-suspenders: Section 3 already filters out files that
            # fail Image.verify() before training starts, but a file can
            # still fail full decode (verify() only checks the header) or
            # go missing/truncated between that check and this read. A
            # single bad file crashing __getitem__ mid-batch previously took
            # down an entire 7-hour run with nothing saved -- resample a
            # different random row instead of raising.
            if _retries_left <= 0:
                raise
            print(f"PillDataset: skipping unreadable {row['path']} ({e}), resampling")
            return self.__getitem__(random.randrange(len(self.rows)), _retries_left - 1)
        if self.training:
            img = augment_for_domain(img, row["domain"])
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return pixel_values, self.label_to_idx[row["label"]]


class PKSampler(Sampler):
    """Yields batches of P classes x K samples/class, matching the
    proj_batch_p / proj_batch_k hyperparameters from pill-id's config.json."""

    def __init__(self, rows, labels, p, k, batches_per_epoch):
        self.by_label = defaultdict(list)
        for i, r in enumerate(rows):
            self.by_label[r["label"]].append(i)
        self.labels = [l for l in labels if len(self.by_label[l]) >= 1]
        self.p, self.k = p, k
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen_labels = random.sample(self.labels, min(self.p, len(self.labels)))
            batch = []
            for label in chosen_labels:
                pool = self.by_label[label]
                batch.extend(random.choices(pool, k=self.k) if len(pool) < self.k
                             else random.sample(pool, self.k))
            yield batch

    def __len__(self):
        return self.batches_per_epoch



# ============================================================================
# 7. Model: frozen DINOv2-large + projection head
# ============================================================================

from transformers import AutoImageProcessor, AutoModel

CFG = {
    "dinov2_model_id": "facebook/dinov2-large",
    "proj_hidden_dim": 1024,
    "proj_embedding_dim": 512,
    "proj_batch_p": 12,
    "proj_batch_k": 4,
    "proj_epochs": 50,
    "proj_lr": 1e-3,
    "proj_weight_decay": 1e-4,
    "ce_weight": 0.5,
    "arcface_weight": 0.3,
    "supcon_weight": 0.7,
    "triplet_weight": 0.7,
    "triplet_margin": 0.2,
    "supcon_temperature": 0.07,
    "arcface_s": 30.0,
    "arcface_m": 0.3,
    "run_lora": True,
    "lora_epochs": 10,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_lr_backbone": 1e-5,
    "lora_lr_head": 5e-4,
}

processor = AutoImageProcessor.from_pretrained(CFG["dinov2_model_id"])
backbone = AutoModel.from_pretrained(CFG["dinov2_model_id"]).to(DEVICE)
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False

# Use both GPUs on a T4 x2 session — the backbone is frozen/inference-only
# (no backward pass through it), so plain DataParallel is sufficient and
# much simpler than DDP: it just splits each batch across available GPUs
# for the forward pass and gathers the results. Selecting T4 x2 without this
# wrapper only ever used cuda:0, wasting half the allocated compute (and
# likely being charged the dual-GPU quota rate regardless).
_n_gpus = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
if _n_gpus > 1:
    print(f"Wrapping backbone in nn.DataParallel across {_n_gpus} GPUs")
    backbone = nn.DataParallel(backbone)
else:
    print(f"Single GPU or CPU ({_n_gpus} GPU(s) visible) — no DataParallel wrapping")


class ProjectionHead(nn.Module):
    """Kept identical to pill-id/backend/app/classifier.py's ProjectionHead
    so exported weights load without modification in the existing backend."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_classes):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(0.10), nn.Linear(hidden_dim, out_dim),
        )
        self.classifier = nn.Linear(out_dim, num_classes)
        self.arc_weight = nn.Parameter(torch.empty(num_classes, out_dim))
        nn.init.xavier_uniform_(self.arc_weight)

    def forward(self, x):
        emb = F.normalize(self.proj(x), dim=1)
        logits = self.classifier(emb)
        return {"emb": emb, "logits": logits}


def extract_backbone_feature(pixel_values):
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
        out = backbone(pixel_values=pixel_values.to(DEVICE))
        feat = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return feat.float()  # cast back to fp32 before the (unfrozen, non-autocast) projection head


def arcface_logits(emb, labels, weight, s, m, num_classes):
    weight_n = F.normalize(weight, dim=1)
    cos = emb @ weight_n.T
    theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
    target_logits = torch.cos(theta + m)
    one_hot = F.one_hot(labels, num_classes).float()
    logits = cos * (1 - one_hot) + target_logits * one_hot
    return logits * s


def supcon_loss(emb, labels, temperature):
    sim = emb @ emb.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim)
    mask_self = torch.eye(len(labels), device=emb.device).bool()
    exp_sim = exp_sim.masked_fill(mask_self, 0)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self
    denom = exp_sim.sum(dim=1) + 1e-12
    log_prob = sim - torch.log(denom.unsqueeze(1) + 1e-12)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=1) / pos_count
    return loss.mean()


def triplet_loss(emb, labels, margin):
    dist = torch.cdist(emb, emb)
    loss_terms = []
    for i in range(len(labels)):
        pos_mask = (labels == labels[i]) & (torch.arange(len(labels), device=emb.device) != i)
        neg_mask = labels != labels[i]
        if pos_mask.any() and neg_mask.any():
            hardest_pos = dist[i][pos_mask].max()
            hardest_neg = dist[i][neg_mask].min()
            loss_terms.append(F.relu(hardest_pos - hardest_neg + margin))
    return torch.stack(loss_terms).mean() if loss_terms else torch.tensor(0.0, device=emb.device)



# ============================================================================
# 8. Training loop
# ============================================================================

import time as _time
import random as _random

# Bounded regardless of dataset size, and a hard wall-clock budget, so the
# training loop always finishes and reaches export/gate rather than risking
# a mid-run kill from Kaggle's session time limit with nothing saved. The
# dataset is now ~230k+ images (much bigger than when this loop was first
# written) - an unbounded batches_per_epoch derived from len(train_rows)
# would make even one epoch's duration unpredictable. Widen these once a
# real run has been timed.
MAX_BATCHES_PER_EPOCH = 800   # 800 * (proj_batch_p*proj_batch_k=48) ~= 38k images/epoch
# Confirmed live at ~3.7s/batch (2 GPUs, DataParallel) -> ~50min/epoch. Was
# 6h; bumped to 7h now that the quick_val_top5 and C3PI-path-search bugs
# (which used to burn most of a run's wall clock on validation/preprocessing
# instead of training) are fixed -- with a ~13h Kaggle quota, ~30min
# preprocessing, and export+gate now batched (not the multi-hour unbatched
# cost from before), 7h training leaves real buffer for export/gate to
# still complete and save artifacts even if that estimate is optimistic.
MAX_TRAIN_SECONDS = 7 * 3600
PRINT_EVERY_N_BATCHES = 25    # frequent feedback instead of silence for a whole epoch
VAL_EVERY_N_BATCHES = 200     # cheap periodic validation for best-checkpoint selection

@torch.no_grad()
def _embed_for_quickval(rows, head, batch_size=64):
    """Skips (rather than crashes on) an unreadable image -- this exact
    function is what a real run crashed inside on a corrupt DailyMed file,
    taking down 7 hours of training with no checkpoint saved.

    Batched (not one image at a time) -- confirmed live: an unbatched
    version of this loop, combined with quick_val_top5's gallery covering
    every train class, turned a "quick" validation into an ~80-90 MINUTE
    stall every 200 steps (most of the 6h training budget going to
    validation, not training). Batching alone doesn't fix the gallery-size
    problem (see quick_val_top5), but it's still a real, honest speedup on
    top of that fix."""
    lbls = []
    embs = []
    n_skipped = 0
    buf_imgs, buf_lbls = [], []

    def _flush():
        nonlocal n_skipped
        if not buf_imgs:
            return
        try:
            pv = processor(images=buf_imgs, return_tensors="pt")["pixel_values"]
            feat = extract_backbone_feature(pv)
            emb = head(feat)["emb"]
            embs.append(emb.cpu())
            lbls.extend(buf_lbls)
        except Exception as e:
            # A whole-batch failure (e.g. one image with an unusual mode
            # slipping past .convert("RGB")) shouldn't lose the rest of the
            # batch -- retry one image at a time and skip only the actual
            # offender(s).
            print(f"_embed_for_quickval: batch of {len(buf_imgs)} failed ({e}), retrying individually")
            for img, lbl in zip(buf_imgs, buf_lbls):
                try:
                    pv = processor(images=[img], return_tensors="pt")["pixel_values"]
                    feat = extract_backbone_feature(pv)
                    emb = head(feat)["emb"]
                    embs.append(emb.cpu())
                    lbls.append(lbl)
                except Exception:
                    n_skipped += 1

    for r in rows:
        try:
            img = Image.open(r["path"]).convert("RGB")
        except Exception:
            n_skipped += 1
            continue
        buf_imgs.append(img)
        buf_lbls.append(r["label"])
        if len(buf_imgs) >= batch_size:
            _flush()
            buf_imgs, buf_lbls = [], []
    _flush()

    if n_skipped:
        print(f"_embed_for_quickval: skipped {n_skipped} unreadable image(s)")
    if not embs:
        return torch.empty(0), []
    return torch.cat(embs, dim=0), lbls

@torch.no_grad()
def quick_val_top5(head, train_rows, val_rows, gallery_per_class=2, sample_size=200,
                    max_gallery_classes=400):
    """Cheap proxy top-5 (small capped gallery + a val sample, not the full
    eval/evaluate.py harness) — just for in-training checkpoint selection,
    not a substitute for the real Section 10 gate on the full test set.

    max_gallery_classes actually caps it: with 63k+ train classes,
    "2 images x every class" is up to ~127,000 images, not a small gallery
    -- confirmed live, this (plus running unbatched) turned validation into
    an 80+ minute stall every 200 steps, eating most of the training
    budget. A random subset of classes is still a fair (if noisier) signal
    for "is the head learning at all," which is all this proxy needs to be
    for checkpoint selection -- the real per-class-complete numbers come
    from the Section 10 gate on the full test set after training."""
    head.eval()
    from collections import defaultdict as _dd3
    by_label = _dd3(list)
    for r in train_rows:
        by_label[r["label"]].append(r)
    gallery_classes = list(by_label.keys())
    if len(gallery_classes) > max_gallery_classes:
        gallery_classes = _random.sample(gallery_classes, max_gallery_classes)
    gallery_rows = [r for c in gallery_classes for r in by_label[c][:gallery_per_class]]
    # Filter to gallery_classes BEFORE sampling, not after -- sampling 200
    # random val rows first and filtering after would leave ~1 usable query
    # on average (400/63604 train classes), making the proxy meaningless.
    gallery_class_set = set(gallery_classes)
    query_pool = [r for r in val_rows if r["label"] in gallery_class_set]
    query_rows = query_pool if len(query_pool) <= sample_size else _random.sample(query_pool, sample_size)
    if not gallery_rows or not query_rows:
        head.train()
        return None

    gal_emb, gal_lbls = _embed_for_quickval(gallery_rows, head)
    q_emb, q_lbls = _embed_for_quickval(query_rows, head)
    if not gal_lbls or not q_lbls:
        head.train()
        return None
    gal_emb_n = F.normalize(gal_emb, dim=1)
    q_emb_n = F.normalize(q_emb, dim=1)
    sims = q_emb_n @ gal_emb_n.T
    hits = 0
    for i, true_label in enumerate(q_lbls):
        topk = torch.topk(sims[i], k=min(5, sims.shape[1])).indices.tolist()
        if true_label in {gal_lbls[j] for j in topk}:
            hits += 1
    head.train()
    return hits / len(q_lbls)


def train_projection_head(train_rows, val_rows, cfg):
    labels = sorted({r["label"] for r in train_rows})
    train_ds = PillDataset(train_rows, processor, training=True)
    natural_batches = max(1, len(train_rows) // (cfg["proj_batch_p"] * cfg["proj_batch_k"]))
    batches_per_epoch = min(natural_batches, MAX_BATCHES_PER_EPOCH)
    sampler = PKSampler(train_rows, labels, cfg["proj_batch_p"], cfg["proj_batch_k"],
                         batches_per_epoch=batches_per_epoch)
    print(f"train_projection_head: {batches_per_epoch} batches/epoch "
          f"(capped from {natural_batches} natural batches), {len(labels)} classes", flush=True)

    head = ProjectionHead(in_dim=1024, hidden_dim=cfg["proj_hidden_dim"],
                           out_dim=cfg["proj_embedding_dim"], num_classes=len(labels)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["proj_lr"], weight_decay=cfg["proj_weight_decay"])

    best_val_top5 = -1.0
    best_state_dict = None
    global_step = 0
    start_time = _time.time()
    for epoch in range(cfg["proj_epochs"]):
        elapsed = _time.time() - start_time
        if elapsed > MAX_TRAIN_SECONDS:
            print(f"MAX_TRAIN_SECONDS budget ({MAX_TRAIN_SECONDS}s) reached after "
                  f"{epoch} epoch(s) — stopping here so export/gate still run.", flush=True)
            break

        head.train()
        epoch_loss = 0.0
        epoch_start = _time.time()
        for batch_num, batch_indices in enumerate(sampler, 1):
            pixel_values = torch.stack([train_ds[i][0] for i in batch_indices]).to(DEVICE)
            batch_labels = torch.tensor([train_ds[i][1] for i in batch_indices], device=DEVICE)

            feat = extract_backbone_feature(pixel_values)
            out = head(feat)
            emb, logits = out["emb"], out["logits"]

            ce = F.cross_entropy(logits, batch_labels)
            arc_logits = arcface_logits(emb, batch_labels, head.arc_weight, cfg["arcface_s"], cfg["arcface_m"], len(labels))
            arc = F.cross_entropy(arc_logits, batch_labels)
            sc = supcon_loss(emb, batch_labels, cfg["supcon_temperature"])
            tr = triplet_loss(emb, batch_labels, cfg["triplet_margin"])

            loss = (cfg["ce_weight"] * ce + cfg["arcface_weight"] * arc
                    + cfg["supcon_weight"] * sc + cfg["triplet_weight"] * tr)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            global_step += 1

            if batch_num % PRINT_EVERY_N_BATCHES == 0:
                print(f"  epoch {epoch+1} batch {batch_num}/{batches_per_epoch} "
                      f"loss={loss.item():.4f} ({_time.time() - epoch_start:.0f}s into this epoch)", flush=True)

            if global_step % VAL_EVERY_N_BATCHES == 0:
                val_top5 = quick_val_top5(head, train_rows, val_rows)
                if val_top5 is not None:
                    print(f"  [quick val] step {global_step}: top5≈{val_top5:.3f} "
                          f"(best so far: {max(best_val_top5, val_top5):.3f})", flush=True)
                    if val_top5 > best_val_top5:
                        best_val_top5 = val_top5
                        best_state_dict = {k: v.detach().clone() for k, v in head.state_dict().items()}
                        torch.save({"head_state_dict": best_state_dict, "cfg": cfg,
                                    "in_dim": 1024, "num_classes": len(labels),
                                    "label_classes": labels, "val_top5": val_top5,
                                    "global_step": global_step},
                                   EXPORT_DIR / "best_projection_head.pt")
                        print(f"  -> new best checkpoint saved (val_top5={val_top5:.3f})", flush=True)

        epoch_secs = _time.time() - epoch_start
        print(f"epoch {epoch+1}/{cfg['proj_epochs']} loss={epoch_loss / len(sampler):.4f} "
              f"({epoch_secs:.0f}s, {_time.time() - start_time:.0f}s total elapsed)", flush=True)

        # Cheap per-epoch checkpoint (head weights only, no gallery
        # re-embedding) so progress survives even if this is the last epoch
        # that completes before a session-level kill — separate from the
        # best-val checkpoint above, this is just "most recent."
        torch.save({"head_state_dict": head.state_dict(), "cfg": cfg,
                    "in_dim": 1024, "num_classes": len(labels), "label_classes": labels,
                    "epoch": epoch + 1},
                   EXPORT_DIR / "latest_checkpoint.pt")

    if best_state_dict is not None:
        print(f"Restoring best-validated checkpoint (val_top5≈{best_val_top5:.3f}) "
              f"instead of returning the last epoch's weights.", flush=True)
        head.load_state_dict(best_state_dict)
    else:
        print("No quick-val checkpoint was ever taken (training too short?) — "
              "returning the final epoch's weights as-is.", flush=True)

    return head, labels

# Pre-flight summary — know what you're about to spend GPU time on before it
# actually starts training.
print("=" * 60, flush=True)
print("PRE-TRAINING SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"Train rows: {len(train_rows)} | Val rows: {len(val_rows)} | Test rows: {len(test_rows)}", flush=True)
_train_cat = Counter(r["category"] for r in train_rows)
print(f"Train by category: {dict(_train_cat)}", flush=True)
_train_tier = Counter(r["tier"] for r in train_rows)
print(f"Train by tier: {dict(_train_tier)}", flush=True)
print(f"Distinct classes in train: {len(set(r['label'] for r in train_rows))}", flush=True)
print(f"Total distinct classes across all rows: {len(set(r['label'] for r in labeled_rows))}", flush=True)
_natural_batches = max(1, len(train_rows) // (CFG['proj_batch_p'] * CFG['proj_batch_k']))
_capped_batches = min(_natural_batches, MAX_BATCHES_PER_EPOCH)
print(f"Batch size: {CFG['proj_batch_p'] * CFG['proj_batch_k']} "
      f"({CFG['proj_batch_p']} classes x {CFG['proj_batch_k']} images)", flush=True)
print(f"Batches/epoch: {_capped_batches} (capped from {_natural_batches} natural)", flush=True)
print(f"Max epochs configured: {CFG['proj_epochs']} | Wall-clock budget: {MAX_TRAIN_SECONDS/3600:.1f}h "
      f"(whichever limit hits first stops training and moves to export/gate)", flush=True)
print("=" * 60, flush=True)

try:
    head, label_list = train_projection_head(train_rows, val_rows, CFG)
except Exception as e:
    # Last-resort fallback: if training itself crashes despite the
    # resampling/skip logic already in PillDataset/quick_val_top5, don't
    # lose the run -- load whichever checkpoint train_projection_head last
    # wrote to disk (best_projection_head.pt if any quickval ever ran,
    # otherwise latest_checkpoint.pt from the most recent completed epoch)
    # and let export/gate still run against real trained weights instead
    # of aborting the whole notebook with nothing usable.
    import traceback as _tb2
    print(f"train_projection_head CRASHED: {type(e).__name__}: {e}", flush=True)
    print(_tb2.format_exc(), flush=True)
    _fallback_ckpt = None
    for _name in ("best_projection_head.pt", "latest_checkpoint.pt"):
        _p = EXPORT_DIR / _name
        if _p.exists():
            _fallback_ckpt = _p
            break
    if _fallback_ckpt is None:
        raise RuntimeError(
            "Training crashed and no checkpoint was ever saved -- nothing "
            "to fall back to. See the traceback above for the real cause."
        ) from e
    print(f"Falling back to {_fallback_ckpt} (from before the crash)", flush=True)
    _ckpt = torch.load(_fallback_ckpt, map_location=DEVICE)
    label_list = _ckpt["label_classes"]
    head = ProjectionHead(in_dim=_ckpt["in_dim"], hidden_dim=CFG["proj_hidden_dim"],
                           out_dim=CFG["proj_embedding_dim"], num_classes=_ckpt["num_classes"]).to(DEVICE)
    head.load_state_dict(_ckpt["head_state_dict"])



# ============================================================================
# 9. Export reference gallery + deployment artifacts
# ============================================================================

@torch.no_grad()
def embed_rows(rows, head, label_list, batch_size=64):
    """label_list must be the model's canonical training label order (the
    list train_projection_head returned) — NOT re-derived from `rows` here,
    since that silently corrupted label_indices whenever `rows` didn't cover
    every training class in the same sorted order. Rows whose label isn't in
    label_list (e.g. a val/test singleton class never seen in training) are
    skipped and reported, not silently mis-indexed; returns the kept rows
    alongside the embeddings so callers can zip them safely.

    Batched -- this is called on the full train+val set (180k+ images) for
    the deployment gallery export, and on the full test set for the
    accuracy gate. An unbatched version of this exact pattern (one image at
    a time through DINOv2-large) is what turned "quick" in-training
    validation into an 80+ minute stall in a real run; run unbatched here,
    on 180k+ images, it would cost hours post-training for no reason."""
    head.eval()
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    embeddings, label_indices, abs_paths, kept_rows = [], [], [], []
    skipped = 0
    unreadable = 0
    buf_imgs, buf_rows = [], []

    def _flush():
        nonlocal unreadable
        if not buf_imgs:
            return
        try:
            pixel_values = processor(images=buf_imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
            feat = extract_backbone_feature(pixel_values)
            emb = head(feat.float())["emb"]
            embeddings.append(emb.cpu())
            for r in buf_rows:
                label_indices.append(label_to_idx[r["label"]])
                abs_paths.append(r["path"])
                kept_rows.append(r)
        except Exception as e:
            # Same reasoning as _embed_for_quickval: don't lose an entire
            # 64-image batch (this function embeds the full deployment
            # gallery and the gate's test set -- both too expensive to
            # risk on one bad image) -- retry individually and skip only
            # the actual offender(s).
            print(f"embed_rows: batch of {len(buf_imgs)} failed ({e}), retrying individually")
            for img, r in zip(buf_imgs, buf_rows):
                try:
                    pixel_values = processor(images=[img], return_tensors="pt")["pixel_values"].to(DEVICE)
                    feat = extract_backbone_feature(pixel_values)
                    emb = head(feat.float())["emb"]
                    embeddings.append(emb.cpu())
                    label_indices.append(label_to_idx[r["label"]])
                    abs_paths.append(r["path"])
                    kept_rows.append(r)
                except Exception:
                    unreadable += 1

    for row in rows:
        if row["label"] not in label_to_idx:
            skipped += 1
            continue
        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception:
            unreadable += 1
            continue
        buf_imgs.append(img)
        buf_rows.append(row)
        if len(buf_imgs) >= batch_size:
            _flush()
            buf_imgs, buf_rows = [], []
    _flush()

    if skipped:
        print(f"embed_rows: skipped {skipped} rows with a label not in the model's training label_list")
    if unreadable:
        print(f"embed_rows: skipped {unreadable} rows with an unreadable image file")
    return {
        "embeddings": torch.cat(embeddings, dim=0) if embeddings else torch.empty(0),
        "label_indices": torch.tensor(label_indices, dtype=torch.long),
        "abs_paths": abs_paths,
    }, kept_rows


def export_artifacts(head, label_list, gallery_rows):
    """Returns the gallery embeddings it computed -- the Section 10 gate
    needs the exact same train+val embeddings for its reference gallery,
    and re-embedding 180k+ images a second time from scratch would double
    an already-expensive pass for no reason."""
    torch.save({
        "head_state_dict": head.state_dict(),
        "cfg": CFG,
        "in_dim": 1024,
        "num_classes": len(label_list),
        "label_classes": label_list,
    }, EXPORT_DIR / "best_projection_head.pt")

    gallery, _ = embed_rows(gallery_rows, head, label_list)
    torch.save(gallery, EXPORT_DIR / "deployed_ref_embeddings.pt")
    print(f"Exported artifacts to {EXPORT_DIR}")
    return gallery

deployed_gallery = export_artifacts(head, label_list, train_rows + val_rows)



# ============================================================================
# 10. Accuracy gate — mandatory before deploying to the app
# ============================================================================

# Inlined from eval/evaluate.py (kept in sync manually) — GitHub-repo
# attachment has been unreliable in practice, so this cell doesn't depend on
# it for the unattended overnight run.
from dataclasses import dataclass as _dataclass
from collections import defaultdict as _defaultdict

TOPKS = (1, 5, 10, 20, 50)
CONFIDENCE_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]

@_dataclass
class QueryRow:
    embedding: torch.Tensor
    true_label: str
    domain: str
    side: str
    images_in_class: int
    category: str = "unknown"
    tier: str = "unknown"  # "priority" (top-500 RX/OTC) | "long_tail" | "unknown"

def _topk_hit(true_label, ranked_labels, k):
    return true_label in ranked_labels[:k]

def _rank_chunk(query_emb_n, gallery_emb_n, gallery_labels):
    """Ranks a whole chunk of queries against the gallery in one batched
    matmul. Chunked (not the whole query set at once) to keep memory
    bounded against a 180k+-image gallery -- a single (n_queries x
    gallery_size) similarity matrix at that scale would be tens of GB."""
    sims_chunk = query_emb_n @ gallery_emb_n.T
    order_chunk = torch.argsort(sims_chunk, dim=1, descending=True)
    top1_chunk = sims_chunk.max(dim=1).values
    out = []
    for i in range(order_chunk.shape[0]):
        seen = set()
        ranked = []
        for idx in order_chunk[i].tolist():
            lbl = gallery_labels[idx]
            if lbl not in seen:
                seen.add(lbl)
                ranked.append(lbl)
        out.append((ranked, float(top1_chunk[i])))
    return out

def _rank_all_rows(rows, gallery_emb, gallery_labels, chunk_size=256):
    """Ranks every row exactly once, normalizing the gallery exactly once --
    the naive per-axis approach (overall/domain/category/tier/
    tier_and_category/images_per_class/worst_classes = 7 axes) used to
    recompute this from scratch per axis, i.e. re-normalize the entire
    ~180k-image gallery up to 7x PER TEST ROW. At tens of thousands of test
    rows that's billions of wasted floating-point ops -- confirmed to
    matter at this project's actual scale, not a micro-optimization."""
    if not rows:
        return []
    gallery_emb_n = F.normalize(gallery_emb, dim=1)
    query_emb_n = F.normalize(torch.stack([r.embedding for r in rows]), dim=1)
    precomputed = []
    for start in range(0, len(rows), chunk_size):
        chunk_result = _rank_chunk(query_emb_n[start:start + chunk_size], gallery_emb_n, gallery_labels)
        for local_i, (ranked, top1_score) in enumerate(chunk_result):
            row = rows[start + local_i]
            correct = bool(ranked and ranked[0] == row.true_label)
            precomputed.append((ranked, top1_score, correct))
    return precomputed

def _accuracy_block(rows, rankings):
    hits = {k: 0 for k in TOPKS}
    top1_scores, correct_at_top1 = [], []
    n = 0
    for row, (ranked, top1_score, correct) in zip(rows, rankings):
        if not ranked:
            continue
        n += 1
        for k in TOPKS:
            if _topk_hit(row.true_label, ranked, k):
                hits[k] += 1
        top1_scores.append(top1_score)
        correct_at_top1.append(correct)
    if n == 0:
        return {"n": 0}
    result = {"n": n, **{f"top{k}_acc": hits[k] / n for k in TOPKS}}
    calibration = []
    for t in CONFIDENCE_THRESHOLDS:
        answered = [c for s, c in zip(top1_scores, correct_at_top1) if s >= t]
        coverage = len(answered) / n
        precision = (sum(answered) / len(answered)) if answered else None
        calibration.append({"threshold": t, "coverage": round(coverage, 4), "precision_if_answered": precision})
    result["calibration"] = calibration
    return result

def evaluate(rows, gallery_emb, gallery_labels):
    report = {}
    rankings = _rank_all_rows(rows, gallery_emb, gallery_labels)

    def _block_for(indices):
        return _accuracy_block([rows[i] for i in indices], [rankings[i] for i in indices])

    report["overall"] = _block_for(list(range(len(rows))))
    by_domain = _defaultdict(list)
    for i, r in enumerate(rows):
        by_domain[r.domain].append(i)
    report["by_domain"] = {d: _block_for(idxs) for d, idxs in by_domain.items()}
    by_category = _defaultdict(list)
    for i, r in enumerate(rows):
        by_category[r.category].append(i)
    report["by_category"] = {c: _block_for(idxs) for c, idxs in by_category.items()}
    by_tier = _defaultdict(list)
    for i, r in enumerate(rows):
        by_tier[r.tier].append(i)
    report["by_tier"] = {t: _block_for(idxs) for t, idxs in by_tier.items()}
    # The actual granular goal numbers: top-500 RX and top-500 OTC as their
    # own distinct accuracies, not just tier and category reported separately.
    by_group = _defaultdict(list)
    for i, r in enumerate(rows):
        by_group[f"{r.tier}_{r.category}"].append(i)
    report["by_tier_and_category"] = {g: _block_for(idxs) for g, idxs in by_group.items()}
    by_depth = _defaultdict(list)
    for i, r in enumerate(rows):
        bucket = "1-2" if r.images_in_class <= 2 else "3-5" if r.images_in_class <= 5 else "6+"
        by_depth[bucket].append(i)
    report["by_images_per_class"] = {b: _block_for(idxs) for b, idxs in by_depth.items()}
    per_class = _defaultdict(list)
    for row, (ranked, _, _) in zip(rows, rankings):
        per_class[row.true_label].append(_topk_hit(row.true_label, ranked, 5))
    worst = sorted(
        ((label, sum(hits) / len(hits), len(hits)) for label, hits in per_class.items()),
        key=lambda x: x[1],
    )[:50]
    report["worst_classes_top5"] = [
        {"label": label, "top5_acc": round(acc, 3), "n_queries": n} for label, acc, n in worst
    ]
    return report

MIN_TOP5_CONSUMER = 0.70  # tune this to what you actually need before shipping

# Everything below is wrapped so that ANY unexpected failure here (the gate
# is the most complex remaining code path: it touches every row's
# category/tier/domain and the full similarity ranking) still leaves a
# RUN_SUMMARY.txt behind explaining what happened -- the model weights and
# gallery are already safely on disk from Section 9's export_artifacts
# BEFORE this cell runs, so a gate-only failure should never look like the
# whole run produced nothing.
_summary_lines = []
def _log(line):
    print(line, flush=True)
    _summary_lines.append(line)

_summary_lines.append("PILL SCANNER RUN SUMMARY")
_summary_lines.append("=" * 60)
_summary_lines.append(f"Total images in manifest: {len(manifest_rows)}")
_summary_lines.append(f"Labeled images (post-validation): {len(labeled_rows)}")
_summary_lines.append(f"By source: {dict(by_source)}")
_summary_lines.append(f"By category: {dict(by_cat)}")
_summary_lines.append(f"By tier: {dict(by_tier)}")
_summary_lines.append(f"Priority-tier NDC9 prefixes resolved: {len(PRIORITY_NDC9_SET)}")
_summary_lines.append(f"Train/val/test rows: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
_summary_lines.append(f"Distinct classes: {len(label_list)}")
_summary_lines.append("")

try:
    if evaluate is None:
        _log("Gate skipped: evaluate() unavailable (should not happen -- inlined in this cell).")
    else:
        # Reuse the gallery export_artifacts already computed (Section 9) --
        # same train+val rows, same head, same label_list, so re-embedding
        # them here would just be a second multi-hour pass over the same
        # 180k+ images for an identical result.
        gallery = deployed_gallery
        test_embedded, test_kept_rows = embed_rows(test_rows, head, label_list)
        query_rows = [
            QueryRow(embedding=emb, true_label=r["label"], domain=r["domain"],
                     side=r["side"], images_in_class=by_label_count[r["label"]],
                     category=r["category"], tier=r.get("tier", "unknown"))
            for r, emb in zip(test_kept_rows, test_embedded["embeddings"])
        ]
        report = evaluate(query_rows, gallery["embeddings"], [label_list[i] for i in gallery["label_indices"].tolist()])
        json.dump(report, open(EXPORT_DIR / "eval_report.json", "w"), indent=2, default=str)
        consumer_top5 = report["by_domain"].get("consumer", {}).get("top5_acc")
        otc_top5 = report["by_category"].get("OTC", {}).get("top5_acc")
        priority_top5 = report["by_tier"].get("priority", {}).get("top5_acc")
        priority_top10 = report["by_tier"].get("priority", {}).get("top10_acc")
        _log(f"OVERALL: n={report['overall'].get('n')} top1={report['overall'].get('top1_acc')} "
             f"top5={report['overall'].get('top5_acc')} top10={report['overall'].get('top10_acc')}")
        _log(f"consumer-domain top5: {consumer_top5} | OTC-category top5: {otc_top5}")
        _log(f"PRIORITY TIER (top-500 RX/OTC -- the actual doctor-testing goal): "
             f"top5={priority_top5} top10={priority_top10}")
        _log("By the 500 most common drugs, RX and OTC separately (the real goal numbers):")
        for group, block in sorted(report["by_tier_and_category"].items()):
            _log(f"  {group}: n={block.get('n')} top5={block.get('top5_acc')} top10={block.get('top10_acc')}")
        _log("Worst 10 classes by top5 accuracy:")
        for w in report["worst_classes_top5"][:10]:
            _log(f"  {w['label']}: top5={w['top5_acc']} (n={w['n_queries']})")
        if consumer_top5 is not None and consumer_top5 >= MIN_TOP5_CONSUMER:
            (EXPORT_DIR / "APPROVED_FOR_DEPLOY").write_text(f"consumer_top5={consumer_top5} otc_top5={otc_top5}")
            _log(f"GATE PASSED -- artifacts in {EXPORT_DIR} are approved to copy into pill-scanner/backend.")
        else:
            _log("GATE FAILED -- do not deploy this model. Get more consumer-domain data or train longer.")
except Exception as e:
    import traceback as _tb
    _log(f"GATE CRASHED: {type(e).__name__}: {e}")
    _log("Full traceback (model weights + gallery from Section 9 are still saved on disk regardless):")
    _log(_tb.format_exc())

(EXPORT_DIR / "RUN_SUMMARY.txt").write_text("\n".join(_summary_lines))
print(f"\nWrote {EXPORT_DIR / 'RUN_SUMMARY.txt'} -- read this first if anything looks wrong.")

