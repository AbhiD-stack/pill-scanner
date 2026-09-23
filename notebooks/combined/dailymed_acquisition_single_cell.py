# ============================================================================
# DailyMed Acquisition — US RX+OTC pill images + metadata
# ============================================================================


# ============================================================================
# 0. Setup
# ============================================================================
import json
import re
import shutil
import time
import zipfile
import urllib.request
import urllib.parse
from dataclasses import dataclass, field as _field
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

MIN_FREE_DISK_BYTES = 2 * 1024**3  # stop starting new parts below 2GB free, rather than crash mid-part

WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(exist_ok=True)
SPL_IMAGE_DIR = WORK_DIR / "dailymed_images"
SPL_IMAGE_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = WORK_DIR / "dailymed_manifest.csv"
METADATA_PATH = WORK_DIR / "dailymed_metadata.json"
STATE_PATH = WORK_DIR / "acquisition_state.json"

print("Setup OK. Working directory:", WORK_DIR)


# ============================================================================
# 1. Scope + discover current zip URLs
# ============================================================================
SPL_RESOURCES_PAGE = "https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm"

# Widened from the original conservative 2-OTC-part default: a real training
# run showed only 6,728 OTC images total (vs. 188,464 RX) and just 9 test
# rows landing in the priority_OTC tier -- nowhere near covering the top-500
# OTC drugs. Fetch EVERY discovered OTC part (not a hardcoded count, since
# DailyMed's part count has grown before) to close that gap.
#
# RX was originally left at just part1 ("RX coverage is not the current
# gap") -- that assumption is now stale. The seed list grew from 150 to
# 337 real RX drug names (ClinCalc-sourced), and Trial 6's real gate
# numbers showed priority-tier images/class is thin on BOTH sides (median
# ~2.8 images/class for the combined priority tier), not just OTC. A
# single arbitrary RX part almost certainly doesn't cover many of the
# newly-added, lower-prescription-volume drugs. Widened to fetch every
# discovered RX part too, same as OTC -- this is real free data DailyMed
# already has, not a new external source needed. The existing disk-safety
# guard (MIN_FREE_DISK_BYTES, stop starting new parts below 2GB free)
# means this fails safe (stops early) rather than crashing if the RX
# corpus turns out too large for the session's disk, so widening carries
# low risk even without knowing the exact part count/size in advance.
SPL_PARTS_TO_FETCH = None  # resolved below, after discovery, to "all discovered RX + OTC parts"

# Safety cap per part so one huge part can't silently run for many hours
# unattended. None = no cap (process every document in the part).
MAX_DOCS_PER_PART = None

def discover_spl_zip_urls():
    resp = requests.get(SPL_RESOURCES_PAGE, timeout=30)
    resp.raise_for_status()
    hrefs = re.findall(
        r'href="([^"]*dm_spl_release_(?:human_rx|human_otc)_part\d+\.zip)"',
        resp.text,
    )
    urls = {}
    for href in hrefs:
        url = href if (href.startswith("http") or href.startswith("ftp")) else f"https://dailymed.nlm.nih.gov{href}"
        m = re.search(r"(human_rx_part\d+|human_otc_part\d+)\.zip", url)
        if m:
            urls[m.group(1)] = url
    return urls

spl_zip_urls = discover_spl_zip_urls()

if SPL_PARTS_TO_FETCH is None:
    otc_parts_found = sorted(p for p in spl_zip_urls if p.startswith("human_otc_part"))
    rx_parts_found = sorted(p for p in spl_zip_urls if p.startswith("human_rx_part"))
    SPL_PARTS_TO_FETCH = otc_parts_found + rx_parts_found
    print(f"Resolved SPL_PARTS_TO_FETCH to all {len(otc_parts_found)} discovered OTC parts "
          f"+ all {len(rx_parts_found)} discovered RX parts: {SPL_PARTS_TO_FETCH}")

print(f"\nDiscovered {len(spl_zip_urls)} SPL zip URLs total:")
for part, url in sorted(spl_zip_urls.items()):
    marker = " <-- WILL FETCH" if part in SPL_PARTS_TO_FETCH else ""
    print(f"  {part}: {url}{marker}")

missing = [p for p in SPL_PARTS_TO_FETCH if p not in spl_zip_urls]
if missing:
    print(f"\nWARNING: requested parts not found on the page: {missing}")


# ============================================================================
# 2. SPL parser (inlined — no external file dependency)
# ============================================================================
SPL_NS = {"v3": "urn:hl7-org:v3"}
ORAL_SOLID_DOSAGE_FORM_CODES = {
    "C25158", "C42895", "C42896", "C42917", "C42902", "C42904", "C42916",
    "C42928", "C42936", "C42954", "C42998", "C42893", "C42897", "C60997",
    "C42905", "C42997", "C42910", "C42927", "C42931", "C42930", "C61004",
    "C61005", "C42964", "C42963", "C42999", "C61006", "C42985", "C42992",
}

@dataclass
class SplPillRecord:
    ndc: str
    setid: str | None
    name: str | None
    rx_or_otc: str
    imprint: str | None = None
    color: str | None = None
    shape: str | None = None
    score_marks: str | None = None
    size_mm: str | None = None
    image_refs: list = _field(default_factory=list)

def _text(el):
    return el.text.strip() if el is not None and el.text else None

PACKAGE_LABEL_SECTION_CODE = "51945-4"  # LOINC: "PACKAGE LABEL.PRINCIPAL DISPLAY PANEL"

def _document_image_refs(root):
    # Confirmed against a live full-scale run: an unscoped ".//observationMedia"
    # search (this function's first version) also picks up unrelated images
    # elsewhere in the document - RX labels especially embed chemical
    # structure diagrams, dosing charts, etc. RX part1 alone produced ~4.4
    # images/pill-record vs ~1.3-1.7 for OTC, a strong signal of
    # contamination. Restricting to <section> elements coded 51945-4
    # (PACKAGE LABEL.PRINCIPAL DISPLAY PANEL) keeps just package/box photos.
    refs = []
    for section in root.iterfind(".//v3:section", SPL_NS):
        code_el = section.find("./v3:code", SPL_NS)
        if code_el is None or code_el.get("code") != PACKAGE_LABEL_SECTION_CODE:
            continue
        for om in section.iterfind(".//v3:observationMedia", SPL_NS):
            ref = om.find("./v3:value/v3:reference", SPL_NS)
            if ref is not None and ref.get("value"):
                refs.append(ref.get("value"))
    return refs

def parse_spl_bytes(xml_bytes, rx_or_otc):
    from lxml import etree
    root = etree.fromstring(xml_bytes)
    document_image_refs = _document_image_refs(root)
    setid_el = root.find(".//v3:setId", SPL_NS)
    setid = setid_el.get("root") if setid_el is not None else None
    records = []
    for product in root.iterfind(".//v3:manufacturedProduct", SPL_NS):
        form_code_el = product.find("./v3:formCode", SPL_NS)
        form_code = form_code_el.get("code") if form_code_el is not None else None
        if form_code not in ORAL_SOLID_DOSAGE_FORM_CODES:
            continue
        ndc_codes = sorted({
            c.get("code") for c in product.iterfind(".//v3:code", SPL_NS)
            if c.get("code") and "-" in c.get("code")
        })
        if not ndc_codes:
            continue
        name = _text(product.find(".//v3:name", SPL_NS))
        attrs = {"SPLCOLOR": [], "SPLIMPRINT": [], "SPLSHAPE": [], "SPLSCORE": [], "SPLSIZE": [], "SPLIMAGE": []}
        for characteristic in product.iterfind(".//v3:subjectOf/v3:characteristic", SPL_NS):
            code_el = characteristic.find("./v3:code", SPL_NS)
            if code_el is None:
                continue
            ctype = code_el.get("code")
            if ctype not in attrs:
                continue
            if ctype == "SPLIMPRINT":
                text = _text(characteristic.find("./v3:value", SPL_NS))
                if text:
                    attrs[ctype].append(text)
            elif ctype == "SPLIMAGE":
                ref = characteristic.find(".//v3:reference", SPL_NS)
                if ref is not None and ref.get("value"):
                    attrs[ctype].extend(ref.get("value").split())
            else:
                value_el = characteristic.find("./v3:value", SPL_NS)
                if value_el is not None:
                    v = value_el.get("displayName") or value_el.get("code") or value_el.get("value")
                    if v:
                        attrs[ctype].append(v)
        for ndc in ndc_codes:
            records.append(SplPillRecord(
                ndc=ndc, setid=setid, name=name, rx_or_otc=rx_or_otc,
                imprint=";".join(attrs["SPLIMPRINT"]) or None,
                color=",".join(attrs["SPLCOLOR"]) or None,
                shape=attrs["SPLSHAPE"][0] if attrs["SPLSHAPE"] else None,
                score_marks=attrs["SPLSCORE"][0] if attrs["SPLSCORE"] else None,
                size_mm=attrs["SPLSIZE"][0] if attrs["SPLSIZE"] else None,
                image_refs=list(dict.fromkeys(attrs["SPLIMAGE"] + document_image_refs)),
            ))
    return records

print("Parser defined.")


# ============================================================================
# 3. Download + process each part
# ============================================================================
import csv as _csv

def load_state():
    return json.load(open(STATE_PATH)) if STATE_PATH.exists() else {}

def save_state(state):
    json.dump(state, open(STATE_PATH, "w"), indent=2)

def append_manifest_rows(rows):
    file_exists = MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["path", "label", "side", "domain", "source", "category"])
        if not file_exists:
            w.writeheader()
        w.writerows(rows)

def merge_metadata(records):
    existing = json.load(open(METADATA_PATH)) if METADATA_PATH.exists() else {}
    for rec in records:
        existing[rec.ndc] = {
            "name": rec.name, "imprint": rec.imprint, "color": rec.color,
            "shape": rec.shape, "score_marks": rec.score_marks,
            "size_mm": rec.size_mm, "rx_or_otc": rec.rx_or_otc,
        }
    json.dump(existing, open(METADATA_PATH, "w"), indent=0)

MAX_IMAGE_DIM = 640  # plenty for a DINOv2-style model that resizes to ~224-518px anyway

def _save_downscaled(raw_bytes, out_path):
    """Real package-label photos from DailyMed run 1-3MB+ each at original
    resolution -- confirmed live: ~44k of them exhausted a Kaggle CPU
    session's disk before even half the requested parts finished, hanging
    for 12h on the next download (writes silently stalling once disk was
    full) until the notebook-level timeout killed the whole run with
    nothing more saved. Downscaling + recompressing on save (JPEG q85,
    max dimension 640px) cuts each image to roughly 20-80KB -- the model
    resizes to ~224-518px for its own input anyway, so nothing informative
    is lost, and 12+ parts now fit comfortably in the session's disk."""
    try:
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return  # not a real image (corrupt/unexpected format) -- skip, don't crash the part
    img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
    img.save(out_path, "JPEG", quality=85)


def _free_disk_bytes():
    return shutil.disk_usage(WORK_DIR).free


def download_to(url, dest_path, retries=3):
    last_err = None
    for attempt in range(retries):
        try:
            if url.startswith("ftp://"):
                urllib.request.urlretrieve(url, dest_path)
            else:
                resp = requests.get(url, stream=True, timeout=120)
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            return
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"  download attempt {attempt+1}/{retries} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise last_err

def process_part(part, url, rx_or_otc):
    print(f"\n=== {part} ===")
    tmp_zip_path = WORK_DIR / f"{part}.zip"
    print(f"downloading {url} ...")
    download_to(url, tmp_zip_path)

    all_rows = []
    all_records = []
    with zipfile.ZipFile(tmp_zip_path) as zf:
        doc_zip_names = [n for n in zf.namelist() if n.lower().endswith(".zip")]
        total = len(doc_zip_names) if MAX_DOCS_PER_PART is None else min(len(doc_zip_names), MAX_DOCS_PER_PART)
        print(f"  {len(doc_zip_names)} nested per-document zips found; processing {total}")

        n_processed = 0
        for doc_zip_name in doc_zip_names:
            if MAX_DOCS_PER_PART and n_processed >= MAX_DOCS_PER_PART:
                break
            try:
                nested_bytes = zf.read(doc_zip_name)
                with zipfile.ZipFile(BytesIO(nested_bytes)) as nested_zf:
                    nested_names = nested_zf.namelist()
                    xml_names = [n for n in nested_names if n.lower().endswith(".xml")]
                    for xml_name in xml_names:
                        records = parse_spl_bytes(nested_zf.read(xml_name), rx_or_otc=rx_or_otc)
                        for rec in records:
                            local_paths = []
                            for image_ref in rec.image_refs:
                                matches = [n for n in nested_names if n.split("/")[-1] == image_ref]
                                if matches:
                                    # Force .jpg regardless of original extension -- these are
                                    # downscaled+recompressed below, never written as raw bytes.
                                    out_path = SPL_IMAGE_DIR / f"{rec.ndc}_{Path(image_ref).stem}.jpg"
                                    if not out_path.exists():
                                        _save_downscaled(nested_zf.read(matches[0]), out_path)
                                    local_paths.append(str(out_path))
                            all_records.append(rec)
                            for path in local_paths:
                                all_rows.append({
                                    "path": path, "label": rec.ndc, "side": "unknown",
                                    "domain": "reference_pool", "source": "dailymed",
                                    "category": rec.rx_or_otc,
                                })
            except Exception:
                pass  # one bad nested zip/XML shouldn't kill the whole part
            n_processed += 1
            if n_processed % 1000 == 0:
                print(f"  ...{n_processed}/{total} processed, {len(all_rows)} images resolved so far")

    append_manifest_rows(all_rows)
    merge_metadata(all_records)
    tmp_zip_path.unlink(missing_ok=True)
    print(f"  done: {n_processed} documents, {len(all_rows)} images resolved, "
          f"{len(all_records)} pill records (metadata merged even without an image)")
    return len(all_rows)

state = load_state()
for part in SPL_PARTS_TO_FETCH:
    if state.get(part) == "done":
        print(f"skipping {part} (already completed in a prior run)")
        continue
    if part not in spl_zip_urls:
        print(f"skipping {part} (URL not found on the resources page)")
        continue
    free_bytes = _free_disk_bytes()
    if free_bytes < MIN_FREE_DISK_BYTES:
        print(f"Stopping before {part}: only {free_bytes / 1024**3:.1f}GB free "
              f"(below the {MIN_FREE_DISK_BYTES / 1024**3:.0f}GB safety margin) -- "
              f"better to stop cleanly here with what's already downloaded than risk "
              f"a mid-download hang. Re-run this notebook later to pick up the rest "
              f"(already-'done' parts are skipped via acquisition_state.json).")
        break
    rx_or_otc = "RX" if "rx" in part else "OTC"
    try:
        process_part(part, spl_zip_urls[part], rx_or_otc)
        state[part] = "done"
    except Exception as e:
        print(f"FAILED on {part}: {e}")
        state[part] = f"failed: {e}"
    save_state(state)
    print(f"  free disk after {part}: {_free_disk_bytes() / 1024**3:.1f}GB")

print("\nAll requested parts processed (or skipped/failed as logged above).")


# ============================================================================
# 4. Summary
# ============================================================================
if MANIFEST_PATH.exists():
    with open(MANIFEST_PATH) as f:
        rows = list(_csv.DictReader(f))
    print(f"Total manifest rows: {len(rows)}")
    from collections import Counter
    print("By category:", dict(Counter(r["category"] for r in rows)))
    print("By part (source is always 'dailymed'; check acquisition_state.json for per-part status)")
else:
    print("No manifest written yet — check the per-part logs above for errors.")

print("\nState:", json.load(open(STATE_PATH)) if STATE_PATH.exists() else "no state file")
print("\nNext step: commit this notebook (Save & Run All), then in the training")
print("notebook, Add Input -> Notebook -> this notebook, to read its output.")
