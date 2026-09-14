"""Parse DailyMed's bulk SPL (Structured Product Labeling) XML files into
per-NDC pill records: imprint, color, shape, score marks, and referenced
image filenames.

This is a modernized (Python 3, simplified) port of the parsing logic in
HHS's archived pillbox-data-process/scripts/xpath.py — that repo is the
actual source of the `IMAGE_SOURCE`/SPLCOLOR/SPLIMPRINT/SPLSHAPE/SPLSCORE
extraction approach; a newer-looking alternative (pharmaDB/dailymed_data_processor)
was checked and turned out to only handle label text/history, not pill
images or physical characteristics, so it wasn't useful as a base here.

DailyMed's bulk zips split into human RX and human OTC parts (confirmed live
at https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm at
authoring time) — this is the one source in this project that is US-only
AND covers OTC with real images, which is why it exists as its own module
rather than folding into build_drug_metadata.py (which only resolves
metadata for the seed-list names via RxNav, no images).

Simplification vs. the original script: this only handles the common
single-part <manufacturedProduct> case, not the nested <part>/<partProduct>
structure used by multi-part kits. Multi-part products will have their
non-primary parts silently skipped rather than mis-parsed — acceptable for
a first pass since most oral solid dosage SPL documents are single-part, but
worth revisiting if the yield looks low relative to the zip's file count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lxml import etree

NS = {"v3": "urn:hl7-org:v3"}

# Ported verbatim from pillbox-data-process/scripts/xpath.py's `codeChecks` —
# NCI Thesaurus dosage-form codes for oral tablets/capsules, used to filter
# out liquids, injections, patches, etc. that aren't "pills."
ORAL_SOLID_DOSAGE_FORM_CODES = {
    "C25158", "C42895", "C42896", "C42917", "C42902", "C42904", "C42916",
    "C42928", "C42936", "C42954", "C42998", "C42893", "C42897", "C60997",
    "C42905", "C42997", "C42910", "C42927", "C42931", "C42930", "C61004",
    "C61005", "C42964", "C42963", "C42999", "C61006", "C42985", "C42992",
}


@dataclass
class SplPillRecord:
    ndc: str
    setid: Optional[str]
    name: Optional[str]
    rx_or_otc: str  # "RX" | "OTC"
    imprint: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None
    score_marks: Optional[str] = None
    size_mm: Optional[str] = None
    image_refs: list[str] = field(default_factory=list)


def _text(el) -> Optional[str]:
    return el.text.strip() if el is not None and el.text else None


def parse_spl_file(xml_path: str | Path, rx_or_otc: str) -> list[SplPillRecord]:
    tree = etree.parse(str(xml_path))
    return _parse_root(tree.getroot(), rx_or_otc)


def parse_spl_bytes(xml_bytes: bytes, rx_or_otc: str) -> list[SplPillRecord]:
    """Same as parse_spl_file but from in-memory bytes — lets the Kaggle
    notebook read an XML member straight out of a multi-GB DailyMed zip via
    zipfile.read() without extracting the whole archive to disk first."""
    root = etree.fromstring(xml_bytes)
    return _parse_root(root, rx_or_otc)


PACKAGE_LABEL_SECTION_CODE = "51945-4"  # LOINC: "PACKAGE LABEL.PRINCIPAL DISPLAY PANEL"


def _document_image_refs(root) -> list[str]:
    """Real DailyMed documents (confirmed against a live sample — the
    archived pillbox-data-process script's SPLIMAGE-characteristic approach
    does not apply here, that characteristic type simply isn't present)
    reference package-label photos via <observationMedia><value
    mediaType="image/..."><reference value="foo.jpg"/></value></observationMedia>
    blocks that live in a completely separate part of the document (package
    label sections) from the manufacturedProduct/characteristic elements.

    Restricted to <section> elements coded 51945-4 (PACKAGE LABEL.PRINCIPAL
    DISPLAY PANEL) — confirmed necessary via a live full-scale run: RX
    documents in particular embed many other observationMedia images
    (chemical structure diagrams, dosing charts, medication-guide
    illustrations) elsewhere in the document, which an unscoped `.//` search
    also picks up — RX part1 alone produced ~4.4 images per pill record vs.
    ~1.3-1.7 for OTC, a strong signal the unscoped version was pulling in
    unrelated images. Even scoped to the package label panel, these remain
    package/box photos, not always an isolated loose-pill shot.
    """
    refs = []
    for section in root.iterfind(".//v3:section", NS):
        code_el = section.find("./v3:code", NS)
        if code_el is None or code_el.get("code") != PACKAGE_LABEL_SECTION_CODE:
            continue
        for om in section.iterfind(".//v3:observationMedia", NS):
            ref = om.find("./v3:value/v3:reference", NS)
            if ref is not None and ref.get("value"):
                refs.append(ref.get("value"))
    return refs


def _parse_root(root, rx_or_otc: str) -> list[SplPillRecord]:
    # Document-level image list — most SPL documents describe one product
    # (possibly at several package sizes), so attach every image found
    # anywhere in the document to every NDC record produced from it, rather
    # than trying to correlate a specific package section to a specific NDC.
    document_image_refs = _document_image_refs(root)

    setid_el = root.find(".//v3:setId", NS)
    setid = setid_el.get("root") if setid_el is not None else None

    records: list[SplPillRecord] = []

    for product in root.iterfind(".//v3:manufacturedProduct", NS):
        form_code_el = product.find("./v3:formCode", NS)
        form_code = form_code_el.get("code") if form_code_el is not None else None
        if form_code not in ORAL_SOLID_DOSAGE_FORM_CODES:
            continue

        ndc_codes = sorted({
            c.get("code") for c in product.iterfind(".//v3:code", NS)
            if c.get("code") and "-" in c.get("code")  # crude NDC-shaped filter
        })
        if not ndc_codes:
            continue

        name = _text(product.find(".//v3:name", NS))

        attrs: dict[str, list[str]] = {
            "SPLCOLOR": [], "SPLIMPRINT": [], "SPLSHAPE": [],
            "SPLSCORE": [], "SPLSIZE": [], "SPLIMAGE": [],
        }
        for characteristic in product.iterfind(".//v3:subjectOf/v3:characteristic", NS):
            code_el = characteristic.find("./v3:code", NS)
            if code_el is None:
                continue
            ctype = code_el.get("code")
            if ctype not in attrs:
                continue
            if ctype == "SPLIMPRINT":
                value_el = characteristic.find("./v3:value", NS)
                text = _text(value_el)
                if text:
                    attrs[ctype].append(text)
            elif ctype == "SPLIMAGE":
                ref = characteristic.find(".//v3:reference", NS)
                if ref is not None and ref.get("value"):
                    attrs[ctype].extend(ref.get("value").split())
            else:
                value_el = characteristic.find("./v3:value", NS)
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
                # SPLIMAGE-characteristic (attrs["SPLIMAGE"]) kept as a
                # harmless fallback in case some documents do use it, but
                # document_image_refs (from observationMedia) is the one
                # confirmed to actually exist in real data.
                image_refs=list(dict.fromkeys(attrs["SPLIMAGE"] + document_image_refs)),
            ))

    return records
