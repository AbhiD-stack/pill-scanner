"""Resolve seed-list drug names into full metadata (RxCUI, NDCs, imprint, color,
shape, score marks) via the RxNav REST API.

Unlike pill-id's build_ndc_names.py (which starts from NDCs already present in
a trained model's reference set), this starts from generic drug *names* in
data/seed_lists/*.csv, since the class list itself doesn't exist yet:

    name -> rxcui (rxnav /rxcui.json)
         -> all associated NDCs (rxnav /rxcui/{rxcui}/ndcs.json)
         -> per-NDC properties: imprint, color, shape, score marks
            (rxnav /ndcproperties.json)

Run this from a Kaggle notebook (or any environment with real internet -
RxNav is blocked from the coding sandbox this was authored in). Resumable:
rerun to retry NDCs that were rate-limited.

    python3 scripts/build_drug_metadata.py --seed data/seed_lists/top_500_seed.csv --out data/metadata/rx_metadata.json
    python3 scripts/build_drug_metadata.py --seed data/seed_lists/top_otc_seed.csv --out data/metadata/otc_metadata.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://rxnav.nlm.nih.gov/REST"


class RateLimited(Exception):
    pass


def _get(url: str) -> dict:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return json.load(resp)
        except Exception:
            time.sleep(2**attempt)  # 1,2,4,8,16s
    raise RateLimited(url)


def _rxcuis_for_name(name: str) -> list[str]:
    data = _get(f"{BASE}/rxcui.json?name={urllib.parse.quote(name)}&search=2")
    return data.get("idGroup", {}).get("rxnormId", []) or []


def _ndcs_for_rxcui(rxcui: str) -> list[str]:
    data = _get(f"{BASE}/rxcui/{rxcui}/ndcs.json")
    return data.get("ndcGroup", {}).get("ndcList", {}).get("ndc", []) or []


def _properties_for_ndc(ndc: str) -> dict | None:
    data = _get(f"{BASE}/ndcproperties.json?id={ndc}&ndcstatus=ALL")
    pl = data.get("ndcPropertyList", {}).get("ndcProperty", [])
    if not pl:
        return None
    p = pl[0]
    props = {
        x["propName"]: x["propValue"]
        for x in p.get("propertyConceptList", {}).get("propertyConcept", [])
    }
    return {
        "ndc": ndc,
        "rxcui": p.get("rxcui"),
        "imprint": props.get("IMPRINT_CODE") or None,
        "color": props.get("COLORTEXT") or None,
        "shape": props.get("SPLSHAPE") or None,
        "score_marks": props.get("SPLSCORE") or None,
        "size_mm": props.get("SPLSIZE") or None,
        "status": props.get("NDC_STATUS") or None,
    }


def resolve_one(name: str) -> tuple[str, list[dict]]:
    entries: list[dict] = []
    try:
        for rxcui in _rxcuis_for_name(name):
            for ndc in _ndcs_for_rxcui(rxcui):
                try:
                    props = _properties_for_ndc(ndc)
                except RateLimited:
                    continue
                if props:
                    entries.append(props)
    except RateLimited:
        pass
    return name, entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, type=Path, help="Seed CSV with a generic_name column")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    with open(args.seed) as f:
        names = [row["generic_name"] for row in csv.DictReader(f)]

    table: dict[str, list[dict]] = json.load(open(args.out)) if args.out.exists() else {}
    todo = [n for n in names if n not in table]
    print(f"{len(names)} seed names; {len(table)} already resolved; {len(todo)} to fetch")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(resolve_one, n) for n in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            name, entries = fut.result()
            table[name] = entries
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} processed")
                args.out.parent.mkdir(parents=True, exist_ok=True)
                json.dump(table, open(args.out, "w"), indent=0, sort_keys=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(args.out, "w"), indent=0, sort_keys=True)
    total_ndcs = sum(len(v) for v in table.values())
    print(f"Wrote {len(table)} names / {total_ndcs} NDC entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
