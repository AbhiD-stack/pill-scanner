"""Multi-axis accuracy gate: run this after every training run, BEFORE any
model artifact is exported to the app. Nothing gets deployed on the strength
of a single top-k number again.

Input contract (kept deliberately simple/model-agnostic so this works with
whatever backbone the Kaggle notebook trains): a query set of
(embedding, true_label, domain, side, images_in_class) rows and a reference
gallery of (embedding, label). "domain" is "reference" (studio photo) or
"consumer" (phone/real-world photo) — the two must be reported separately,
never blended into one headline number, because blending is exactly how the
ePillID-only baseline's offline metrics looked great while the deployed app
did not.

Usage:
    python3 eval/evaluate.py --query query_embeddings.pt --gallery ref_embeddings.pt --out report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

TOPKS = (1, 5, 10, 20, 50)
CONFIDENCE_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95


@dataclass
class QueryRow:
    embedding: torch.Tensor
    true_label: str
    domain: str  # "reference" | "reference_pool" | "consumer"
    side: str  # "front" | "back" | "unknown"
    images_in_class: int  # how many reference images exist for true_label
    category: str = "unknown"  # "RX" | "OTC" | "unknown"
    tier: str = "unknown"  # "priority" (top-500 RX/OTC) | "long_tail" | "unknown"


def _topk_hit(true_label: str, ranked_labels: list[str], k: int) -> bool:
    return true_label in ranked_labels[:k]


def _rank_chunk(
    query_emb_n: torch.Tensor, gallery_emb_n: torch.Tensor, gallery_labels: list[str]
) -> list[tuple[list[str], float]]:
    """Ranks a whole chunk of (already-normalized) query embeddings against
    the (already-normalized) gallery in one batched matmul, returning
    (deduped ranked labels, top1 score) per row. Kept chunked rather than
    doing the whole query set in one matmul — with a 180k+-image gallery, a
    single (n_queries x gallery_size) similarity matrix would be tens of GB;
    a few hundred rows at a time keeps memory bounded."""
    sims_chunk = query_emb_n @ gallery_emb_n.T
    order_chunk = torch.argsort(sims_chunk, dim=1, descending=True)
    top1_chunk = sims_chunk.max(dim=1).values
    out = []
    for i in range(order_chunk.shape[0]):
        seen = set()
        ranked: list[str] = []
        for idx in order_chunk[i].tolist():
            lbl = gallery_labels[idx]
            if lbl not in seen:
                seen.add(lbl)
                ranked.append(lbl)
        out.append((ranked, float(top1_chunk[i])))
    return out


def _rank_all_rows(
    rows: list[QueryRow], gallery_emb: torch.Tensor, gallery_labels: list[str], chunk_size: int = 256
) -> list[tuple[list[str], float, bool]]:
    """Ranks every row exactly once (normalizing the gallery exactly once),
    instead of the naive approach of calling a per-row rank function once
    per report axis (overall/domain/category/tier/tier_and_category/
    images_per_class/worst_classes = 7x) -- each of which used to
    re-normalize the *entire* gallery from scratch. Confirmed via a live run
    to matter a lot at this scale: a ~180k-image gallery normalized ~7x per
    test row, for tens of thousands of test rows, is billions of wasted
    floating-point ops. Returns (ranked_labels, top1_score, top1_correct)
    per row, reused by every axis below."""
    if not rows:
        return []
    gallery_emb_n = F.normalize(gallery_emb, dim=1)
    query_emb_n = F.normalize(torch.stack([r.embedding for r in rows]), dim=1)
    precomputed: list[tuple[list[str], float, bool]] = []
    for start in range(0, len(rows), chunk_size):
        chunk_result = _rank_chunk(query_emb_n[start:start + chunk_size], gallery_emb_n, gallery_labels)
        for local_i, (ranked, top1_score) in enumerate(chunk_result):
            row = rows[start + local_i]
            correct = bool(ranked and ranked[0] == row.true_label)
            precomputed.append((ranked, top1_score, correct))
    return precomputed


def _accuracy_block(rows: list[QueryRow], rankings: list[tuple[list[str], float, bool]]) -> dict:
    """rankings must be the precomputed (ranked_labels, top1_score, correct)
    tuples for these exact rows, in the same order (see _rank_all_rows)."""
    hits = {k: 0 for k in TOPKS}
    top1_scores: list[float] = []
    correct_at_top1: list[bool] = []
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

    # Confidence calibration: at each threshold, what fraction of queries would
    # we answer (coverage) and how accurate are we on those we do answer?
    calibration = []
    for t in CONFIDENCE_THRESHOLDS:
        answered = [c for s, c in zip(top1_scores, correct_at_top1) if s >= t]
        coverage = len(answered) / n
        precision = (sum(answered) / len(answered)) if answered else None
        calibration.append({"threshold": t, "coverage": round(coverage, 4), "precision_if_answered": precision})
    result["calibration"] = calibration
    return result


def restrict_gallery(
    gallery_emb: torch.Tensor, gallery_labels: list[str], keep_labels: set[str]
) -> tuple[torch.Tensor, list[str]]:
    """Filters the reference gallery down to only `keep_labels` before
    ranking -- simulates an app mode where the search space is deliberately
    narrowed (e.g. "this is a common drug" / user picked RX or OTC) instead
    of always searching the full ~95k-class gallery. Every long-tail class
    removed from the candidate pool is one less thing a priority-tier query
    can be confused with, which is a different (and likely bigger) lever
    than more training time on an unrestricted gallery -- priority-tier RX
    currently competes against ~85,000 long-tail RX classes it would never
    face in a "top-500 common drugs" deployment mode."""
    keep_idx = [i for i, lbl in enumerate(gallery_labels) if lbl in keep_labels]
    if not keep_idx:
        return gallery_emb[:0], []
    idx_t = torch.tensor(keep_idx, dtype=torch.long)
    return gallery_emb[idx_t], [gallery_labels[i] for i in keep_idx]


def evaluate(rows: list[QueryRow], gallery_emb: torch.Tensor, gallery_labels: list[str]) -> dict:
    report: dict = {}

    # Rank every row exactly once, up front; every split below is just a
    # different partition of the same (rows, rankings) pairs.
    rankings = _rank_all_rows(rows, gallery_emb, gallery_labels)

    def _block_for(indices: list[int]) -> dict:
        return _accuracy_block([rows[i] for i in indices], [rankings[i] for i in indices])

    report["overall"] = _block_for(list(range(len(rows))))

    by_domain: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_domain[r.domain].append(i)
    report["by_domain"] = {d: _block_for(idxs) for d, idxs in by_domain.items()}

    # RX vs. OTC, reported separately for the same reason domain is: OTC has
    # historically had ~zero coverage in this project, so blending it into
    # one "overall" number would hide a regression or a persistently-thin
    # OTC accuracy behind a healthy RX-dominated average.
    by_category: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_category[r.category].append(i)
    report["by_category"] = {c: _block_for(idxs) for c, idxs in by_category.items()}

    # The number that actually matters for the "doctors testing the top-500
    # RX/OTC" goal — never blend this into the overall number, which is
    # dominated by however many long-tail classes happen to be in the data.
    by_tier: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_tier[r.tier].append(i)
    report["by_tier"] = {t: _block_for(idxs) for t, idxs in by_tier.items()}

    # The actual granular breakdown requested: "top-500 RX" and "top-500 OTC"
    # as their own distinct numbers, not just tier and category reported
    # separately (which can't tell you the priority-RX number on its own).
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_group[f"{r.tier}_{r.category}"].append(i)
    report["by_tier_and_category"] = {g: _block_for(idxs) for g, idxs in by_group.items()}

    by_depth: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        bucket = "1-2" if r.images_in_class <= 2 else "3-5" if r.images_in_class <= 5 else "6+"
        by_depth[bucket].append(i)
    report["by_images_per_class"] = {b: _block_for(idxs) for b, idxs in by_depth.items()}

    # Per-class worst performers (top-5 miss), so data-collection effort goes
    # to the classes that actually need it instead of guessing.
    per_class: dict[str, list[bool]] = defaultdict(list)
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


def evaluate_deployment_modes(
    rows: list[QueryRow], gallery_emb: torch.Tensor, gallery_labels: list[str]
) -> dict:
    """Simulates two app deployment modes that restrict the search gallery
    instead of always searching the full ~95k-class gallery -- a different,
    likely bigger lever than more training time, since priority-tier RX
    currently competes against ~85,000 long-tail RX classes a "common
    drugs only" mode would never expose it to. Only meaningful for
    priority-tier queries -- a long-tail pill genuinely isn't one of the
    500 common drugs, so restricting the gallery to priority-only would
    just make it unfindable, not more findable.

    Two modes, from weakest to strongest restriction:
      "priority_tier_gallery": app knows "this is a common drug" (e.g. a
        deliberate product mode) but not RX vs OTC -- gallery restricted
        to ALL priority-tier classes (RX + OTC together).
      "priority_tier_and_category_gallery": app additionally knows RX vs
        OTC (user picked it, or a separate RX/OTC classifier decided it) --
        gallery restricted to just that category's priority-tier classes.
        This is the scenario from the "ask RX or OTC, browse the rest if
        unsure" idea.

    Rows without category in {RX, OTC} or tier != "priority" are excluded
    -- this function is specifically about the common-drug deployment
    mode's numbers, not a replacement for the unrestricted full-gallery
    evaluate() results, which remain the honest baseline."""
    priority_rows = [r for r in rows if r.tier == "priority" and r.category in ("RX", "OTC")]
    if not priority_rows:
        return {"priority_tier_gallery": {"n": 0}, "priority_tier_and_category_gallery": {}}

    priority_labels = {r.true_label for r in priority_rows}
    otc_labels = {r.true_label for r in priority_rows if r.category == "OTC"}
    rx_labels = {r.true_label for r in priority_rows if r.category == "RX"}

    report: dict = {}

    # Mode 1: gallery restricted to priority-tier only (RX+OTC together).
    pt_gallery_emb, pt_gallery_labels = restrict_gallery(gallery_emb, gallery_labels, priority_labels)
    rankings = _rank_all_rows(priority_rows, pt_gallery_emb, pt_gallery_labels)
    report["priority_tier_gallery"] = _accuracy_block(priority_rows, rankings)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(priority_rows):
        by_cat[r.category].append(i)
    report["priority_tier_gallery_by_category"] = {
        c: _accuracy_block([priority_rows[i] for i in idxs], [rankings[i] for i in idxs])
        for c, idxs in by_cat.items()
    }

    # Mode 2: gallery restricted to the query's own priority-tier category
    # (RX query -> RX-priority-only gallery, OTC query -> OTC-priority-only
    # gallery) -- the "user tells the app RX or OTC" scenario.
    otc_rows = [r for r in priority_rows if r.category == "OTC"]
    rx_rows = [r for r in priority_rows if r.category == "RX"]
    cat_report = {}
    if otc_rows:
        otc_gallery_emb, otc_gallery_labels = restrict_gallery(gallery_emb, gallery_labels, otc_labels)
        otc_rankings = _rank_all_rows(otc_rows, otc_gallery_emb, otc_gallery_labels)
        cat_report["OTC"] = _accuracy_block(otc_rows, otc_rankings)
    if rx_rows:
        rx_gallery_emb, rx_gallery_labels = restrict_gallery(gallery_emb, gallery_labels, rx_labels)
        rx_rankings = _rank_all_rows(rx_rows, rx_gallery_emb, rx_gallery_labels)
        cat_report["RX"] = _accuracy_block(rx_rows, rx_rankings)
    report["priority_tier_and_category_gallery"] = cat_report

    return report


def build_dual_side_query_rows(rows: list[QueryRow], max_pairs_per_label: int = 3) -> list[QueryRow]:
    """Simulates the app's planned "scan both sides" feature: fuses a
    front-side query embedding with a back-side query embedding of the
    same class into ONE combined query (normalize, sum, renormalize),
    instead of evaluating single images only. Only classes with at least
    one row tagged side="front" AND one tagged side="back" in this query
    set can be paired.

    This is currently real for ePillID and C3PI (both tag genuine
    front/back), but NOT for DailyMed -- SPL package-label photos aren't
    modeled as front/back, so every DailyMed row (all of this project's
    OTC volume) is side="unknown" and can never be paired here. Callers
    MUST report n_classes_pairable/n_pairs alongside any dual-side
    accuracy number, so a good RX number can't be misread as "this is what
    scanning both sides does for OTC too" when there's currently no OTC
    coverage to measure that against at all.
    """
    by_label_side: dict[str, dict[str, list[QueryRow]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_label_side[r.true_label][r.side].append(r)

    fused_rows: list[QueryRow] = []
    for by_side in by_label_side.values():
        fronts = by_side.get("front", [])
        backs = by_side.get("back", [])
        n_pairs = min(len(fronts), len(backs), max_pairs_per_label)
        for i in range(n_pairs):
            f_emb = F.normalize(fronts[i].embedding, dim=0)
            b_emb = F.normalize(backs[i].embedding, dim=0)
            fused_emb = F.normalize(f_emb + b_emb, dim=0)
            src = fronts[i]
            fused_rows.append(QueryRow(
                embedding=fused_emb, true_label=src.true_label, domain=src.domain,
                side="dual_front_back", images_in_class=src.images_in_class,
                category=src.category, tier=src.tier,
            ))
    return fused_rows


def load_query_rows(path: Path) -> list[QueryRow]:
    """Expects a torch-saved list of dicts with keys: embedding, true_label,
    domain, side, images_in_class. Adapt this loader to whatever format the
    training notebook actually dumps."""
    raw = torch.load(path, map_location="cpu")
    return [
        QueryRow(
            embedding=r["embedding"],
            true_label=r["true_label"],
            domain=r.get("domain", "reference"),
            side=r.get("side", "unknown"),
            images_in_class=int(r.get("images_in_class", 1)),
        )
        for r in raw
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, type=Path)
    ap.add_argument("--gallery", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--min-top5-consumer",
        type=float,
        default=None,
        help="If set, exit with a nonzero status when by_domain.consumer.top5_acc "
        "is below this bar — wire this into CI/notebook so a regression can't "
        "silently ship to the app.",
    )
    ap.add_argument(
        "--min-top5-priority",
        type=float,
        default=None,
        help="If set, exit with a nonzero status when by_tier.priority.top5_acc "
        "(the top-500 RX/OTC tier) is below this bar — this is the number that "
        "actually matters for doctor testing, separate from the long tail.",
    )
    args = ap.parse_args()

    rows = load_query_rows(args.query)
    gallery = torch.load(args.gallery, map_location="cpu")
    gallery_emb, gallery_labels = gallery["embeddings"], list(gallery["labels"])

    report = evaluate(rows, gallery_emb, gallery_labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps(report["overall"], indent=2))
    print("\nBy domain:")
    for domain, block in report["by_domain"].items():
        print(f"  {domain}: n={block.get('n')} top5={block.get('top5_acc')} top10={block.get('top10_acc')}")
    print("\nBy category (RX/OTC):")
    for category, block in report["by_category"].items():
        print(f"  {category}: n={block.get('n')} top5={block.get('top5_acc')} top10={block.get('top10_acc')}")
        if category == "OTC" and block.get("n", 0) == 0:
            print("    ^ zero OTC queries in this eval run — OTC accuracy is unknown, not zero.")
    print("\nBy tier (priority = top-500 RX/OTC — the number that actually matters for the doctor-testing goal):")
    for tier, block in report["by_tier"].items():
        print(f"  {tier}: n={block.get('n')} top5={block.get('top5_acc')} top10={block.get('top10_acc')}")
    print("\nBy tier x category (the actual granular goal numbers — top-500 RX and top-500 OTC separately):")
    for group, block in sorted(report["by_tier_and_category"].items()):
        print(f"  {group}: n={block.get('n')} top5={block.get('top5_acc')} top10={block.get('top10_acc')}")
        if group == "priority_OTC" and block.get("n", 0) == 0:
            print("    ^ zero priority-tier OTC queries — the top-500 OTC number is unknown, not zero.")

    # Restricted-gallery deployment modes -- "common drugs only" and
    # "common drugs + known RX/OTC" search spaces, tested against the SAME
    # embeddings already computed for the unrestricted gate above (no new
    # training needed). See evaluate_deployment_modes()'s docstring.
    deploy_report = evaluate_deployment_modes(rows, gallery_emb, gallery_labels)
    report["deployment_modes"] = deploy_report
    print("\nRestricted-gallery deployment modes (same embeddings, smaller search space):")
    pt = deploy_report.get("priority_tier_gallery", {})
    print(f"  priority_tier_gallery (gallery = priority-tier only, RX+OTC): "
          f"n={pt.get('n')} top5={pt.get('top5_acc')} top10={pt.get('top10_acc')}")
    for cat, block in deploy_report.get("priority_tier_gallery_by_category", {}).items():
        print(f"    {cat}: n={block.get('n')} top5={block.get('top5_acc')} top10={block.get('top10_acc')}")
    print("  priority_tier_and_category_gallery (gallery = just that category's priority-tier classes):")
    for cat, block in deploy_report.get("priority_tier_and_category_gallery", {}).items():
        print(f"    {cat}: n={block.get('n')} top5={block.get('top5_acc')} top10={block.get('top10_acc')}")
        unrestricted = report["by_tier_and_category"].get(f"priority_{cat}", {})
        if unrestricted.get("top10_acc") is not None and block.get("top10_acc") is not None:
            delta = block["top10_acc"] - unrestricted["top10_acc"]
            print(f"      vs unrestricted full-gallery top10={unrestricted.get('top10_acc')} (delta={delta:+.3f})")

    # Dual-side ("scan both sides") simulation — fuses a front+back query
    # pair into one combined embedding instead of evaluating single images
    # only, since that's the actual planned app feature. Only real for
    # classes with genuine front AND back tags in this query set (ePillID/
    # C3PI right now, not DailyMed/OTC) — report coverage explicitly so a
    # good RX number here is never misread as an OTC number too.
    dual_rows = build_dual_side_query_rows(rows)
    print(f"\nDual-side (scan-both-sides) simulation: {len(dual_rows)} fused query pairs "
          f"across {len({r.true_label for r in dual_rows})} distinct classes with real "
          f"front+back data available (currently ePillID/C3PI only — DailyMed/OTC rows "
          f"are all side='unknown' and can't be paired here yet).")
    dual_report = None
    if dual_rows:
        dual_report = evaluate(dual_rows, gallery_emb, gallery_labels)
        report["dual_side"] = dual_report
        print(f"  overall: n={dual_report['overall'].get('n')} "
              f"top5={dual_report['overall'].get('top5_acc')} top10={dual_report['overall'].get('top10_acc')}")
        for group, block in sorted(dual_report.get("by_tier_and_category", {}).items()):
            print(f"  {group}: n={block.get('n')} top5={block.get('top5_acc')} top10={block.get('top10_acc')}")
        single_rx = report["by_category"].get("RX", {}).get("top5_acc")
        dual_rx = dual_report["by_category"].get("RX", {}).get("top5_acc") if "RX" in dual_report.get("by_category", {}) else None
        if single_rx is not None and dual_rx is not None:
            print(f"  RX single-image top5={single_rx} vs RX dual-side top5={dual_rx} "
                  f"(delta={dual_rx - single_rx:+.3f})")
    else:
        print("  No classes had both a front-tagged and back-tagged query row — "
              "dual-side accuracy is unmeasured, not zero.")
    json.dump(report, open(args.out, "w"), indent=2, default=str)

    failed = False
    if args.min_top5_consumer is not None:
        consumer = report["by_domain"].get("consumer", {})
        acc = consumer.get("top5_acc")
        if acc is None or acc < args.min_top5_consumer:
            print(f"\nGATE FAILED: consumer-domain top5 ({acc}) below bar ({args.min_top5_consumer}).")
            failed = True
        else:
            print(f"\nGATE PASSED (consumer): top5 ({acc}) meets bar ({args.min_top5_consumer}).")
    if args.min_top5_priority is not None:
        priority = report["by_tier"].get("priority", {})
        acc = priority.get("top5_acc")
        if acc is None or acc < args.min_top5_priority:
            print(f"GATE FAILED: priority-tier (top-500 RX/OTC) top5 ({acc}) below bar ({args.min_top5_priority}).")
            failed = True
        else:
            print(f"GATE PASSED (priority tier): top5 ({acc}) meets bar ({args.min_top5_priority}).")
    if failed:
        print("\nNot exporting this model to the app.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
