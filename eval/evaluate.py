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


def _rank_query(
    query_emb: torch.Tensor, gallery_emb: torch.Tensor, gallery_labels: list[str]
) -> tuple[list[str], torch.Tensor]:
    sims = F.normalize(query_emb, dim=0) @ F.normalize(gallery_emb, dim=1).T
    order = torch.argsort(sims, descending=True)
    seen = set()
    ranked: list[str] = []
    for idx in order.tolist():
        lbl = gallery_labels[idx]
        if lbl not in seen:
            seen.add(lbl)
            ranked.append(lbl)
    return ranked, sims


def _accuracy_block(rows: list[QueryRow], gallery_emb: torch.Tensor, gallery_labels: list[str]) -> dict:
    hits = {k: 0 for k in TOPKS}
    top1_scores: list[float] = []
    correct_at_top1: list[bool] = []
    n = 0
    for row in rows:
        ranked, sims = _rank_query(row.embedding, gallery_emb, gallery_labels)
        if not ranked:
            continue
        n += 1
        for k in TOPKS:
            if _topk_hit(row.true_label, ranked, k):
                hits[k] += 1
        top1_scores.append(float(sims.max()))
        correct_at_top1.append(ranked[0] == row.true_label)

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


def evaluate(rows: list[QueryRow], gallery_emb: torch.Tensor, gallery_labels: list[str]) -> dict:
    report: dict = {}

    report["overall"] = _accuracy_block(rows, gallery_emb, gallery_labels)

    by_domain: dict[str, list[QueryRow]] = defaultdict(list)
    for r in rows:
        by_domain[r.domain].append(r)
    report["by_domain"] = {d: _accuracy_block(rs, gallery_emb, gallery_labels) for d, rs in by_domain.items()}

    # RX vs. OTC, reported separately for the same reason domain is: OTC has
    # historically had ~zero coverage in this project, so blending it into
    # one "overall" number would hide a regression or a persistently-thin
    # OTC accuracy behind a healthy RX-dominated average.
    by_category: dict[str, list[QueryRow]] = defaultdict(list)
    for r in rows:
        by_category[r.category].append(r)
    report["by_category"] = {c: _accuracy_block(rs, gallery_emb, gallery_labels) for c, rs in by_category.items()}

    # The number that actually matters for the "doctors testing the top-500
    # RX/OTC" goal — never blend this into the overall number, which is
    # dominated by however many long-tail classes happen to be in the data.
    by_tier: dict[str, list[QueryRow]] = defaultdict(list)
    for r in rows:
        by_tier[r.tier].append(r)
    report["by_tier"] = {t: _accuracy_block(rs, gallery_emb, gallery_labels) for t, rs in by_tier.items()}

    by_depth: dict[str, list[QueryRow]] = defaultdict(list)
    for r in rows:
        bucket = "1-2" if r.images_in_class <= 2 else "3-5" if r.images_in_class <= 5 else "6+"
        by_depth[bucket].append(r)
    report["by_images_per_class"] = {b: _accuracy_block(rs, gallery_emb, gallery_labels) for b, rs in by_depth.items()}

    # Per-class worst performers (top-5 miss), so data-collection effort goes
    # to the classes that actually need it instead of guessing.
    per_class: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        ranked, _ = _rank_query(r.embedding, gallery_emb, gallery_labels)
        per_class[r.true_label].append(_topk_hit(r.true_label, ranked, 5))
    worst = sorted(
        ((label, sum(hits) / len(hits), len(hits)) for label, hits in per_class.items()),
        key=lambda x: x[1],
    )[:50]
    report["worst_classes_top5"] = [
        {"label": label, "top5_acc": round(acc, 3), "n_queries": n} for label, acc, n in worst
    ]

    return report


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
    json.dump(report, open(args.out, "w"), indent=2, default=str)

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
