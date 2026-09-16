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
    print("\nBy tier x category (the actual granular goal numbers — top-500 RX and top-500 OTC separately):")
    for group, block in sorted(report["by_tier_and_category"].items()):
        print(f"  {group}: n={block.get('n')} top5={block.get('top5_acc')} top10={block.get('top10_acc')}")
        if group == "priority_OTC" and block.get("n", 0) == 0:
            print("    ^ zero priority-tier OTC queries — the top-500 OTC number is unknown, not zero.")

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
