#!/usr/bin/env python3
"""
Support Ticket Auto-Tagger — Main Runner
Usage:
    python main.py --mode single  --ticket "My app crashes on startup"
    python main.py --mode batch   --max 10
    python main.py --mode compare --ticket "I can't login and need my data exported"
"""

import argparse
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.tagger import SupportTicketTagger

BANNER = """
╔══════════════════════════════════════════════════════╗
║       🏷️  Support Ticket Auto-Tagger                 ║
║       Zero-Shot vs Few-Shot LLM Classification       ║
╚══════════════════════════════════════════════════════╝
"""


def load_tickets(path: str = "data/tickets.json"):
    with open(path) as f:
        return json.load(f)


def print_result(result: dict, mode: str = "both"):
    print(f"\n📋 Ticket: \"{result.get('ticket', '')[:80]}...\"")
    if result.get("true_tags"):
        print(f"   ✅ True Tags: {result['true_tags']}")

    if mode in ("zero", "both") and result.get("zero_shot"):
        z = result["zero_shot"]
        print(f"\n   🔹 ZERO-SHOT ({z['latency_s']}s)")
        for tag, conf in zip(z["tags"], z["confidence"]):
            bar = "█" * int(conf * 20)
            print(f"      {tag:<30} {bar} {conf:.0%}")
        print(f"      💬 {z['reasoning']}")

    if mode in ("few", "both") and result.get("few_shot"):
        f = result["few_shot"]
        print(f"\n   🔸 FEW-SHOT ({f['latency_s']}s)")
        for tag, conf in zip(f["tags"], f["confidence"]):
            bar = "█" * int(conf * 20)
            print(f"      {tag:<30} {bar} {conf:.0%}")
        print(f"      💬 {f['reasoning']}")

    if result.get("evaluation"):
        ev = result["evaluation"]
        print(f"\n   📊 Evaluation:")
        print(f"      Zero-Shot Top-1 Correct : {'✅' if ev['zero_shot_top1_correct'] else '❌'}")
        print(f"      Few-Shot  Top-1 Correct : {'✅' if ev['few_shot_top1_correct'] else '❌'}")
        print(f"      Zero-Shot Top-3 Overlap : {ev['zero_shot_top3_overlap']}/3")
        print(f"      Few-Shot  Top-3 Overlap : {ev['few_shot_top3_overlap']}/3")


def print_summary(summary: dict):
    print("\n" + "═" * 56)
    print("  📈 BATCH EVALUATION SUMMARY")
    print("═" * 56)

    z = summary["zero_shot"]
    f = summary["few_shot"]

    print(f"  Tickets evaluated     : {summary['total_tickets']}")
    print(f"\n  {'Metric':<28} {'Zero-Shot':>10} {'Few-Shot':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Top-1 Accuracy':<28} {z['top1_accuracy']:>9.1%} {f['top1_accuracy']:>9.1%}")
    print(f"  {'Avg Top-3 Overlap (of 3)':<28} {z['avg_top3_overlap']:>10} {f['avg_top3_overlap']:>10}")
    print(f"  {'Avg Latency (s)':<28} {z['avg_latency_s']:>10} {f['avg_latency_s']:>10}")

    winner = "Few-Shot 🏆" if f["top1_accuracy"] >= z["top1_accuracy"] else "Zero-Shot 🏆"
    print(f"\n  Winner by Top-1 Accuracy: {winner}")
    print("═" * 56)


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="Support Ticket Auto-Tagger")
    parser.add_argument("--mode", choices=["single", "batch", "compare"], default="compare")
    parser.add_argument("--ticket", type=str, default=None)
    parser.add_argument("--max", type=int, default=10)
    parser.add_argument("--strategy", choices=["zero", "few", "both"], default="both")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("❌  OPENROUTER_API_KEY not set.")
        print("    Add it to your .env file: OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    tagger = SupportTicketTagger(api_key=api_key)

    if args.mode == "single":
        text = args.ticket or "My app keeps crashing every time I try to open it."
        print(f"\n🎯 Mode: Single Ticket | Strategy: {args.strategy}")

        if args.strategy == "zero":
            result = tagger.zero_shot_tag(text)
            print_result({"ticket": text, "zero_shot": result})
        elif args.strategy == "few":
            result = tagger.few_shot_tag(text)
            print_result({"ticket": text, "few_shot": result})
        else:
            result = tagger.compare(text)
            print_result(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")

    elif args.mode == "compare":
        text = args.ticket or "I was charged twice and I can't log in to get a refund."
        print(f"\n⚖️  Mode: Compare Zero-Shot vs Few-Shot")
        result = tagger.compare(text)
        print_result(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)

    elif args.mode == "batch":
        print(f"\n📦 Mode: Batch Evaluation (max {args.max} tickets)")
        tickets = load_tickets()
        eval_result = tagger.evaluate_dataset(tickets, max_tickets=args.max)

        for r in eval_result["results"]:
            print_result(r)
            print()

        print_summary(eval_result["summary"])

        output_path = args.output or "results/batch_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(eval_result, f, indent=2)
        print(f"\n💾 Full results saved to {output_path}")


if __name__ == "__main__":
    main()