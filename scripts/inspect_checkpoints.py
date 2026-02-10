#!/usr/bin/env python3
"""Checkpoint inspection script for viewing and filtering training checkpoints."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokentango.train import list_checkpoints
from tokentango.config import Checkpoint


def format_checkpoint(checkpoint: Checkpoint, path: str = None) -> str:
    """Format checkpoint information for display."""
    lines = [
        f"Run: {checkpoint.config.run_name or 'N/A'}",
        f"  Path: {path or 'N/A'}",
        f"  Epoch: {checkpoint.epoch}",
        f"  Accuracy: {checkpoint.accuracy:.2f}%",
        f"  Timestamp: {checkpoint.timestamp}",
        f"  Config:",
        f"    - train_frac: {checkpoint.config.train_frac}",
        f"    - optimizer: {checkpoint.config.optimizer_type}",
        f"    - use_mlm: {checkpoint.config.use_mlm}",
        f"    - batch_size: {checkpoint.config.batch_size}",
        f"    - lr: {checkpoint.config.lr}",
        f"    - seed: {checkpoint.config.seed}",
    ]

    if checkpoint.cls_losses:
        lines.append(
            f"  Loss samples: {len(checkpoint.cls_losses)} cls, {len(checkpoint.mlm_losses)} mlm"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect training checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  inspect_checkpoints                          # List all checkpoints
  inspect_checkpoints --sort accuracy          # Sort by accuracy (highest first)
  inspect_checkpoints --optimizer adamw        # Filter by optimizer
  inspect_checkpoints --use-mlm false          # Filter by MLM setting
  inspect_checkpoints --min-accuracy 80        # Only show checkpoints >= 80%
  inspect_checkpoints --run-name happy-panda   # Filter by run name
        """,
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="data/checkpoints",
        help="Directory containing checkpoints (default: data/checkpoints)",
    )
    parser.add_argument(
        "--sort",
        choices=["accuracy", "timestamp"],
        default="timestamp",
        help="Sort checkpoints by field (default: timestamp)",
    )
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw", "sgd"], help="Filter by optimizer type"
    )
    parser.add_argument(
        "--use-mlm", choices=["true", "false"], help="Filter by MLM usage"
    )
    parser.add_argument("--min-accuracy", type=float, help="Minimum accuracy threshold")
    parser.add_argument("--run-name", help="Filter by run name (partial match)")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of checkpoints to display (default: 20)",
    )
    parser.add_argument(
        "--losses", action="store_true", help="Show detailed loss information"
    )

    args = parser.parse_args()

    # Load all checkpoints
    print(f"Loading checkpoints from {args.checkpoint_dir}...")
    checkpoints = list_checkpoints(args.checkpoint_dir)

    if not checkpoints:
        print("No checkpoints found.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s)\n")

    # Apply filters
    filtered = checkpoints

    if args.optimizer:
        filtered = [c for c in filtered if c.config.optimizer_type == args.optimizer]

    if args.use_mlm is not None:
        use_mlm = args.use_mlm == "true"
        filtered = [c for c in filtered if c.config.use_mlm == use_mlm]

    if args.min_accuracy is not None:
        filtered = [c for c in filtered if c.accuracy >= args.min_accuracy]

    if args.run_name:
        filtered = [
            c
            for c in filtered
            if args.run_name.lower() in (c.config.run_name or "").lower()
        ]

    # Sort
    if args.sort == "accuracy":
        filtered = sorted(filtered, key=lambda c: c.accuracy, reverse=True)
    else:  # timestamp
        filtered = sorted(filtered, key=lambda c: c.timestamp, reverse=True)

    # Limit
    filtered = filtered[: args.limit]

    # Display
    if not filtered:
        print("No checkpoints match the specified filters.")
        return

    print(f"Showing {len(filtered)} checkpoint(s):\n")
    print("=" * 60)

    for i, checkpoint in enumerate(filtered, 1):
        print(f"\n[{i}/{len(filtered)}]")
        print(
            format_checkpoint(checkpoint, getattr(checkpoint, "checkpoint_path", None))
        )

        if args.losses and checkpoint.cls_losses:
            print(f"\n  Loss history (last 5):")
            for j in range(min(5, len(checkpoint.cls_losses))):
                cls_loss = checkpoint.cls_losses[-(j + 1)]
                mlm_loss = (
                    checkpoint.mlm_losses[-(j + 1)] if checkpoint.mlm_losses else 0
                )
                print(f"    - cls: {cls_loss:.4f}, mlm: {mlm_loss:.4f}")

        print("\n" + "=" * 60)

    # Summary statistics
    if len(filtered) > 1:
        accuracies = [c.accuracy for c in filtered]
        print(f"\nSummary:")
        print(f"  Best accuracy: {max(accuracies):.2f}%")
        print(f"  Worst accuracy: {min(accuracies):.2f}%")
        print(f"  Average accuracy: {sum(accuracies) / len(accuracies):.2f}%")


if __name__ == "__main__":
    main()
