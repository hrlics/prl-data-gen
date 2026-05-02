"""One-shot script: subsample N unique row_ids from the source genvf dataset and
push the subset to HuggingFace Hub for reuse.

Default: 100 row_ids from
  haoranli-ml/genvf-filtered-proof-graded_score7_only_with_summaries-full
into
  haoranli-ml/genvf-data-generator-100prefix-v1

Usage:
    huggingface-cli login
    python tests/subsample.py
    python tests/subsample.py --n 50 --dst haoranli-ml/genvf-data-generator-50prefix
"""

import argparse
import random

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="haoranli-ml/genvf-filtered-proof-graded_score7_only_with_summaries-full",
    )
    ap.add_argument(
        "--dst",
        default="haoranli-ml/genvf-data-generator-100prefix-v1",
    )
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--id-field", default="row_id")
    ap.add_argument("--src-split", default="train")
    ap.add_argument(
        "--test-size",
        type=int,
        default=10,
        help="Number of rows to mirror into the `test` split (sampled from picked rows).",
    )
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    ds = load_dataset(args.src, split=args.src_split)
    print(f"Loaded {len(ds)} rows from {args.src}:{args.src_split}")

    unique_ids = sorted({row[args.id_field] for row in ds})
    if len(unique_ids) < args.n:
        raise ValueError(
            f"Source has only {len(unique_ids)} unique {args.id_field} values; "
            f"cannot sample {args.n}."
        )

    rng = random.Random(args.seed)
    picked = set(rng.sample(unique_ids, args.n))
    sub = ds.filter(lambda r: r[args.id_field] in picked)
    print(
        f"Selected {len(sub)} rows covering "
        f"{len({r[args.id_field] for r in sub})} unique {args.id_field}s"
    )

    sub.push_to_hub(args.dst, split="train", private=args.private)
    print(f"Pushed train split to https://huggingface.co/datasets/{args.dst}")

    if args.test_size > 0:
        test_size = min(args.test_size, len(sub))
        sub_test = sub.shuffle(seed=args.seed).select(range(test_size))
        sub_test.push_to_hub(args.dst, split="test", private=args.private)
        print(f"Pushed test split ({test_size} rows) to {args.dst}")


if __name__ == "__main__":
    main()
