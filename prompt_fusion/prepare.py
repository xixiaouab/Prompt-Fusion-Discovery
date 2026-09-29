import argparse
from collections import Counter

from .data import apply_protocol, read_manifest, scan_imagefolder, write_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare image manifests with fixed train/val/test splits.")
    parser.add_argument("--root", required=True, help="ImageFolder root or base directory for input CSV paths")
    parser.add_argument("--manifest", help="Existing CSV; preserves supplied validation and test splits")
    parser.add_argument("--output", required=True, help="Destination CSV")
    parser.add_argument("--protocol", choices=("official", "fgvc", "vtab", "hta"), default="official")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    records = read_manifest(args.manifest, args.root) if args.manifest else scan_imagefolder(args.root)
    records = apply_protocol(records, args.protocol, args.seed, args.val_fraction)
    write_manifest(records, args.output, args.root)
    counts = Counter(sample.split for sample in records)
    print(f"Saved {args.output}: " + ", ".join(f"{key}={counts[key]}" for key in ("train", "val", "test")))


if __name__ == "__main__":
    main()
