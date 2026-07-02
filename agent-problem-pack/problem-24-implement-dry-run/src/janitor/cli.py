"""Command-line entry point."""

import argparse

from src.janitor.cleanup import run_cleanup


def main(argv=None):
    parser = argparse.ArgumentParser(description="Delete stale scratch files.")
    parser.add_argument("root")
    parser.add_argument("--max-age-seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)
    run_cleanup(args.root, args.max_age_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
