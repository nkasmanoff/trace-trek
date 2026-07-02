# janitor

Deletes stale scratch files (`*.tmp`, `*.log`) older than a cutoff.

Usage:

    python -m src.janitor.cli <root> --max-age-seconds 3600

Prints one `deleted <path>` line per removed file, in sorted path order.

## Roadmap

- `--dry-run`: print one `would delete <path>` line per candidate (same
  selection and ordering as a real run), delete nothing, exit 0.
