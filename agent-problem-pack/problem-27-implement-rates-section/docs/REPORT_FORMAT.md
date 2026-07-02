# Summary report format

The report is a plain-text document assembled from named sections. Each
section starts with a `== <name> ==` banner line, followed by the section's
body lines. Sections appear in the order they are registered in
`src/reportkit/sections.py` (`SECTIONS`).

## Current sections

### header

```
== header ==
suite: <suite>
records: <total record count>
```

### counts

```
== counts ==
pass: <n>
fail: <n>
error: <n>
skip: <n>
```

Statuses other than these four are counted as `error`.

## Planned: rates (not yet implemented)

A `rates` section, registered immediately AFTER `counts`, computing rates
over the records that were actually attempted (every record whose status is
not `skip`):

```
== rates ==
pass_rate: <percent>
error_rate: <percent>
```

- `pass_rate` is `pass / attempted`; `error_rate` is `(fail + error) / attempted`
  (`fail` and `error` both count as errors here).
- Percentages are rendered with exactly one decimal place and a trailing
  `%`, e.g. `87.5%` (use Python's `.1f` formatting, no extra spaces).
- If there are no attempted records, both lines must read `n/a` instead of
  a percentage: `pass_rate: n/a`.
