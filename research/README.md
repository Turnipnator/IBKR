# Research studies and decision trails

Not part of the bot. Nothing in `src/` imports from here, the Dockerfile copies
only `src/` into the image, and pytest discovers `tests/` only — so these files
never reach the running container.

- `../RESEARCH.md` — the protocol to follow for any research or strategy study.
- `../research_notes.md` — running findings, newest study first (the protocol
  asks for this file at the repo root).
- `<date>_<topic>/` — the script and results behind each study, kept so a
  parameter decision can be re-derived later. Commit scripts, notes and small
  result files; if a study produces raw bar dumps, gitignore those instead.
