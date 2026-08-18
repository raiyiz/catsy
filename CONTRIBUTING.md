# Contributing to catsy

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://gitlab.uni-hannover.de/inl/catsy.git
cd catsy
uv sync --group dev
```

## Workflow

All work happens on `main` via merge requests; there are no long-lived
feature branches. Before opening a merge request:

```bash
uv run pytest          # full test suite
uv run pytest --plot   # include the opt-in plotting tests
uv run ruff check .    # lint
```

Both `pytest` and `ruff` run in CI (see `.gitlab-ci.yml`) and must pass
before a merge request can land.

## Conventions

* Follow the phase-space and naming conventions documented in the
  [README](README.md) and in `docs/`.
* Keep new public API covered by tests, including the physical invariants
  (uncertainty relations, symplectic transformations, exact loss limits)
  the existing suite already checks for.
* Plotting helpers should be exercised through tests marked
  `@pytest.mark.visual` so they stay opt-in (`--plot`) and don't slow down
  the default test run.

## Documentation

The architectural-specs PDF (`docs/book.typ`) is built manually via the
`typst` CI job. If you change public API, check whether `docs/` needs a
corresponding update.
