#!/usr/bin/env python3
"""Check that every source-code link in docs/*.typ and README.md points at a
file (and, where a line number is given, a line) that actually exists.

Two link shapes are checked:

  * Typst: ``#src-link("path/to/file.py", ..., line: N)`` in docs/*.typ.
  * Markdown: ``[`Symbol`](path/to/file.py#L123)`` in README.md.

For each link:
  1. The referenced path must exist in the repository.
  2. If a line number is given, it must be within the file's line count.
  3. If a line number is given, that line must be a ``class``/``def``
     declaration -- these links are meant to point at definitions, not at
     arbitrary lines, so a link into the middle of a function body is
     itself a sign the target drifted after an edit.
  4. Where the link's own label names a symbol (a backtick-quoted
     identifier, optionally ``Class.method`` or ``func()``), that line must
     declare *that* symbol specifically, not just some definition.

This is a repo-hygiene check, not a Typst or Markdown parser: it uses
targeted regexes rather than parsing either format fully, so it can produce
false positives on unusual formatting. It has no false negatives for the
patterns above, which is the property that matters for catching stale line
numbers after a refactor.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# A definition line: optional indent, then `class Name` or `def name`.
DEF_RE = re.compile(r"^\s*(?:class|def)\s+(\w+)")

# #src-link("path"[, ..., line: N][, ...])  -- path and an optional `line:`
# kwarg, in either order, on a single line (all current usages are single-line).
SRC_LINK_RE = re.compile(r'#src-link\(\s*"(?P<path>[^"]+)"(?P<rest>[^)]*)\)')
LINE_KWARG_RE = re.compile(r"line:\s*(?P<line>\d+)")

# [`Symbol`](path/to/file.py#L123) or [`Symbol.method()`](path#L123)
MD_LINK_RE = re.compile(
    r"\[`(?P<label>[^`]+)`\]\((?P<path>src/[^)#\s]+)(?:#L(?P<line>\d+))?\)"
)


def _symbol_from_label(label: str) -> str | None:
    """Extract the bare def/class name a doc link's label refers to.

    ``compute_wigner_analytically()`` -> ``compute_wigner_analytically``
    ``GaussianState.to_qutip()``       -> ``to_qutip``
    ``GaussianState``                  -> ``GaussianState``
    """
    label = label.strip().rstrip("()")
    if "." in label:
        label = label.rsplit(".", 1)[-1]
    if not re.fullmatch(r"\w+", label):
        return None
    return label


def _nearby_backticked_symbol(text: str, match_start: int) -> str | None:
    """Best-effort: the last `` `Identifier` `` token before a Typst
    src-link call on the same line, e.g. in
    "`KerrCavity` (in #src-link(...))" this returns "KerrCavity"."""
    before = text[:match_start]
    tokens = re.findall(r"`([A-Za-z_][\w.]*)`", before)
    if not tokens:
        return None
    return _symbol_from_label(tokens[-1])


def _check(
    path_str: str, line_no: int | None, symbol: str | None, where: str
) -> list[str]:
    errors: list[str] = []
    target = REPO_ROOT / path_str
    if not target.is_file():
        errors.append(f"{where}: linked path does not exist: {path_str}")
        return errors

    if line_no is None:
        return errors

    lines = target.read_text().splitlines()
    if not (1 <= line_no <= len(lines)):
        errors.append(
            f"{where}: line {line_no} is out of range for {path_str} ({len(lines)} lines)"
        )
        return errors

    line_text = lines[line_no - 1]
    def_match = DEF_RE.match(line_text)
    if def_match is None:
        errors.append(
            f"{where}: {path_str}#L{line_no} is not a class/def line "
            f"(got: {line_text.strip()!r})"
        )
        return errors

    if symbol is not None and def_match.group(1) != symbol:
        errors.append(
            f"{where}: {path_str}#L{line_no} defines {def_match.group(1)!r}, "
            f"expected {symbol!r} -- the link is stale, most likely because "
            f"the source moved without the doc link being updated"
        )
    return errors


def check_typst_docs() -> list[str]:
    errors: list[str] = []
    for typ_file in sorted((REPO_ROOT / "docs").glob("*.typ")):
        text = typ_file.read_text()
        for match in SRC_LINK_RE.finditer(text):
            path_str = match.group("path")
            line_match = LINE_KWARG_RE.search(match.group("rest"))
            line_no = int(line_match.group("line")) if line_match else None
            symbol = (
                _nearby_backticked_symbol(text, match.start())
                if line_no is not None
                else None
            )
            where = f"{typ_file.relative_to(REPO_ROOT)}"
            errors.extend(_check(path_str, line_no, symbol, where))
    return errors


def check_readme() -> list[str]:
    errors: list[str] = []
    readme = REPO_ROOT / "README.md"
    text = readme.read_text()
    for match in MD_LINK_RE.finditer(text):
        path_str = match.group("path")
        line_str = match.group("line")
        line_no = int(line_str) if line_str is not None else None
        symbol = _symbol_from_label(match.group("label")) if line_no is not None else None
        errors.extend(_check(path_str, line_no, symbol, "README.md"))
    return errors


def main() -> int:
    errors = check_typst_docs() + check_readme()
    if errors:
        print(f"Found {len(errors)} stale/broken source link(s):\n", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("All doc/README source links point at the right file and line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
