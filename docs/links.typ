// Shared helper for linking documentation text back to the exact source on
// GitLab. `repo-url` and `commit-sha` are injected by the documentation CI
// job via `--input`, so every link resolves to the exact commit this PDF was
// built from. Falls back to the private GitLab repository's `main` branch
// when compiled locally without `--input`.
#let repo-url = sys.inputs.at("repo-url", default: "https://gitlab.uni-hannover.de/inl/catsy")
#let commit-sha = sys.inputs.at("commit-sha", default: "main")

// Renders a clickable reference to `path` (optionally anchored to `line`)
// at the pinned commit. `label` (content, e.g. `` [`optics.py`] ``) defaults
// to the path itself shown as inline code. Note: `label` is intentionally a
// named parameter, not positional -- pass it as `label: [...]` rather than
// trailing-bracket syntax, since a trailing `[...]` after the call binds to
// the next *positional* parameter, and `src-link` only has one (`path`).
#let src-link(path, label: none, line: none) = {
  let url = repo-url + "/-/blob/" + commit-sha + "/" + path
  if line != none {
    url = url + "#L" + str(line)
  }
  let shown = if label == none { raw(path, lang: none) } else { label }
  link(url)[#shown]
}
