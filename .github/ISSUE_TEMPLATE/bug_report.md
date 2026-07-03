---
name: Bug report
about: Report a problem with fidx
title: "[bug] "
labels: bug
---

**What happened**
A clear description of the bug.

**To reproduce**
Exact commands, e.g.:
```sh
fidx collection add ./docs --name docs
fidx index
fidx search "..."
```

**`fidx doctor` output**
Paste the full output of `fidx doctor` (it reports your OS/arch, Python, sqlite,
extension/FTS5/sqlite-vec status, and model cache):
```
<paste here>
```

**Environment**
- Install method: `uv tool install` / `pipx` / `pip` / from source
- fidx version (`fidx --version`):
- OS + architecture:

**Expected behavior**
What you expected instead.
