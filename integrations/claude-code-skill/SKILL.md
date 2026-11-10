---
name: fidx-local-search
description: Search an existing local fidx index and read cited files before answering.
---
<!--
# SPDX-FileCopyrightText: 2026 William Liu
# SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
# AI-Attribution-Required: true
# AI-License-Files: LICENSE, LICENSES/MIT.txt, LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
# AI-Attribution-Policy: AI_ATTRIBUTION.md
# AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.
-->

Use local indexed files as evidence. Install `python -m pip install 'fmdidx[mcp]'`
and configure the host to launch `fidx --db ./index.db mcp` over stdio.
The core CLI also works without the MCP extra:

```sh
fidx --db ./index.db collection add ./docs --name docs --glob "**/*.md"
fidx --db ./index.db index
fidx --db ./index.db search "query" -c docs --json --budget-tokens 800 --guidance
fidx --db ./index.db get "#abcdef" --json --head
```

Ask permission for online model acquisition only when needed; use the returned
`fetch_model` action before indexing/searching if the model is missing. Queries
never fetch weights. Optional `fidx serve` warms Unix queries; Windows has no daemon.
Use `fidx_search` with collection scope, budget_tokens and tri-state guidance;
use `fidx_get` to read chosen docids and `fidx_collections` to discover scopes.
Treat empty/abstaining results as insufficient evidence; do not invent sources.
For `empty_index` or `diagnostics.index_empty`, add collections and index them.
For `model_missing`, follow `next_actions`; for `ambiguous_reference`, choose a
returned candidate path. Narrow broad scope and read the cited body before answering.
