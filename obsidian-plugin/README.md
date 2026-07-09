# fidx Search for Obsidian

Desktop-only Obsidian plugin for registering, indexing, and searching the
current vault with an installed `fidx` command.

## Prerequisites

Install the fidx CLI first. The Python package is named `fmdidx` and installs the `fidx` command:

```sh
uv tool install fmdidx
```

A regular Python or pipx install is also fine as long as Obsidian can run the
resulting `fidx` executable. GUI-launched Obsidian may not inherit the same
`PATH` as a terminal, so setting an absolute `fidx executable` path in plugin
settings is preferred.

## Development

```sh
pnpm install
pnpm test
pnpm build
```

The build writes `main.js` in this directory. `main.js`, `node_modules/`, and coverage output are generated local artifacts and are ignored by git.

## Manual Install

Build the plugin, then copy or symlink these files into your vault at `.obsidian/plugins/fidx-search/`:

```text
main.js
manifest.json
styles.css
```

Enable the plugin from Obsidian's Community Plugins settings. The MVP is
desktop-only because it uses local Node APIs to run the external fidx CLI.

## Setup

Open the fidx Search settings tab and configure:

- `fidx executable`: absolute path to the `fidx` command.
- Database path: plugin-owned fidx database for this vault.
- Collection: collection name used for index and search commands.
- Indexed root: vault folder to register and index.
- Include globs and exclude globs: one glob per line. The default excludes keep
  `.obsidian/**`, `.trash/**`, and `.git/**` out of the index.
- Result limit and truncate policy: search output controls passed to fidx.

Run these commands from Obsidian's command palette:

- `fidx: Run doctor`
- `fidx: Register vault collection`
- `fidx: Index vault collection`
- `fidx: Cancel index`
- `fidx: Search vault`

Register after changing include/exclude globs or indexed root. Index after
registering, then use Search vault to open matching notes.

## Smoke Check

The plugin test suite includes a generated-vault subprocess smoke test:

```sh
pnpm test tests/plugin-smoke.test.ts --run
```

That check verifies the plugin service can register, index, and search through a
fidx-compatible executable without relying on private data or publishing latency
claims.
