# fidx Search for Obsidian

Desktop-only Obsidian plugin scaffold for searching the current vault with an installed `fidx` command.

## Prerequisites

Install the fidx CLI first. The Python package is named `fmdidx` and installs the `fidx` command:

```sh
uv tool install fmdidx
```

A regular Python or pipx install is also fine as long as Obsidian can run the resulting `fidx` executable. GUI-launched Obsidian may not inherit the same `PATH` as a terminal, so setting an absolute executable path in plugin settings is preferred.

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

Enable the plugin from Obsidian's Community Plugins settings. The MVP is desktop-only because it uses local Node APIs to run the external fidx CLI.
