# Auto-Indexing with systemd Path Units

fidx doesn't include a built-in file watcher, but on Linux you can use
systemd **path units** to trigger automatic reindexing whenever files in
your collections change. This approach has three advantages over a
watcher daemon:

- **Zero idle memory** — systemd is already running; no watcher process
  sits in RAM between events.
- **Zero dependencies** — no `watchfiles`, `inotifywait`, or any extra
  packages. Only systemd (present on virtually all modern Linux).
- **Built-in debounce** — a `sleep` in the service unit collapses bursts
  of events (git pull, rsync, cloud sync) into a single reindex pass.

## How It Works

systemd's `path` unit type watches directories via inotify. When a file
changes, it starts a companion `service` unit. The service sleeps briefly
(debounce), then runs `fidx index` — which is incremental, so only
changed files are re-chunked and re-embedded.

```
file changes → systemd path unit → service starts
                                    → sleep 10s (debounce)
                                    → fidx index (incremental)
                                    → service exits
```

## Setup

Create two files in `~/.config/systemd/user/`:

### `fidx-index.path`

```ini
[Unit]
Description=Watch for changes in fidx collections

[Path]
# Add one PathModified line per collection directory.
# Run `fidx collection list` to see your registered collections.
PathModified=/home/user/notes
PathModified=/home/user/documents
PathModified=/home/user/journal

[Install]
WantedBy=default.target
```

### `fidx-index.service`

```ini
[Unit]
Description=fidx incremental reindex
# Rate limit: max 5 reindexes per 5 minutes.
# If exceeded (massive bulk change), the unit enters failed state.
# Reset with: systemctl --user reset-failed fidx-index.service
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=oneshot
# Debounce: wait 10 seconds before reindexing.
# Collapses bursts of events (git pull, rsync, cloud sync)
# into a single reindex pass.
ExecStartPre=/bin/sleep 10
ExecStart=/usr/local/bin/fidx index --parallel 1 --threads 4
```

### Enable

```bash
systemctl --user daemon-reload
systemctl --user enable --now fidx-index.path
```

### Verify

```bash
# Check the path unit is active
systemctl --user status fidx-index.path

# Edit a file in a watched directory, then watch:
journalctl --user -u fidx-index.service -f
```

## Tuning

| Parameter | What to change | Why |
|-----------|---------------|-----|
| **Debounce** | `ExecStartPre=/bin/sleep 30` | Longer wait for large bulk operations (git clone, initial sync) |
| **Threads** | `--threads 8` in ExecStart | Use all cores for faster embedding. Reduce if memory-constrained. |
| **Parallel** | Keep `--parallel 1` | `parallel > 1` spawns multiple ONNX model copies (~1-2 GB each). On memory-limited machines this can trigger OOM. |
| **Collections** | Add/remove `PathModified` lines | Must match `fidx collection list`. Run `systemctl --user daemon-reload` after editing. |
| **Rate limit** | `StartLimitIntervalSec` / `StartLimitBurst` | Prevents runaway reindex loops. Adjust based on your change frequency. |

## How Debounce Works

The `ExecStartPre=/bin/sleep 10` line is the debounce mechanism:

1. First file changes → systemd starts the service.
2. Service sleeps 10 seconds. More events arrive during this window —
   they're ignored because the service is already running.
3. After 10 seconds, `fidx index` runs and catches **all** changes
   (incremental, hash-based — only changed files are reprocessed).
4. Service exits.

If another event fires while the service is running, systemd won't
restart it (it's still active). Those changes wait until the next file
event triggers a new cycle.

## Edge Cases

- **Changes during reindex:** files modified while `fidx index` is
  running won't trigger an immediate reindex (service is still active).
  They'll be caught on the next triggered cycle. Self-correcting.

- **New collection added:** add a `PathModified` line to `fidx-index.path`,
  then `systemctl --user daemon-reload`.

- **`fidx serve` running concurrently:** safe. SQLite handles concurrent
  read (serve daemon) and write (index). No lock contention at typical
  corpus sizes.

- **Profile mismatch:** if you created the index with `--profile bge-384`,
  include `--profile bge-384` in the `ExecStart` line. Otherwise fidx
  will refuse to reindex (profile pinned in index metadata).

- **Massive bulk changes** (thousands of files): the rate limit
  (`StartLimitBurst`) may trip, putting the service into failed state.
  Reset with `systemctl --user reset-failed fidx-index.service` and
  run `fidx index` manually once.

## Limitations

- **Linux + systemd only.** macOS users can use `launchd` `WatchPaths`.
  Windows users can use `watchfiles` or a scheduled task.
- **`PathModified` is not recursive by default.** Each subdirectory you
  want watched needs its own `PathModified` line. Alternatively, use
  `PathChanged` on the top-level directory (less precise, triggers on
  any change in the tree).
- **No push notification.** The index updates silently. If you need to
  know when reindexing happens, check `journalctl --user -u fidx-index`.
