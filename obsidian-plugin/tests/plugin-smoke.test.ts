// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { chmod, mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { FidxService, parseSearchEnvelope } from "../src/fidx-service";
import { createDefaultSettings } from "../src/settings";

async function writeFakeFidxExecutable(dir: string): Promise<string> {
  const executable = join(dir, "fidx-fake.cjs");
  await writeFile(executable, `#!/usr/bin/env node
const fs = require("node:fs");
const args = process.argv.slice(2);
const dbIndex = args.indexOf("--db");
const dbPath = dbIndex >= 0 ? args[dbIndex + 1] : "";
const command = dbIndex >= 0 ? args.slice(dbIndex + 2) : args;
if (dbPath) {
  fs.appendFileSync(dbPath + ".calls.jsonl", JSON.stringify(command) + "\\n");
}
if (command[0] === "doctor") {
  process.stdout.write("doctor ok");
  process.exit(0);
}
if (command[0] === "collection" && command[1] === "add") {
  process.stdout.write("registered");
  process.exit(0);
}
if (command[0] === "index") {
  process.stderr.write("indexed Notes/Auth.md\\n");
  process.exit(0);
}
if (command[0] === "search") {
  const collection = command[command.indexOf("-c") + 1];
  process.stdout.write(JSON.stringify({
    schema: "fidx.search.v2",
    results: [{
      path: collection + "/Notes/Auth.md",
      collection,
      relpath: "Notes/Auth.md",
      title: "Auth",
      snippet: "OAuth local note",
      score: 0.5
    }]
  }));
  process.exit(0);
}
process.stderr.write("unexpected command: " + command.join(" "));
process.exit(2);
`);
  await chmod(executable, 0o755);
  return executable;
}

describe("plugin subprocess smoke", () => {
  it("registers, indexes, and searches a generated vault fixture through FidxService", async () => {
    const root = await mkdtemp(join(tmpdir(), "fidx-plugin-smoke-"));
    const vaultRoot = join(root, "Vault");
    await mkdir(join(vaultRoot, "Notes"), { recursive: true });
    await mkdir(join(vaultRoot, ".obsidian", "plugins", "fidx-search"), { recursive: true });
    await mkdir(join(vaultRoot, ".trash"), { recursive: true });
    await writeFile(join(vaultRoot, "Notes", "Auth.md"), "# Auth\nOAuth local note\n");
    await writeFile(join(vaultRoot, ".trash", "Deleted.md"), "do not index\n");

    const fidxPath = await writeFakeFidxExecutable(root);
    const dbPath = join(root, "fidx.db");
    const settings = {
      ...createDefaultSettings(vaultRoot),
      fidxPath,
      dbPath,
      indexInactivityTimeoutMs: 1_000,
    };
    const service = new FidxService(settings);

    await expect(service.registerCollection()).resolves.toMatchObject({ ok: true });
    await expect(service.index()).resolves.toMatchObject({ ok: true });
    const search = await service.search("oauth");
    const envelope = parseSearchEnvelope(search.stdout);
    const calls = (await readFile(`${dbPath}.calls.jsonl`, "utf8"))
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as string[]);

    expect(envelope.results[0]).toMatchObject({
      collection: "fidx-search-vault",
      relpath: "Notes/Auth.md",
    });
    expect(calls).toContainEqual(expect.arrayContaining([
      "collection",
      "add",
      vaultRoot,
      "--exclude-glob",
      ".obsidian/**",
      "--exclude-glob",
      ".trash/**",
    ]));
    expect(calls).toContainEqual(["index", "-c", "fidx-search-vault"]);
    expect(calls).toContainEqual(expect.arrayContaining([
      "search",
      "-c",
      "fidx-search-vault",
      "--json",
      "-n",
      "10",
      "--",
      "oauth",
    ]));
  });
});
