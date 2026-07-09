// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { execFile } from "node:child_process";
import { promisify } from "node:util";

import { describe, expect, it } from "vitest";

import { createDefaultSettings, FidxSettings } from "../src/settings";
import {
  buildCollectionAddCommand,
  buildDoctorCommand,
  buildIndexCommand,
  buildSearchCommand,
  buildStatusCommand,
  parseSearchEnvelope,
  parseStatusOutput,
} from "../src/fidx-service";

const execFileAsync = promisify(execFile);

function settings(overrides: Partial<FidxSettings> = {}): FidxSettings {
  return {
    ...createDefaultSettings("/Vault"),
    fidxPath: "/usr/local/bin/fidx",
    ...overrides,
  };
}

describe("fidx command builders", () => {
  it("builds doctor without shell interpolation or db arguments", () => {
    expect(buildDoctorCommand(settings())).toMatchObject({
      executable: "/usr/local/bin/fidx",
      args: ["doctor"],
      timeoutMs: 30_000,
    });
  });

  it("builds collection add with include and exclude globs", () => {
    const command = buildCollectionAddCommand(settings());

    expect(command.args).toEqual([
      "--db",
      "/Vault/.obsidian/plugins/fidx-search/fidx.db",
      "collection",
      "add",
      "/Vault",
      "--name",
      "fidx-search-vault",
      "--glob",
      "**/*.md",
      "--glob",
      "**/*.markdown",
      "--glob",
      "**/*.txt",
      "--exclude-glob",
      ".obsidian/**",
      "--exclude-glob",
      ".trash/**",
      "--exclude-glob",
      ".git/**",
    ]);
  });

  it("builds scoped index, search, and status commands", () => {
    expect(buildIndexCommand(settings()).args).toEqual([
      "--db",
      "/Vault/.obsidian/plugins/fidx-search/fidx.db",
      "index",
      "-c",
      "fidx-search-vault",
    ]);

    expect(buildSearchCommand(settings(), "oauth login").args).toEqual([
      "--db",
      "/Vault/.obsidian/plugins/fidx-search/fidx.db",
      "search",
      "-c",
      "fidx-search-vault",
      "--json",
      "-n",
      "10",
      "--",
      "oauth login",
    ]);

    expect(buildStatusCommand(settings()).args).toEqual([
      "--db",
      "/Vault/.obsidian/plugins/fidx-search/fidx.db",
      "status",
    ]);
  });

  it("adds truncate only when a policy is enabled", () => {
    expect(buildSearchCommand(settings({ truncatePolicy: "knee" }), "query").args)
      .toContain("--truncate");
    expect(buildSearchCommand(settings({ truncatePolicy: "off" }), "query").args)
      .not.toContain("--truncate");
  });

  it("isolates a leading-dash query behind an option terminator", () => {
    const args = buildSearchCommand(settings(), "-c").args;
    const separator = args.indexOf("--");

    expect(separator).toBeGreaterThan(-1);
    expect(args.slice(separator + 1)).toEqual(["-c"]);
  });
});

describe("fidx CLI help conformance", () => {
  async function fidxHelp(...args: string[]): Promise<string> {
    const result = await execFileAsync("uv", ["run", "fidx", ...args, "--help"], {
      cwd: "..",
    });
    return result.stdout;
  }

  it("uses flags exposed by the current fidx CLI", async () => {
    await expect(fidxHelp("collection", "add")).resolves.toContain("--exclude-glob");
    await expect(fidxHelp("index")).resolves.toContain("-c, --collection");
    const searchHelp = await fidxHelp("search");
    expect(searchHelp).toContain("-c, --collection");
    expect(searchHelp).toContain("-n, --limit");
    expect(searchHelp).toContain("--json");
    expect(searchHelp).toContain("--truncate");
  });
});

describe("fidx search parsing", () => {
  it("accepts the fidx.search.v2 envelope and tolerates extra fields", () => {
    const envelope = parseSearchEnvelope(JSON.stringify({
      schema: "fidx.search.v2",
      results: [{
        path: "fidx-search-vault/Folder/Note.md",
        collection: "fidx-search-vault",
        relpath: "Folder/Note.md",
        docid: "#abc",
        snippet: "hello",
        extra: { ok: true },
      }],
      diagnostics: { ignored: true },
    }));

    expect(envelope.results[0]).toMatchObject({
      collection: "fidx-search-vault",
      relpath: "Folder/Note.md",
      path: "fidx-search-vault/Folder/Note.md",
    });
  });

  it("accepts an empty results array", () => {
    const envelope = parseSearchEnvelope(JSON.stringify({
      schema: "fidx.search.v2",
      results: [],
    }));

    expect(envelope.results).toEqual([]);
  });

  it("rejects invalid JSON, wrong schema, and missing result fields", () => {
    expect(() => parseSearchEnvelope("{")).toThrow(/valid JSON/);
    expect(() => parseSearchEnvelope(JSON.stringify({ schema: "other", results: [] })))
      .toThrow(/fidx.search.v2/);
    expect(() => parseSearchEnvelope(JSON.stringify({
      schema: "fidx.search.v2",
      results: [{ collection: "fidx-search-vault", path: "x.md" }],
    }))).toThrow(/relpath/);
    expect(() => parseSearchEnvelope(JSON.stringify({
      schema: "fidx.search.v2",
      results: ["x"],
    }))).toThrow(/must be an object/);
  });

  it("treats status output as opaque text", () => {
    expect(parseStatusOutput("collections: 1\nvectors: 42\n")).toBe("collections: 1\nvectors: 42\n");
  });
});
