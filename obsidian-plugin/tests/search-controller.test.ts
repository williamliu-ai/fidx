// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { describe, expect, it, vi } from "vitest";

import { FidxRunResult } from "../src/fidx-command-runner";
import { FidxSearchResult } from "../src/fidx-service";
import {
  FidxSearchController,
  FidxSearchMapper,
  FidxSearchService,
} from "../src/search-controller";
import { createDefaultSettings, FidxSettings } from "../src/settings";
import { SettingsValidationResult } from "../src/settings-validation";
import { VaultFile, VaultPathMapResult } from "../src/vault-path-mapper";

function runResult(overrides: Partial<FidxRunResult> = {}): FidxRunResult {
  return {
    ok: true,
    exitCode: 0,
    stdout: JSON.stringify({
      schema: "fidx.search.v2",
      results: [{
        path: "fidx-search-vault/Folder/Note.md",
        collection: "fidx-search-vault",
        relpath: "Folder/Note.md",
        title: "Auth Note",
        snippet: "OAuth setup steps",
        score: 0.42,
        docid: "doc-1",
      }],
    }),
    stderr: "",
    timedOut: false,
    category: "ok",
    ...overrides,
  };
}

function searchEnvelope(results: FidxSearchResult[]): string {
  return JSON.stringify({
    schema: "fidx.search.v2",
    results,
  });
}

function resultFor(relpath: string, title: string): FidxSearchResult {
  return {
    path: `fidx-search-vault/${relpath}`,
    collection: "fidx-search-vault",
    relpath,
    title,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function settings(overrides: Partial<FidxSettings> = {}): FidxSettings {
  return {
    ...createDefaultSettings("/Vault"),
    fidxPath: "/usr/local/bin/fidx",
    ...overrides,
  };
}

function validMap(vaultPath = "Folder/Note.md"): VaultPathMapResult {
  return {
    ok: true,
    vaultPath,
    rootPrefix: "",
    file: { path: vaultPath },
  };
}

function harness(options: {
  service?: Partial<FidxSearchService>;
  mapper?: Partial<FidxSearchMapper>;
  validation?: SettingsValidationResult;
  settings?: FidxSettings;
  openFile?: (file: VaultFile) => Promise<void>;
} = {}) {
  const notifications: string[] = [];
  const opened: VaultFile[] = [];
  const service: FidxSearchService = {
    search: vi.fn(async () => runResult()),
    ...options.service,
  };
  const mapper: FidxSearchMapper = {
    mapResult: vi.fn(() => validMap()),
    ...options.mapper,
  };
  const instance = new FidxSearchController({
    settings: options.settings ?? settings(),
    service,
    mapper,
    notify: (message) => notifications.push(message),
    openFile: options.openFile ?? (async (file) => {
      opened.push(file);
    }),
    validateSettings: options.validation ? () => options.validation! : undefined,
  });
  return { instance, service, mapper, notifications, opened };
}

describe("FidxSearchController", () => {
  it("blocks search when settings validation fails", async () => {
    const invalid: SettingsValidationResult = {
      valid: false,
      derivedRootPrefix: "",
      issues: [{
        code: "fidx-path-empty",
        field: "fidxPath",
        message: "Set an absolute path to the fidx executable.",
      }],
    };
    const { instance, service } = harness({ validation: invalid });

    const state = await instance.search("oauth");

    expect(state).toMatchObject({
      status: "setup-error",
      message: "Set an absolute path to the fidx executable.",
    });
    expect(service.search).not.toHaveBeenCalled();
  });

  it("does not run fidx for blank queries", async () => {
    const { instance, service } = harness();

    const state = await instance.search("   ");

    expect(state).toMatchObject({
      status: "idle",
      message: "Enter a search query.",
    });
    expect(service.search).not.toHaveBeenCalled();
  });

  it("returns normalized result view models for successful searches", async () => {
    const { instance, service } = harness();

    const state = await instance.search("oauth login");

    expect(service.search).toHaveBeenCalledWith("oauth login");
    expect(state).toMatchObject({
      status: "results",
      query: "oauth login",
      results: [{
        title: "Auth Note",
        path: "Folder/Note.md",
        snippet: "OAuth setup steps",
        score: 0.42,
        docid: "doc-1",
      }],
    });
  });

  it("falls back to relpath as the result title", async () => {
    const { instance } = harness({
      service: {
        search: vi.fn(async () => runResult({
          stdout: searchEnvelope([{
            path: "fidx-search-vault/Folder/Untitled.md",
            collection: "fidx-search-vault",
            relpath: "Folder/Untitled.md",
          }]),
        })),
      },
    });

    const state = await instance.search("untitled");

    expect(state.results[0]?.title).toBe("Folder/Untitled.md");
  });

  it("distinguishes empty results from command and parse failures", async () => {
    const empty = harness({
      service: {
        search: vi.fn(async () => runResult({ stdout: searchEnvelope([]) })),
      },
    });
    await expect(empty.instance.search("missing")).resolves.toMatchObject({
      status: "empty",
      message: "No fidx results found.",
    });

    const commandFailure = harness({
      service: {
        search: vi.fn(async () => runResult({
          ok: false,
          exitCode: 2,
          stdout: "",
          stderr: "unexpected search failure",
          category: "nonzero-exit" as const,
          errorMessage: "fidx exited with code 2.",
        })),
      },
    });
    await expect(commandFailure.instance.search("oauth")).resolves.toMatchObject({
      status: "error",
      message: "fidx exited with code 2.",
    });

    const noIndex = harness({
      service: {
        search: vi.fn(async () => runResult({
          ok: false,
          exitCode: 2,
          stdout: "",
          stderr: "database has no index",
          category: "nonzero-exit" as const,
          errorMessage: "fidx exited with code 2.",
        })),
      },
    });
    await expect(noIndex.instance.search("oauth")).resolves.toMatchObject({
      status: "no-index",
      message: "No fidx index found. Run fidx indexing for this vault.",
    });

    const parseFailure = harness({
      service: {
        search: vi.fn(async () => runResult({ stdout: "not json" })),
      },
    });
    await expect(parseFailure.instance.search("oauth")).resolves.toMatchObject({
      status: "error",
      message: expect.stringMatching(/valid JSON/),
    });
  });

  it("does not let an older slow search replace newer results used for opening", async () => {
    const first = deferred<FidxRunResult>();
    const second = deferred<FidxRunResult>();
    const serviceSearch = vi.fn((query: string) => query === "first" ? first.promise : second.promise);
    const { instance, mapper, opened } = harness({
      service: { search: serviceSearch },
      mapper: {
        mapResult: vi.fn((_settings, result) => validMap(result.relpath)),
      },
    });

    const firstSearch = instance.search("first");
    const secondSearch = instance.search("second");
    second.resolve(runResult({ stdout: searchEnvelope([resultFor("Second.md", "Second")]) }));
    await expect(secondSearch).resolves.toMatchObject({
      status: "results",
      query: "second",
      results: [expect.objectContaining({ path: "Second.md" })],
    });

    first.resolve(runResult({ stdout: searchEnvelope([resultFor("First.md", "First")]) }));
    await expect(firstSearch).resolves.toMatchObject({
      status: "results",
      query: "second",
      results: [expect.objectContaining({ path: "Second.md" })],
    });
    await expect(instance.openResult(0)).resolves.toBe(true);

    expect(mapper.mapResult).toHaveBeenCalledWith(settings(), expect.objectContaining({
      relpath: "Second.md",
    }));
    expect(opened).toEqual([{ path: "Second.md" }]);
  });

  it("opens the mapped vault file for a selected result", async () => {
    const { instance, mapper, opened } = harness();

    await instance.search("oauth");
    const openedResult = await instance.openResult(0);

    expect(openedResult).toBe(true);
    expect(mapper.mapResult).toHaveBeenCalledWith(settings(), expect.objectContaining({
      relpath: "Folder/Note.md",
    }));
    expect(opened).toEqual([{ path: "Folder/Note.md" }]);
  });

  it("reports stale or unsafe mapped results instead of opening them", async () => {
    const { instance, notifications, opened } = harness({
      mapper: {
        mapResult: vi.fn(() => ({
          ok: false as const,
          reason: "stale-file" as const,
          vaultPath: "Folder/Missing.md",
          message: "The search result is stale; re-index the vault.",
        })),
      },
    });

    await instance.search("oauth");
    const openedResult = await instance.openResult(0);

    expect(openedResult).toBe(false);
    expect(opened).toEqual([]);
    expect(notifications).toEqual(["The search result is stale; re-index the vault."]);
  });

  it("notifies and returns false when Obsidian cannot open the mapped file", async () => {
    const { instance, notifications } = harness({
      openFile: async () => {
        throw new Error("workspace rejected the file");
      },
    });

    await instance.search("oauth");
    const openedResult = await instance.openResult(0);

    expect(openedResult).toBe(false);
    expect(notifications).toEqual(["Could not open search result: workspace rejected the file"]);
  });
});
