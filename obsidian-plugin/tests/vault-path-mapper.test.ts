// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { describe, expect, it } from "vitest";

import { FidxSearchResult } from "../src/fidx-service";
import { createDefaultSettings, FidxSettings } from "../src/settings";
import { VaultFileLookup, VaultPathMapper } from "../src/vault-path-mapper";

class FakeVaultLookup implements VaultFileLookup {
  constructor(private readonly paths: Set<string>) {}

  getFileByPath(path: string) {
    return this.paths.has(path) ? { path } : null;
  }
}

function settings(overrides: Partial<FidxSettings> = {}): FidxSettings {
  return {
    ...createDefaultSettings("/Vault"),
    collectionName: "fidx-search-vault",
    ...overrides,
  };
}

function result(overrides: Partial<FidxSearchResult> = {}): FidxSearchResult {
  return {
    path: "fidx-search-vault/Folder/Note.md",
    collection: "fidx-search-vault",
    relpath: "Folder/Note.md",
    ...overrides,
  };
}

function mapper(paths: string[], vaultRoot = "/Vault") {
  return new VaultPathMapper(vaultRoot, new FakeVaultLookup(new Set(paths)));
}

describe("VaultPathMapper", () => {
  it("maps root-vault results directly to vault paths", () => {
    const mapped = mapper(["Folder/Note.md"]).mapResult(settings(), result());

    expect(mapped).toMatchObject({
      ok: true,
      vaultPath: "Folder/Note.md",
      file: { path: "Folder/Note.md" },
    });
  });

  it("prefixes results for indexed subfolders", () => {
    const mapped = mapper(["Projects/Folder/Note.md"]).mapResult(
      settings({ indexedRoot: "/Vault/Projects", rootPrefix: "wrong/manual" }),
      result(),
    );

    expect(mapped).toMatchObject({
      ok: true,
      vaultPath: "Projects/Folder/Note.md",
    });
  });

  it("rejects out-of-vault indexed roots", () => {
    const mapped = mapper(["Folder/Note.md"]).mapResult(
      settings({ indexedRoot: "/Other" }),
      result(),
    );

    expect(mapped).toMatchObject({ ok: false, reason: "root-outside-vault" });
  });

  it("rejects traversal in indexed roots before joining paths", () => {
    const mapped = mapper(["Secret/Folder/Note.md"]).mapResult(
      settings({ indexedRoot: "/Vault/../Secret" }),
      result(),
    );

    expect(mapped).toMatchObject({ ok: false, reason: "root-outside-vault" });
  });

  it("rejects collection mismatches", () => {
    const mapped = mapper(["Folder/Note.md"]).mapResult(settings(), result({
      collection: "other",
    }));

    expect(mapped).toMatchObject({ ok: false, reason: "collection-mismatch" });
  });

  it("rejects traversal attempts before vault lookup", () => {
    const mapped = mapper(["Secrets.md"]).mapResult(settings(), result({
      relpath: "../Secrets.md",
    }));

    expect(mapped).toMatchObject({ ok: false, reason: "path-traversal" });
  });

  it("rejects absolute and empty relpaths before vault lookup", () => {
    for (const relpath of ["/etc/passwd", "C:/Windows/win.ini", "   "]) {
      const mapped = mapper(["Folder/Note.md"]).mapResult(settings(), result({ relpath }));

      expect(mapped).toMatchObject({ ok: false, reason: "path-traversal" });
    }
  });

  it("reports stale files when the mapped vault path is missing", () => {
    const mapped = mapper([]).mapResult(settings(), result());

    expect(mapped).toMatchObject({
      ok: false,
      reason: "stale-file",
      vaultPath: "Folder/Note.md",
    });
  });

  it("normalizes Windows backslash relpaths before lookup", () => {
    const mapped = mapper(["Folder/Note.md"]).mapResult(settings(), result({
      relpath: "Folder\\Note.md",
    }));

    expect(mapped).toMatchObject({
      ok: true,
      vaultPath: "Folder/Note.md",
    });
  });

  it("maps Windows indexed roots and relpaths", () => {
    const mapped = mapper(["Projects/Folder/Note.md"], "C:\\Vault").mapResult(
      settings({ indexedRoot: "C:\\Vault\\Projects" }),
      result({ relpath: "Folder\\Note.md" }),
    );

    expect(mapped).toMatchObject({
      ok: true,
      vaultPath: "Projects/Folder/Note.md",
    });
  });
});
