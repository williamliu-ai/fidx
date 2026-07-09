// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { describe, expect, it } from "vitest";

import {
  DEFAULT_COLLECTION_NAME,
  DEFAULT_EXCLUDE_GLOBS,
  DEFAULT_INCLUDE_GLOBS,
  DEFAULT_RESULT_LIMIT,
} from "../src/constants";
import { createDefaultSettings } from "../src/settings";

describe("default settings", () => {
  it("uses vault-scoped fidx defaults with hidden directory excludes", () => {
    const settings = createDefaultSettings("/Vault");

    expect(settings.fidxPath).toBe("");
    expect(settings.collectionName).toBe(DEFAULT_COLLECTION_NAME);
    expect(settings.indexedRoot).toBe("/Vault");
    expect(settings.rootPrefix).toBe("");
    expect(settings.dbPath).toBe("/Vault/.obsidian/plugins/fidx-search/fidx.db");
    expect(settings.includeGlobs).toEqual(DEFAULT_INCLUDE_GLOBS);
    expect(settings.excludeGlobs).toEqual(DEFAULT_EXCLUDE_GLOBS);
    expect(settings.resultLimit).toBe(DEFAULT_RESULT_LIMIT);
    expect(settings.truncatePolicy).toBe("");
    expect(settings.searchTimeoutMs).toBe(30000);
    expect(settings.indexInactivityTimeoutMs).toBe(0);
    expect(settings.maxOutputBytes).toBeGreaterThan(0);
  });

  it("returns fresh glob arrays for each settings object", () => {
    const first = createDefaultSettings("/Vault");
    const second = createDefaultSettings("/Vault");

    first.includeGlobs.push("extra.md");
    first.excludeGlobs.length = 0;

    expect(second.includeGlobs).toEqual(DEFAULT_INCLUDE_GLOBS);
    expect(second.excludeGlobs).toEqual(DEFAULT_EXCLUDE_GLOBS);
  });

  it("uses empty db and root paths when the vault base path is unavailable", () => {
    const settings = createDefaultSettings("");

    expect(settings.indexedRoot).toBe("");
    expect(settings.dbPath).toBe("");
    expect(settings.collectionName).toBe(DEFAULT_COLLECTION_NAME);
    expect(settings.excludeGlobs).toEqual([".obsidian/**", ".trash/**", ".git/**"]);
  });
});
