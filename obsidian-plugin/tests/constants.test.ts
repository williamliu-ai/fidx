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
  PLUGIN_ID,
  PLUGIN_NAME,
} from "../src/constants";

describe("plugin constants", () => {
  it("pins the privacy-critical hidden-directory excludes", () => {
    expect(DEFAULT_EXCLUDE_GLOBS).toEqual([".obsidian/**", ".trash/**", ".git/**"]);
  });

  it("pins include globs, collection name, and result limit", () => {
    expect(DEFAULT_INCLUDE_GLOBS).toEqual(["**/*.md", "**/*.markdown", "**/*.txt"]);
    expect(DEFAULT_COLLECTION_NAME).toBe("fidx-search-vault");
    expect(DEFAULT_RESULT_LIMIT).toBe(10);
  });

  it("uses the fidx-search identity", () => {
    expect(PLUGIN_ID).toBe("fidx-search");
    expect(PLUGIN_NAME).toBe("fidx Search");
  });
});
