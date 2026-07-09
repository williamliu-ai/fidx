// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const pluginReadme = readFileSync(new URL("../README.md", import.meta.url), "utf8");
const rootReadme = readFileSync(new URL("../../README.md", import.meta.url), "utf8");

describe("plugin README", () => {
  it("documents install, setup, and MVP commands", () => {
    expect(pluginReadme).toContain("uv tool install fmdidx");
    expect(pluginReadme).toContain(".obsidian/plugins/fidx-search/");
    expect(pluginReadme).toContain("fidx executable");
    expect(pluginReadme).toContain("Register vault collection");
    expect(pluginReadme).toContain("Index vault collection");
    expect(pluginReadme).toContain("Search vault");
    expect(pluginReadme).toContain("Cancel index");
    expect(pluginReadme).toContain("pnpm test tests/plugin-smoke.test.ts --run");
  });
});

describe("top-level README", () => {
  it("points users to the Obsidian plugin instructions", () => {
    expect(rootReadme).toContain("obsidian-plugin/README.md");
  });
});
