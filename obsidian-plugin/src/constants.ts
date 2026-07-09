// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

export const PLUGIN_ID = "fidx-search";
export const PLUGIN_NAME = "fidx Search";

export const DEFAULT_COLLECTION_NAME = "fidx-search-vault";
export const DEFAULT_INCLUDE_GLOBS = ["**/*.md", "**/*.markdown", "**/*.txt"] as const;
export const DEFAULT_EXCLUDE_GLOBS = [".obsidian/**", ".trash/**", ".git/**"] as const;
export const DEFAULT_RESULT_LIMIT = 10;
export const DEFAULT_SEARCH_TIMEOUT_MS = 30_000;
export const DEFAULT_INDEX_INACTIVITY_TIMEOUT_MS = 0;
export const DEFAULT_MAX_OUTPUT_BYTES = 1_048_576;
export const DEFAULT_TRUNCATE_POLICY = "";
