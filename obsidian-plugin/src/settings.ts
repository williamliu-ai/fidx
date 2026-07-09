// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import {
  DEFAULT_ALLOW_HIDDEN_DIRECTORY_INDEXING,
  DEFAULT_COLLECTION_NAME,
  DEFAULT_EXCLUDE_GLOBS,
  DEFAULT_INCLUDE_GLOBS,
  DEFAULT_INDEX_INACTIVITY_TIMEOUT_MS,
  DEFAULT_MAX_OUTPUT_BYTES,
  DEFAULT_RESULT_LIMIT,
  DEFAULT_SEARCH_TIMEOUT_MS,
  DEFAULT_TRUNCATE_POLICY,
  PLUGIN_ID,
} from "./constants";

export interface FidxSettings {
  fidxPath: string;
  dbPath: string;
  collectionName: string;
  indexedRoot: string;
  rootPrefix: string;
  includeGlobs: string[];
  excludeGlobs: string[];
  resultLimit: number;
  truncatePolicy: string;
  searchTimeoutMs: number;
  indexInactivityTimeoutMs: number;
  maxOutputBytes: number;
  allowHiddenDirectoryIndexing: boolean;
}

export function normalizeSettingsPath(value: string): string {
  const normalized = value.replace(/\\/g, "/").replace(/\/+/g, "/");
  if (normalized.length <= 1) {
    return normalized;
  }
  return normalized.replace(/\/+$/g, "");
}

function joinSettingsPath(root: string, child: string): string {
  const normalizedRoot = normalizeSettingsPath(root);
  const normalizedChild = normalizeSettingsPath(child).replace(/^\/+/, "");
  if (!normalizedRoot) {
    return normalizedChild;
  }
  return `${normalizedRoot}/${normalizedChild}`;
}

export function createDefaultSettings(vaultRoot: string): FidxSettings {
  const indexedRoot = normalizeSettingsPath(vaultRoot);
  return {
    fidxPath: "",
    dbPath: indexedRoot ? joinSettingsPath(indexedRoot, `.obsidian/plugins/${PLUGIN_ID}/fidx.db`) : "",
    collectionName: DEFAULT_COLLECTION_NAME,
    indexedRoot,
    rootPrefix: "",
    includeGlobs: [...DEFAULT_INCLUDE_GLOBS],
    excludeGlobs: [...DEFAULT_EXCLUDE_GLOBS],
    resultLimit: DEFAULT_RESULT_LIMIT,
    truncatePolicy: DEFAULT_TRUNCATE_POLICY,
    searchTimeoutMs: DEFAULT_SEARCH_TIMEOUT_MS,
    indexInactivityTimeoutMs: DEFAULT_INDEX_INACTIVITY_TIMEOUT_MS,
    maxOutputBytes: DEFAULT_MAX_OUTPUT_BYTES,
    allowHiddenDirectoryIndexing: DEFAULT_ALLOW_HIDDEN_DIRECTORY_INDEXING,
  };
}
