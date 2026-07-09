// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { DEFAULT_EXCLUDE_GLOBS, PLUGIN_ID } from "./constants";
import { FidxSettings, normalizeSettingsPath } from "./settings";

export type SettingsValidationCode =
  | "fidx-path-empty"
  | "fidx-path-missing"
  | "fidx-path-not-executable"
  | "indexed-root-invalid"
  | "indexed-root-outside-vault"
  | "include-globs-empty"
  | "include-globs-invalid"
  | "exclude-globs-invalid"
  | "default-excludes-missing"
  | "result-limit-out-of-range"
  | "truncate-policy-unsupported";

export interface SettingsValidationIssue {
  code: SettingsValidationCode;
  field: keyof FidxSettings;
  message: string;
}

export interface SettingsValidationEnv {
  fileExists(path: string): boolean;
  isDirectory(path: string): boolean;
  isExecutable(path: string): boolean;
}

export interface SettingsValidationResult {
  valid: boolean;
  issues: SettingsValidationIssue[];
  derivedRootPrefix: string;
}

const SUPPORTED_TRUNCATE_POLICIES = new Set(["", "off", "knee"]);

function issue(
  code: SettingsValidationCode,
  field: keyof FidxSettings,
  message: string,
): SettingsValidationIssue {
  return { code, field, message };
}

export function inferVaultRoot(settings: FidxSettings): string {
  const dbPath = normalizeSettingsPath(settings.dbPath);
  const marker = `/.obsidian/plugins/${PLUGIN_ID}/`;
  const markerIndex = dbPath.indexOf(marker);
  if (markerIndex > 0) {
    return dbPath.slice(0, markerIndex);
  }
  return normalizeSettingsPath(settings.indexedRoot);
}

export function deriveRootPrefix(vaultRoot: string, indexedRoot: string): string {
  const vault = normalizeSettingsPath(vaultRoot);
  const root = normalizeSettingsPath(indexedRoot);
  if (!vault || !root || hasTraversalSegment(vault) || hasTraversalSegment(root)) {
    throw new Error("indexed root must be inside the active vault");
  }
  if (root === vault) {
    return "";
  }
  const vaultPrefix = `${vault}/`;
  if (!root.startsWith(vaultPrefix)) {
    throw new Error("indexed root must be inside the active vault");
  }
  return root.slice(vaultPrefix.length);
}

function hasTraversalSegment(path: string): boolean {
  return path.split("/").some((part) => part === "." || part === "..");
}

function listIsNonEmptyStrings(values: string[]): boolean {
  return values.length > 0 && values.every((value) => value.trim().length > 0);
}

function missingDefaultExcludes(excludeGlobs: string[]): string[] {
  return DEFAULT_EXCLUDE_GLOBS.filter((glob) => !excludeGlobs.includes(glob));
}

export function validateFidxSettings(
  settings: FidxSettings,
  env: SettingsValidationEnv,
  vaultRoot = inferVaultRoot(settings),
): SettingsValidationResult {
  const issues: SettingsValidationIssue[] = [];

  if (!settings.fidxPath.trim()) {
    issues.push(issue("fidx-path-empty", "fidxPath", "Set an absolute path to the fidx executable."));
  } else if (!env.fileExists(settings.fidxPath)) {
    issues.push(issue("fidx-path-missing", "fidxPath", "The configured fidx executable does not exist."));
  } else if (!env.isExecutable(settings.fidxPath)) {
    issues.push(issue("fidx-path-not-executable", "fidxPath", "The configured fidx path is not executable."));
  }

  if (!settings.indexedRoot.trim() || !env.isDirectory(settings.indexedRoot)) {
    issues.push(issue("indexed-root-invalid", "indexedRoot", "The indexed root must be an existing directory."));
  }

  let derivedRootPrefix = "";
  try {
    derivedRootPrefix = deriveRootPrefix(vaultRoot, settings.indexedRoot);
  } catch {
    issues.push(issue(
      "indexed-root-outside-vault",
      "indexedRoot",
      "The indexed root must be inside the active vault.",
    ));
  }

  if (settings.includeGlobs.length === 0) {
    issues.push(issue("include-globs-empty", "includeGlobs", "At least one include glob is required."));
  } else if (!listIsNonEmptyStrings(settings.includeGlobs)) {
    issues.push(issue("include-globs-invalid", "includeGlobs", "Include globs must be non-empty strings."));
  }

  if (!settings.allowHiddenDirectoryIndexing && missingDefaultExcludes(settings.excludeGlobs).length > 0) {
    issues.push(issue(
      "default-excludes-missing",
      "excludeGlobs",
      "Default hidden/private directory excludes require an explicit opt-out to remove.",
    ));
  }

  if (settings.excludeGlobs.length > 0 && !listIsNonEmptyStrings(settings.excludeGlobs)) {
    issues.push(issue("exclude-globs-invalid", "excludeGlobs", "Exclude globs must be non-empty strings."));
  }

  if (!Number.isInteger(settings.resultLimit) || settings.resultLimit < 1 || settings.resultLimit > 100) {
    issues.push(issue("result-limit-out-of-range", "resultLimit", "Result limit must be between 1 and 100."));
  }

  if (!SUPPORTED_TRUNCATE_POLICIES.has(settings.truncatePolicy.trim())) {
    issues.push(issue(
      "truncate-policy-unsupported",
      "truncatePolicy",
      "Supported truncate policies are empty, off, and knee.",
    ));
  }

  return {
    valid: issues.length === 0,
    issues,
    derivedRootPrefix,
  };
}
