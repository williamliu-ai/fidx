// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { describe, expect, it } from "vitest";

import { createDefaultSettings, FidxSettings } from "../src/settings";
import {
  deriveRootPrefix,
  inferVaultRoot,
  validateFidxSettings,
  SettingsValidationEnv,
} from "../src/settings-validation";

const validEnv: SettingsValidationEnv = {
  fileExists: () => true,
  isDirectory: () => true,
  isExecutable: () => true,
};

function validSettings(overrides: Partial<FidxSettings> = {}): FidxSettings {
  return {
    ...createDefaultSettings("/Vault"),
    fidxPath: "/usr/local/bin/fidx",
    ...overrides,
  };
}

describe("deriveRootPrefix", () => {
  it("derives an empty prefix when indexed root is the vault root", () => {
    expect(deriveRootPrefix("/Vault", "/Vault")).toBe("");
  });

  it("derives a vault-relative prefix for an indexed subfolder", () => {
    expect(deriveRootPrefix("/Vault", "/Vault/Projects/Notes")).toBe("Projects/Notes");
  });

  it("normalizes native separators and trailing slashes", () => {
    expect(deriveRootPrefix("C:\\Vault\\", "C:\\Vault\\Projects\\")).toBe("Projects");
  });

  it("rejects indexed roots outside the active vault", () => {
    expect(() => deriveRootPrefix("/Vault", "/Other/Notes")).toThrow(/inside the active vault/);
  });

  it("rejects traversal and sibling-prefix indexed roots", () => {
    expect(() => deriveRootPrefix("/Vault", "/Vault/../Secret")).toThrow(/inside the active vault/);
    expect(() => deriveRootPrefix("/Vault", "/Vault/../../etc")).toThrow(/inside the active vault/);
    expect(() => deriveRootPrefix("/Vault", "/VaultOther")).toThrow(/inside the active vault/);
  });
});

describe("inferVaultRoot", () => {
  it("infers the vault root from the plugin-owned database path", () => {
    expect(inferVaultRoot(validSettings())).toBe("/Vault");
  });

  it("falls back to indexedRoot when the database path is custom", () => {
    expect(inferVaultRoot(validSettings({
      dbPath: "/tmp/shared-fidx.db",
      indexedRoot: "/Vault/Projects",
    }))).toBe("/Vault/Projects");
  });
});

describe("validateFidxSettings", () => {
  it("accepts valid settings and returns the derived root prefix", () => {
    const result = validateFidxSettings(validSettings({ indexedRoot: "/Vault/Projects" }), validEnv);

    expect(result.valid).toBe(true);
    expect(result.issues).toEqual([]);
    expect(result.derivedRootPrefix).toBe("Projects");
  });

  it("reports empty and missing fidx executables", () => {
    expect(validateFidxSettings(validSettings({ fidxPath: "" }), validEnv).issues)
      .toContainEqual(expect.objectContaining({ code: "fidx-path-empty" }));

    const missingEnv: SettingsValidationEnv = {
      ...validEnv,
      fileExists: (path) => path !== "/missing/fidx",
    };
    expect(validateFidxSettings(validSettings({ fidxPath: "/missing/fidx" }), missingEnv).issues)
      .toContainEqual(expect.objectContaining({ code: "fidx-path-missing" }));

    const notExecutableEnv: SettingsValidationEnv = {
      ...validEnv,
      isExecutable: (path) => path !== "/not-executable/fidx",
    };
    expect(validateFidxSettings(validSettings({ fidxPath: "/not-executable/fidx" }), notExecutableEnv).issues)
      .toContainEqual(expect.objectContaining({ code: "fidx-path-not-executable" }));
  });

  it("reports invalid indexed roots and out-of-vault roots", () => {
    const notDirEnv: SettingsValidationEnv = {
      ...validEnv,
      isDirectory: (path) => path !== "/Vault/Missing",
    };
    expect(validateFidxSettings(validSettings({ indexedRoot: "/Vault/Missing" }), notDirEnv).issues)
      .toContainEqual(expect.objectContaining({ code: "indexed-root-invalid" }));

    expect(validateFidxSettings(validSettings({ indexedRoot: "/Other" }), validEnv).issues)
      .toContainEqual(expect.objectContaining({ code: "indexed-root-outside-vault" }));

    expect(validateFidxSettings(validSettings({ indexedRoot: "/Vault/../../etc" }), validEnv).issues)
      .toContainEqual(expect.objectContaining({ code: "indexed-root-outside-vault" }));

    expect(validateFidxSettings(validSettings({
      dbPath: "/tmp/shared-fidx.db",
      indexedRoot: "/Other",
    }), validEnv, "/Vault").issues)
      .toContainEqual(expect.objectContaining({ code: "indexed-root-outside-vault" }));
  });

  it("reports invalid globs, result limits, and truncate policies", () => {
    const result = validateFidxSettings(validSettings({
      includeGlobs: [],
      excludeGlobs: [""],
      resultLimit: 0,
      truncatePolicy: "calibrated",
    }), validEnv);

    expect(result.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: "include-globs-empty" }),
      expect.objectContaining({ code: "exclude-globs-invalid" }),
      expect.objectContaining({ code: "default-excludes-missing" }),
      expect.objectContaining({ code: "result-limit-out-of-range" }),
      expect.objectContaining({ code: "truncate-policy-unsupported" }),
    ]));

    expect(validateFidxSettings(validSettings({ resultLimit: 101 }), validEnv).issues)
      .toContainEqual(expect.objectContaining({ code: "result-limit-out-of-range" }));
    expect(validateFidxSettings(validSettings({ resultLimit: 10.5 }), validEnv).issues)
      .toContainEqual(expect.objectContaining({ code: "result-limit-out-of-range" }));
    expect(validateFidxSettings(validSettings({ resultLimit: 100 }), validEnv).issues)
      .not.toContainEqual(expect.objectContaining({ code: "result-limit-out-of-range" }));
    expect(validateFidxSettings(validSettings({ truncatePolicy: "off" }), validEnv).issues)
      .not.toContainEqual(expect.objectContaining({ code: "truncate-policy-unsupported" }));
    expect(validateFidxSettings(validSettings({ truncatePolicy: "knee" }), validEnv).issues)
      .not.toContainEqual(expect.objectContaining({ code: "truncate-policy-unsupported" }));
  });

  it("allows removing default excludes only with an explicit opt-out", () => {
    const withoutOptOut = validateFidxSettings(validSettings({ excludeGlobs: [] }), validEnv);
    const withOptOut = validateFidxSettings(validSettings({
      excludeGlobs: [],
      allowHiddenDirectoryIndexing: true,
    }), validEnv);

    expect(withoutOptOut.issues)
      .toContainEqual(expect.objectContaining({ code: "default-excludes-missing" }));
    expect(withOptOut.issues)
      .not.toContainEqual(expect.objectContaining({ code: "default-excludes-missing" }));

    expect(validateFidxSettings(validSettings({
      excludeGlobs: [".obsidian/**"],
    }), validEnv).issues)
      .toContainEqual(expect.objectContaining({ code: "default-excludes-missing" }));
  });
});
