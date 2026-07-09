// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { FidxSearchResult } from "./fidx-service";
import { FidxSettings, normalizeSettingsPath } from "./settings";
import { deriveRootPrefix } from "./settings-validation";

export interface VaultFile {
  path: string;
}

export interface VaultFileLookup {
  getFileByPath(path: string): VaultFile | null;
}

export type VaultPathMapFailureReason =
  | "collection-mismatch"
  | "root-outside-vault"
  | "path-traversal"
  | "stale-file";

export type VaultPathMapResult =
  | {
      ok: true;
      vaultPath: string;
      rootPrefix: string;
      file: VaultFile;
    }
  | {
      ok: false;
      reason: VaultPathMapFailureReason;
      vaultPath?: string;
      message: string;
    };

function normalizeRelpath(relpath: string): string | null {
  const normalized = normalizeSettingsPath(relpath.trim());
  if (!normalized || normalized.startsWith("/") || /^[A-Za-z]:\//.test(normalized)) {
    return null;
  }
  const parts = normalized.split("/").filter((part) => part.length > 0);
  if (parts.some((part) => part === "." || part === "..")) {
    return null;
  }
  return parts.join("/");
}

function joinVaultPath(rootPrefix: string, relpath: string): string {
  if (!rootPrefix) {
    return relpath;
  }
  return `${rootPrefix}/${relpath}`;
}

export class VaultPathMapper {
  constructor(
    private readonly vaultRoot: string,
    private readonly vault: VaultFileLookup,
  ) {}

  mapResult(settings: FidxSettings, result: FidxSearchResult): VaultPathMapResult {
    if (result.collection !== settings.collectionName) {
      return {
        ok: false,
        reason: "collection-mismatch",
        message: `Result collection ${result.collection} does not match ${settings.collectionName}.`,
      };
    }

    let rootPrefix: string;
    try {
      rootPrefix = deriveRootPrefix(this.vaultRoot, settings.indexedRoot);
    } catch {
      return {
        ok: false,
        reason: "root-outside-vault",
        message: "The indexed root is outside the active vault.",
      };
    }

    const normalizedRelpath = normalizeRelpath(result.relpath);
    if (normalizedRelpath === null) {
      return {
        ok: false,
        reason: "path-traversal",
        message: "Search result path is not a safe vault-relative path.",
      };
    }

    const vaultPath = joinVaultPath(rootPrefix, normalizedRelpath);
    const file = this.vault.getFileByPath(vaultPath);
    if (file === null) {
      return {
        ok: false,
        reason: "stale-file",
        vaultPath,
        message: "The search result is stale; re-index the vault.",
      };
    }

    return {
      ok: true,
      vaultPath,
      rootPrefix,
      file,
    };
  }
}
