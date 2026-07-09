// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { FidxOutputChunk, FidxRunResult } from "./fidx-command-runner";
import { FidxSettings } from "./settings";
import { SettingsValidationResult } from "./settings-validation";

export interface FidxPluginService {
  doctor(): Promise<FidxRunResult>;
  registerCollection(): Promise<FidxRunResult>;
  index(onOutput: (chunk: FidxOutputChunk) => void, signal?: AbortSignal): Promise<FidxRunResult>;
}

export interface FidxPluginControllerDeps {
  settings: FidxSettings;
  service: FidxPluginService;
  saveSettings(): Promise<void>;
  notify(message: string): void;
  validateSettings?: () => SettingsValidationResult;
  onIndexProgress?: (progress: string) => void;
}

export function formatGlobList(globs: string[]): string {
  return globs.join("\n");
}

export function parseGlobList(value: string): string[] {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function alreadyRunningResult(): FidxRunResult {
  return {
    ok: false,
    exitCode: null,
    stdout: "",
    stderr: "",
    timedOut: false,
    category: "nonzero-exit",
    errorMessage: "Index is already running.",
  };
}

function resultMessage(result: FidxRunResult, fallback: string): string {
  return result.stdout.trim() || result.stderr.trim() || result.errorMessage || fallback;
}

function lastNonEmptyLine(value: string): string {
  const lines = value.trim().split(/\r?\n/).filter((line) => line.trim().length > 0);
  return lines.at(-1) ?? "";
}

function indexResultMessage(result: FidxRunResult): string {
  if (result.ok) {
    return "fidx index complete.";
  }
  if (result.category === "cancelled") {
    return result.errorMessage || "fidx index cancelled.";
  }
  return result.errorMessage || lastNonEmptyLine(result.stderr) || lastNonEmptyLine(result.stdout) || "fidx index failed.";
}

export class FidxPluginController {
  readonly settings: FidxSettings;
  isIndexing = false;
  indexProgress = "";
  private readonly service: FidxPluginService;
  private readonly saveSettingsFn: () => Promise<void>;
  private readonly notify: (message: string) => void;
  private readonly validateSettingsFn?: () => SettingsValidationResult;
  private readonly onIndexProgress?: (progress: string) => void;
  private indexAbortController: AbortController | null = null;

  constructor(deps: FidxPluginControllerDeps) {
    this.settings = deps.settings;
    this.service = deps.service;
    this.saveSettingsFn = deps.saveSettings;
    this.notify = deps.notify;
    this.validateSettingsFn = deps.validateSettings;
    this.onIndexProgress = deps.onIndexProgress;
  }

  async saveSettings(): Promise<void> {
    await this.saveSettingsFn();
  }

  async runDoctor(): Promise<FidxRunResult> {
    const result = await this.service.doctor();
    this.notify(resultMessage(result, result.ok ? "fidx doctor passed." : "fidx doctor failed."));
    return result;
  }

  async registerCollection(): Promise<FidxRunResult> {
    const validationFailure = this.validationFailure();
    if (validationFailure) {
      return validationFailure;
    }
    const result = await this.service.registerCollection();
    this.notify(resultMessage(result, result.ok ? "fidx collection registered." : "fidx collection registration failed."));
    return result;
  }

  async runIndex(): Promise<FidxRunResult> {
    const validationFailure = this.validationFailure();
    if (validationFailure) {
      return validationFailure;
    }
    if (this.isIndexing) {
      const result = alreadyRunningResult();
      this.notify(result.errorMessage ?? "Index is already running.");
      return result;
    }

    this.isIndexing = true;
    this.indexProgress = "";
    this.indexAbortController = new AbortController();
    try {
      const result = await this.service.index((chunk) => {
        this.indexProgress += chunk.text;
        this.onIndexProgress?.(this.indexProgress);
      }, this.indexAbortController.signal);
      this.notify(indexResultMessage(result));
      return result;
    } finally {
      this.isIndexing = false;
      this.indexAbortController = null;
    }
  }

  cancelIndex(): void {
    this.indexAbortController?.abort();
  }

  private validationFailure(): FidxRunResult | null {
    const validation = this.validateSettingsFn?.();
    if (!validation || validation.valid) {
      return null;
    }
    const message = validation.issues.map((issue) => issue.message).join("\n");
    this.notify(message);
    return {
      ok: false,
      exitCode: null,
      stdout: "",
      stderr: "",
      timedOut: false,
      category: "nonzero-exit",
      errorMessage: message,
    };
  }
}
