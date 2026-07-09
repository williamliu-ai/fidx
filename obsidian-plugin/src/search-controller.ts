// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { FidxRunResult } from "./fidx-command-runner";
import { FidxSearchResult, parseSearchEnvelope } from "./fidx-service";
import { FidxSettings } from "./settings";
import { SettingsValidationResult } from "./settings-validation";
import { VaultFile, VaultPathMapResult } from "./vault-path-mapper";

export interface FidxSearchService {
  search(query: string): Promise<FidxRunResult>;
}

export interface FidxSearchMapper {
  mapResult(settings: FidxSettings, result: FidxSearchResult): VaultPathMapResult;
}

export interface FidxSearchResultView {
  title: string;
  path: string;
  snippet: string;
  score?: number;
  docid?: string;
  source: FidxSearchResult;
}

export type FidxSearchStatus =
  | "idle"
  | "loading"
  | "results"
  | "empty"
  | "no-index"
  | "error"
  | "setup-error";

export interface FidxSearchState {
  status: FidxSearchStatus;
  query: string;
  message: string;
  results: FidxSearchResultView[];
}

export interface FidxSearchControllerDeps {
  settings: FidxSettings;
  service: FidxSearchService;
  mapper: FidxSearchMapper;
  notify(message: string): void;
  openFile(file: VaultFile): Promise<void>;
  validateSettings?: () => SettingsValidationResult;
}

function state(
  status: FidxSearchStatus,
  fields: Partial<FidxSearchState> = {},
): FidxSearchState {
  return {
    status,
    query: "",
    message: "",
    results: [],
    ...fields,
  };
}

function commandFailureMessage(result: FidxRunResult): string {
  return result.errorMessage || result.stderr.trim() || result.stdout.trim() || "fidx search failed.";
}

function isNoIndexResult(result: FidxRunResult): boolean {
  const output = [result.errorMessage, result.stderr, result.stdout].filter(Boolean).join("\n").toLowerCase();
  return /(?:database has no index|no fidx index|not indexed|index not found|missing index)/i.test(output);
}

function validationMessage(validation: SettingsValidationResult): string {
  return validation.issues.map((issue) => issue.message).join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function resultView(result: FidxSearchResult): FidxSearchResultView {
  return {
    title: result.title?.trim() || result.relpath,
    path: result.relpath,
    snippet: result.snippet?.trim() || "",
    score: typeof result.score === "number" ? result.score : undefined,
    docid: result.docid,
    source: result,
  };
}

export class FidxSearchController {
  private readonly settings: FidxSettings;
  private readonly service: FidxSearchService;
  private readonly mapper: FidxSearchMapper;
  private readonly notify: (message: string) => void;
  private readonly openFileFn: (file: VaultFile) => Promise<void>;
  private readonly validateSettingsFn?: () => SettingsValidationResult;
  private sourceResults: FidxSearchResult[] = [];
  private searchGeneration = 0;
  state: FidxSearchState = state("idle");

  constructor(deps: FidxSearchControllerDeps) {
    this.settings = deps.settings;
    this.service = deps.service;
    this.mapper = deps.mapper;
    this.notify = deps.notify;
    this.openFileFn = deps.openFile;
    this.validateSettingsFn = deps.validateSettings;
  }

  async search(query: string): Promise<FidxSearchState> {
    const generation = ++this.searchGeneration;
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      this.sourceResults = [];
      this.state = state("idle", { message: "Enter a search query." });
      return this.state;
    }

    const validation = this.validateSettingsFn?.();
    if (validation && !validation.valid) {
      this.sourceResults = [];
      this.state = state("setup-error", {
        query: trimmedQuery,
        message: validationMessage(validation),
      });
      return this.state;
    }

    this.state = state("loading", { query: trimmedQuery, message: "Searching fidx..." });
    const result = await this.service.search(trimmedQuery);
    if (generation !== this.searchGeneration) {
      return this.state;
    }
    if (!result.ok) {
      this.sourceResults = [];
      if (isNoIndexResult(result)) {
        this.state = state("no-index", {
          query: trimmedQuery,
          message: "No fidx index found. Run fidx indexing for this vault.",
        });
        return this.state;
      }
      this.state = state("error", {
        query: trimmedQuery,
        message: commandFailureMessage(result),
      });
      return this.state;
    }

    try {
      const envelope = parseSearchEnvelope(result.stdout);
      this.sourceResults = envelope.results;
      if (envelope.results.length === 0) {
        this.state = state("empty", {
          query: trimmedQuery,
          message: "No fidx results found.",
        });
        return this.state;
      }
      this.state = state("results", {
        query: trimmedQuery,
        results: envelope.results.map(resultView),
      });
      return this.state;
    } catch (error) {
      this.sourceResults = [];
      this.state = state("error", {
        query: trimmedQuery,
        message: errorMessage(error),
      });
      return this.state;
    }
  }

  async openResult(index: number): Promise<boolean> {
    const result = this.sourceResults[index];
    if (!result) {
      return false;
    }
    const mapped = this.mapper.mapResult(this.settings, result);
    if (!mapped.ok) {
      this.notify(mapped.message);
      return false;
    }
    try {
      await this.openFileFn(mapped.file);
      return true;
    } catch (error) {
      this.notify(`Could not open search result: ${errorMessage(error)}`);
      return false;
    }
  }
}
