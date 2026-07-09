// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import {
  FidxCommand,
  FidxCommandRunner,
  FidxOutputChunk,
  FidxRunResult,
} from "./fidx-command-runner";
import { FidxSettings } from "./settings";

export interface FidxSearchResult {
  path: string;
  collection: string;
  relpath: string;
  docid?: string;
  title?: string;
  snippet?: string;
  score?: number;
  [key: string]: unknown;
}

export interface FidxSearchEnvelope {
  schema: "fidx.search.v2";
  results: FidxSearchResult[];
  diagnostics?: unknown;
  summary?: unknown;
  next_actions?: unknown;
  [key: string]: unknown;
}

function command(settings: FidxSettings, args: string[], timeoutMs = settings.searchTimeoutMs): FidxCommand {
  return {
    executable: settings.fidxPath,
    args,
    timeoutMs,
    maxOutputBytes: settings.maxOutputBytes,
  };
}

function withDb(settings: FidxSettings, args: string[]): string[] {
  return ["--db", settings.dbPath, ...args];
}

function appendRepeated(args: string[], flag: string, values: string[]): string[] {
  for (const value of values) {
    args.push(flag, value);
  }
  return args;
}

export function buildDoctorCommand(settings: FidxSettings): FidxCommand {
  return command(settings, ["doctor"]);
}

export function buildCollectionAddCommand(settings: FidxSettings): FidxCommand {
  const args = withDb(settings, [
    "collection",
    "add",
    settings.indexedRoot,
    "--name",
    settings.collectionName,
  ]);
  appendRepeated(args, "--glob", settings.includeGlobs);
  appendRepeated(args, "--exclude-glob", settings.excludeGlobs);
  return command(settings, args);
}

export function buildIndexCommand(settings: FidxSettings): FidxCommand {
  return command(
    settings,
    withDb(settings, ["index", "-c", settings.collectionName]),
    settings.indexInactivityTimeoutMs,
  );
}

export function buildSearchCommand(settings: FidxSettings, query: string): FidxCommand {
  const args = withDb(settings, [
    "search",
    "-c",
    settings.collectionName,
    "--json",
    "-n",
    String(settings.resultLimit),
  ]);
  const truncatePolicy = settings.truncatePolicy.trim();
  if (truncatePolicy && truncatePolicy !== "off") {
    args.push("--truncate", truncatePolicy);
  }
  args.push("--", query);
  return command(settings, args);
}

export function buildStatusCommand(settings: FidxSettings): FidxCommand {
  return command(settings, withDb(settings, ["status"]));
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireString(value: Record<string, unknown>, key: string, index: number): string {
  const candidate = value[key];
  if (typeof candidate !== "string") {
    throw new Error(`search result ${index} is missing string field ${key}`);
  }
  return candidate;
}

export function parseSearchEnvelope(stdout: string): FidxSearchEnvelope {
  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout);
  } catch (error) {
    throw new Error(`fidx search did not return valid JSON: ${(error as Error).message}`);
  }
  if (!isObject(parsed)) {
    throw new Error("fidx search JSON must be an object");
  }
  if (parsed.schema !== "fidx.search.v2") {
    throw new Error("fidx search JSON schema must be fidx.search.v2");
  }
  if (!Array.isArray(parsed.results)) {
    throw new Error("fidx search JSON must include a results array");
  }

  const results = parsed.results.map((entry, index): FidxSearchResult => {
    if (!isObject(entry)) {
      throw new Error(`search result ${index} must be an object`);
    }
    return {
      ...entry,
      path: requireString(entry, "path", index),
      collection: requireString(entry, "collection", index),
      relpath: requireString(entry, "relpath", index),
    };
  });

  return {
    ...parsed,
    schema: "fidx.search.v2",
    results,
  };
}

export function parseStatusOutput(stdout: string): string {
  return stdout;
}

export class FidxService {
  constructor(
    private readonly settings: FidxSettings,
    private readonly runner = new FidxCommandRunner(),
  ) {}

  doctor(): Promise<FidxRunResult> {
    return this.runner.runBuffered(buildDoctorCommand(this.settings));
  }

  registerCollection(): Promise<FidxRunResult> {
    return this.runner.runBuffered(buildCollectionAddCommand(this.settings));
  }

  index(onOutput?: (chunk: FidxOutputChunk) => void, signal?: AbortSignal): Promise<FidxRunResult> {
    return this.runner.runStreaming(buildIndexCommand(this.settings), onOutput, signal);
  }

  search(query: string): Promise<FidxRunResult> {
    return this.runner.runBuffered(buildSearchCommand(this.settings, query));
  }

  status(): Promise<FidxRunResult> {
    return this.runner.runBuffered(buildStatusCommand(this.settings));
  }
}
