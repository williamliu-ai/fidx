// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { describe, expect, it, vi } from "vitest";

import { FidxRunResult } from "../src/fidx-command-runner";
import { createDefaultSettings } from "../src/settings";
import {
  FidxPluginController,
  FidxPluginControllerDeps,
  FidxPluginService,
  formatGlobList,
  parseGlobList,
} from "../src/plugin-controller";
import { SettingsValidationResult } from "../src/settings-validation";

function ok(stdout = ""): FidxRunResult {
  return {
    ok: true,
    exitCode: 0,
    stdout,
    stderr: "",
    timedOut: false,
    category: "ok",
  };
}

function service(overrides: Partial<FidxPluginService> = {}): FidxPluginService {
  return {
    doctor: vi.fn(async () => ok("doctor ok")),
    registerCollection: vi.fn(async () => ok("registered")),
    index: vi.fn(async () => ok("indexed")),
    ...overrides,
  };
}

function controller(
  overrides: Partial<FidxPluginService> = {},
  validation?: SettingsValidationResult,
  extraDeps: Partial<FidxPluginControllerDeps> = {},
) {
  const notifications: string[] = [];
  const settings = {
    ...createDefaultSettings("/Vault"),
    fidxPath: "/usr/local/bin/fidx",
  };
  const mockService = service(overrides);
  const instance = new FidxPluginController({
    settings,
    service: mockService,
    saveSettings: vi.fn(async () => undefined),
    notify: (message) => notifications.push(message),
    validateSettings: validation ? () => validation : undefined,
    ...extraDeps,
  });
  return { instance, mockService, notifications, settings };
}

describe("glob list helpers", () => {
  it("formats and parses user-editable glob lists", () => {
    expect(formatGlobList(["**/*.md", ".obsidian/**"])).toBe("**/*.md\n.obsidian/**");
    expect(parseGlobList(" **/*.md \n\n.obsidian/** ")).toEqual(["**/*.md", ".obsidian/**"]);
  });
});

describe("FidxPluginController command wiring", () => {
  it("runs doctor and register commands through the service", async () => {
    const { instance, mockService, notifications } = controller();

    await instance.runDoctor();
    await instance.registerCollection();

    expect(mockService.doctor).toHaveBeenCalledOnce();
    expect(mockService.registerCollection).toHaveBeenCalledOnce();
    expect(notifications).toEqual(["doctor ok", "registered"]);
  });

  it("streams index progress and prevents concurrent index runs", async () => {
    let releaseIndex!: () => void;
    const indexPromise = new Promise<FidxRunResult>((resolve) => {
      releaseIndex = () => resolve(ok("indexed"));
    });
    const { instance, mockService, notifications } = controller({
      index: vi.fn(async (onOutput) => {
        onOutput({ stream: "stderr", text: "embedded 1/2\n" });
        return indexPromise;
      }),
    });

    const first = instance.runIndex();
    const second = await instance.runIndex();
    releaseIndex();
    const firstResult = await first;

    expect(mockService.index).toHaveBeenCalledOnce();
    expect(second?.category).toBe("nonzero-exit");
    expect(firstResult.category).toBe("ok");
    expect(instance.indexProgress).toBe("embedded 1/2\n");
    expect(notifications).toContain("Index is already running.");
  });

  it("emits streaming index progress to the progress callback", async () => {
    const progress: string[] = [];
    const { instance } = controller({
      index: vi.fn(async (onOutput) => {
        onOutput({ stream: "stderr", text: "embedded 1/2\n" });
        onOutput({ stream: "stderr", text: "embedded 2/2\n" });
        return ok("indexed");
      }),
    }, undefined, {
      onIndexProgress: (value) => progress.push(value),
    });

    await instance.runIndex();

    expect(progress).toEqual([
      "embedded 1/2\n",
      "embedded 1/2\nembedded 2/2\n",
    ]);
  });

  it("uses concise index notifications instead of dumping streamed output", async () => {
    const streamedLog = "embedded 1/500\n".repeat(500);
    const { instance, notifications } = controller({
      index: vi.fn(async (onOutput) => {
        onOutput({ stream: "stderr", text: streamedLog });
        return {
          ok: true,
          exitCode: 0,
          stdout: "",
          stderr: streamedLog,
          timedOut: false,
          category: "ok" as const,
        };
      }),
    });

    await instance.runIndex();

    expect(notifications.at(-1)).toBe("fidx index complete.");
  });

  it("passes an abort signal and can cancel a running index", async () => {
    let capturedSignal: AbortSignal | undefined;
    const { instance } = controller({
      index: vi.fn(async (_onOutput, signal) => {
        capturedSignal = signal;
        await new Promise((resolve) => setTimeout(resolve, 10));
        return {
          ok: false,
          exitCode: null,
          stdout: "",
          stderr: "",
          timedOut: false,
          category: "cancelled" as const,
        };
      }),
    });

    const running = instance.runIndex();
    instance.cancelIndex();
    await running;

    expect(capturedSignal?.aborted).toBe(true);
    expect(instance.isIndexing).toBe(false);
  });

  it("surfaces the cancellation message instead of streamed partial output", async () => {
    const { instance, notifications } = controller({
      index: vi.fn(async (onOutput) => {
        onOutput({ stream: "stderr", text: "embedded 3/500\n" });
        return {
          ok: false,
          exitCode: null,
          stdout: "",
          stderr: "embedded 3/500\n",
          timedOut: false,
          category: "cancelled" as const,
          errorMessage: "fidx command was cancelled.",
        };
      }),
    });

    await instance.runIndex();

    expect(notifications.at(-1)).toBe("fidx command was cancelled.");
  });

  it("can run index again after a completed run", async () => {
    const index = vi.fn(async () => ok("indexed"));
    const { instance, mockService } = controller({ index });

    await instance.runIndex();
    await instance.runIndex();

    expect(mockService.index).toHaveBeenCalledTimes(2);
    expect(instance.isIndexing).toBe(false);
  });

  it("does not register or index when settings validation fails", async () => {
    const invalid: SettingsValidationResult = {
      valid: false,
      derivedRootPrefix: "",
      issues: [{
        code: "fidx-path-empty",
        field: "fidxPath",
        message: "Set an absolute path to the fidx executable.",
      }],
    };
    const { instance, mockService, notifications } = controller({}, invalid);

    await instance.registerCollection();
    await instance.runIndex();

    expect(mockService.registerCollection).not.toHaveBeenCalled();
    expect(mockService.index).not.toHaveBeenCalled();
    expect(notifications).toEqual([
      "Set an absolute path to the fidx executable.",
      "Set an absolute path to the fidx executable.",
    ]);
  });
});
