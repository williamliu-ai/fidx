// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { describe, expect, it } from "vitest";

import { FidxCommandRunner } from "../src/fidx-command-runner";

const runner = new FidxCommandRunner();

function nodeCommand(script: string, overrides = {}) {
  return {
    executable: process.execPath,
    args: ["-e", script],
    timeoutMs: 1_000,
    maxOutputBytes: 32_768,
    ...overrides,
  };
}

describe("FidxCommandRunner buffered execution", () => {
  it("executes argv without a shell and captures stdout", async () => {
    const result = await runner.runBuffered(nodeCommand("process.stdout.write('ok')"));

    expect(result).toMatchObject({
      ok: true,
      exitCode: 0,
      stdout: "ok",
      stderr: "",
      category: "ok",
    });
  });

  it("categorizes nonzero exits", async () => {
    const result = await runner.runBuffered(nodeCommand(
      "process.stderr.write('bad'); process.exit(7)",
    ));

    expect(result).toMatchObject({
      ok: false,
      exitCode: 7,
      stderr: "bad",
      category: "nonzero-exit",
    });
  });

  it("categorizes missing executables with GUI PATH guidance", async () => {
    const result = await runner.runBuffered({
      executable: "/definitely/missing/fidx",
      args: ["doctor"],
      timeoutMs: 1_000,
      maxOutputBytes: 32_768,
    });

    expect(result.ok).toBe(false);
    expect(result.category).toBe("not-found");
    expect(result.errorMessage).toMatch(/absolute path/i);
  });

  it("categorizes timeouts", async () => {
    const result = await runner.runBuffered(nodeCommand(
      "setTimeout(() => process.stdout.write('late'), 1000)",
      { timeoutMs: 20 },
    ));

    expect(result.ok).toBe(false);
    expect(result.timedOut).toBe(true);
    expect(result.category).toBe("timeout");
  });

  it("categorizes output larger than the configured cap", async () => {
    const result = await runner.runBuffered(nodeCommand(
      "process.stdout.write('x'.repeat(10000))",
      { maxOutputBytes: 128 },
    ));

    expect(result.ok).toBe(false);
    expect(result.category).toBe("output-limit");
  });
});

describe("FidxCommandRunner streaming execution", () => {
  it("streams index-style output without using maxBuffer", async () => {
    const chunks: string[] = [];
    const result = await runner.runStreaming(
      nodeCommand("process.stderr.write('step 1\\n'); process.stdout.write('done')"),
      (chunk) => chunks.push(`${chunk.stream}:${chunk.text}`),
    );

    expect(result.ok).toBe(true);
    expect(result.stdout).toBe("done");
    expect(result.stderr).toBe("step 1\n");
    expect(chunks).toEqual(["stderr:step 1\n", "stdout:done"]);
  });

  it("reports not-found with GUI PATH guidance", async () => {
    const result = await runner.runStreaming({
      executable: "/definitely/missing/fidx",
      args: ["index"],
      timeoutMs: 1_000,
      maxOutputBytes: 32_768,
    });

    expect(result.category).toBe("not-found");
    expect(result.errorMessage).toMatch(/absolute path/i);
  });

  it("categorizes nonzero streaming exits", async () => {
    const result = await runner.runStreaming(
      nodeCommand("process.stderr.write('boom'); process.exit(3)"),
    );

    expect(result).toMatchObject({
      ok: false,
      exitCode: 3,
      category: "nonzero-exit",
      stderr: "boom",
    });
  });

  it("does not time out while streaming output remains active", async () => {
    const result = await runner.runStreaming(
      nodeCommand(`
        let count = 0;
        const timer = setInterval(() => {
          process.stdout.write(".");
          count += 1;
          if (count === 4) {
            clearInterval(timer);
          }
        }, 20);
      `, { timeoutMs: 50 }),
    );

    expect(result.ok).toBe(true);
    expect(result.category).toBe("ok");
  });

  it("times out after streaming output becomes inactive", async () => {
    const result = await runner.runStreaming(
      nodeCommand("process.stdout.write('first'); setTimeout(() => {}, 1000)", { timeoutMs: 20 }),
    );

    expect(result.category).toBe("timeout");
    expect(result.timedOut).toBe(true);
  });

  it("kills the stream once output exceeds the cap", async () => {
    const result = await runner.runStreaming(
      nodeCommand("setInterval(() => process.stdout.write('x'.repeat(1000)), 1)", {
        maxOutputBytes: 256,
      }),
    );

    expect(result.category).toBe("output-limit");
    expect(result.ok).toBe(false);
  });

  it("cancels a streaming command through an AbortSignal", async () => {
    const controller = new AbortController();
    const running = runner.runStreaming(
      nodeCommand("setInterval(() => process.stdout.write('.'), 5)", { timeoutMs: 0 }),
      undefined,
      controller.signal,
    );

    controller.abort();
    const result = await running;

    expect(result.ok).toBe(false);
    expect(result.category).toBe("cancelled");
  });

  it("escalates cancellation when a streaming command ignores SIGTERM", async () => {
    const controller = new AbortController();
    let markReady!: () => void;
    const ready = new Promise<void>((resolve) => {
      markReady = resolve;
    });
    const startedAt = Date.now();
    const running = runner.runStreaming(
      nodeCommand(`
        process.on('SIGTERM', () => {});
        process.stdout.write('ready\\n');
        setInterval(() => process.stdout.write('.'), 20);
        setTimeout(() => process.exit(0), 1000);
      `, {
        timeoutMs: 0,
        cancelKillAfterMs: 50,
      }),
      (chunk) => {
        if (chunk.text.includes("ready")) {
          markReady();
        }
      },
      controller.signal,
    );

    await ready;
    controller.abort();
    const result = await Promise.race([
      running,
      new Promise<never>((_, reject) => setTimeout(() => reject(new Error("cancel did not settle promptly")), 500)),
    ]);

    expect(Date.now() - startedAt).toBeLessThan(500);
    expect(result.ok).toBe(false);
    expect(result.category).toBe("cancelled");
  });
});
