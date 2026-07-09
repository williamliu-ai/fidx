// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { execFile, spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";

export type FidxRunCategory =
  | "ok"
  | "not-found"
  | "timeout"
  | "output-limit"
  | "nonzero-exit"
  | "spawn-error";

export interface FidxCommand {
  executable: string;
  args: string[];
  cwd?: string;
  timeoutMs: number;
  maxOutputBytes: number;
}

export interface FidxOutputChunk {
  stream: "stdout" | "stderr";
  text: string;
}

export interface FidxRunResult {
  ok: boolean;
  exitCode: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  category: FidxRunCategory;
  errorMessage?: string;
}

type ExecFileError = Error & {
  code?: string | number;
  killed?: boolean;
  signal?: NodeJS.Signals;
};

function notFoundMessage(executable: string): string {
  return `Could not run ${executable}. Set an absolute path to the fidx executable; GUI-launched Obsidian may not inherit your terminal PATH.`;
}

function outputLimitMessage(maxOutputBytes: number): string {
  return `fidx produced more than ${maxOutputBytes} bytes of output.`;
}

function result(
  category: FidxRunCategory,
  stdout: string,
  stderr: string,
  fields: Partial<FidxRunResult> = {},
): FidxRunResult {
  const ok = category === "ok";
  return {
    ok,
    exitCode: ok ? 0 : null,
    stdout,
    stderr,
    timedOut: false,
    category,
    ...fields,
  };
}

function mapExecError(error: ExecFileError, command: FidxCommand, stdout: string, stderr: string): FidxRunResult {
  if (error.code === "ENOENT") {
    return result("not-found", stdout, stderr, {
      errorMessage: notFoundMessage(command.executable),
    });
  }
  if (error.killed && error.signal === "SIGTERM") {
    return result("timeout", stdout, stderr, {
      timedOut: true,
      errorMessage: `fidx timed out after ${command.timeoutMs}ms.`,
    });
  }
  if (error.code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER" || error.message.includes("maxBuffer")) {
    return result("output-limit", stdout, stderr, {
      errorMessage: outputLimitMessage(command.maxOutputBytes),
    });
  }
  if (typeof error.code === "number") {
    return result("nonzero-exit", stdout, stderr, {
      exitCode: error.code,
      errorMessage: error.message,
    });
  }
  return result("spawn-error", stdout, stderr, {
    errorMessage: error.message,
  });
}

export class FidxCommandRunner {
  runBuffered(command: FidxCommand): Promise<FidxRunResult> {
    return new Promise((resolve) => {
      execFile(command.executable, command.args, {
        cwd: command.cwd,
        encoding: "utf8",
        maxBuffer: command.maxOutputBytes,
        timeout: command.timeoutMs,
        windowsHide: true,
      }, (error, stdout, stderr) => {
        if (!error) {
          resolve(result("ok", stdout, stderr, { exitCode: 0 }));
          return;
        }
        resolve(mapExecError(error as ExecFileError, command, stdout, stderr));
      });
    });
  }

  runStreaming(
    command: FidxCommand,
    onOutput?: (chunk: FidxOutputChunk) => void,
  ): Promise<FidxRunResult> {
    return new Promise((resolve) => {
      const child = spawn(command.executable, command.args, {
        cwd: command.cwd,
        windowsHide: true,
      });
      let stdout = "";
      let stderr = "";
      let totalBytes = 0;
      let timedOut = false;
      let outputLimited = false;
      let settled = false;
      let timer: NodeJS.Timeout | undefined;
      const stdoutDecoder = new StringDecoder("utf8");
      const stderrDecoder = new StringDecoder("utf8");

      const settle = (value: FidxRunResult) => {
        if (settled) {
          return;
        }
        settled = true;
        if (timer) {
          clearTimeout(timer);
        }
        resolve(value);
      };

      const armTimer = () => {
        if (command.timeoutMs <= 0) {
          return;
        }
        if (timer) {
          clearTimeout(timer);
        }
        timer = setTimeout(() => {
          timedOut = true;
          child.kill();
        }, command.timeoutMs);
      };

      const emitText = (stream: "stdout" | "stderr", text: string) => {
        if (!text) {
          return;
        }
        if (stream === "stdout") {
          stdout += text;
        } else {
          stderr += text;
        }
        onOutput?.({ stream, text });
      };

      const append = (stream: "stdout" | "stderr", chunk: Buffer) => {
        totalBytes += chunk.length;
        if (totalBytes > command.maxOutputBytes) {
          outputLimited = true;
          child.kill();
          return;
        }
        const decoder = stream === "stdout" ? stdoutDecoder : stderrDecoder;
        emitText(stream, decoder.write(chunk));
        armTimer();
      };

      armTimer();
      child.stdout.on("data", (chunk: Buffer) => append("stdout", chunk));
      child.stderr.on("data", (chunk: Buffer) => append("stderr", chunk));
      child.on("error", (error: NodeJS.ErrnoException) => {
        if (error.code === "ENOENT") {
          settle(result("not-found", stdout, stderr, {
            errorMessage: notFoundMessage(command.executable),
          }));
          return;
        }
        settle(result("spawn-error", stdout, stderr, {
          errorMessage: error.message,
        }));
      });
      child.on("close", (code) => {
        emitText("stdout", stdoutDecoder.end());
        emitText("stderr", stderrDecoder.end());
        if (timedOut) {
          settle(result("timeout", stdout, stderr, {
            timedOut: true,
            errorMessage: `fidx timed out after ${command.timeoutMs}ms.`,
          }));
          return;
        }
        if (outputLimited) {
          settle(result("output-limit", stdout, stderr, {
            errorMessage: outputLimitMessage(command.maxOutputBytes),
          }));
          return;
        }
        if (code === 0) {
          settle(result("ok", stdout, stderr, { exitCode: 0 }));
          return;
        }
        settle(result("nonzero-exit", stdout, stderr, {
          exitCode: code,
          errorMessage: `fidx exited with code ${code}.`,
        }));
      });
    });
  }
}
