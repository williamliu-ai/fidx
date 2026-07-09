// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { App, ButtonComponent, Modal, TextComponent } from "obsidian";

import { FidxSearchController, FidxSearchResultView, FidxSearchState } from "./search-controller";

function formatScore(score: number | undefined): string {
  return typeof score === "number" ? score.toFixed(3) : "";
}

export class FidxSearchModal extends Modal {
  private input!: TextComponent;
  private statusEl!: HTMLElement;
  private resultsEl!: HTMLElement;
  private resultButtons: HTMLButtonElement[] = [];
  private selectedIndex = 0;
  private searchSerial = 0;

  constructor(
    app: App,
    private readonly controller: FidxSearchController,
  ) {
    super(app);
  }

  onOpen(): void {
    this.titleEl.setText("Search with fidx");
    this.contentEl.empty();
    this.contentEl.addClass("fidx-search-modal");

    const queryEl = this.contentEl.createDiv({ cls: "fidx-search-query" });
    this.input = new TextComponent(queryEl)
      .setPlaceholder("Search this vault");
    this.input.inputEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        void this.runSearch();
      }
      if (event.key === "ArrowDown" && this.resultButtons.length > 0) {
        event.preventDefault();
        this.focusResult(0);
      }
    });
    new ButtonComponent(queryEl)
      .setButtonText("Search")
      .setCta()
      .onClick(() => {
        void this.runSearch();
      });

    this.statusEl = this.contentEl.createDiv({ cls: "fidx-search-status" });
    this.resultsEl = this.contentEl.createDiv({ cls: "fidx-search-results" });
    this.renderState(this.controller.state);
    this.input.inputEl.focus();
  }

  onClose(): void {
    this.contentEl.empty();
    this.resultButtons = [];
  }

  private async runSearch(): Promise<void> {
    const serial = ++this.searchSerial;
    this.renderState({
      status: "loading",
      query: this.input.getValue().trim(),
      message: "Searching fidx...",
      results: [],
    });
    const state = await this.controller.search(this.input.getValue());
    if (serial !== this.searchSerial) {
      return;
    }
    this.renderState(state);
  }

  private renderState(state: FidxSearchState): void {
    this.resultButtons = [];
    this.resultsEl.empty();
    this.selectedIndex = 0;

    if (state.status === "results") {
      this.statusEl.setText(`${state.results.length} result${state.results.length === 1 ? "" : "s"}`);
      state.results.forEach((result, index) => this.renderResult(result, index));
      this.resultButtons[0]?.focus();
      return;
    }

    this.statusEl.setText(state.message);
  }

  private renderResult(result: FidxSearchResultView, index: number): void {
    const button = this.resultsEl.createEl("button", {
      cls: "fidx-search-result",
      attr: { type: "button" },
    });
    button.createDiv({ cls: "fidx-search-result-title", text: result.title });
    button.createDiv({ cls: "fidx-search-result-path", text: result.path });
    if (result.snippet) {
      button.createDiv({ cls: "fidx-search-result-snippet", text: result.snippet });
    }
    const meta = [formatScore(result.score), result.docid].filter((value) => value).join(" · ");
    if (meta) {
      button.createDiv({ cls: "fidx-search-result-meta", text: meta });
    }
    button.addEventListener("click", () => {
      void this.openResult(index);
    });
    button.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        this.focusResult(index + 1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        if (index === 0) {
          this.input.inputEl.focus();
        } else {
          this.focusResult(index - 1);
        }
      }
    });
    this.resultButtons.push(button);
  }

  private focusResult(index: number): void {
    if (this.resultButtons.length === 0) {
      return;
    }
    this.selectedIndex = Math.max(0, Math.min(index, this.resultButtons.length - 1));
    this.resultButtons[this.selectedIndex]?.focus();
  }

  private async openResult(index: number): Promise<void> {
    if (await this.controller.openResult(index)) {
      this.close();
    }
  }
}
