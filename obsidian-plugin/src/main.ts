// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { App, Notice, Plugin, PluginSettingTab, Setting } from "obsidian";

import { PLUGIN_ID } from "./constants";
import { createDefaultSettings, FidxSettings } from "./settings";

type AdapterWithBasePath = {
  basePath?: string;
  getBasePath?: () => string;
};

export function getVaultBasePath(app: App): string {
  const adapter = app.vault.adapter as AdapterWithBasePath;
  if (typeof adapter.getBasePath === "function") {
    return adapter.getBasePath();
  }
  if (typeof adapter.basePath === "string") {
    return adapter.basePath;
  }
  return "";
}

export default class FidxSearchPlugin extends Plugin {
  settings: FidxSettings = createDefaultSettings("");

  async onload(): Promise<void> {
    await this.loadSettings();
    this.addSettingTab(new FidxSettingTab(this.app, this));

    this.addCommand({
      id: "doctor",
      name: "Run fidx doctor",
      callback: () => new Notice("fidx doctor integration is not implemented yet."),
    });
    this.addCommand({
      id: "register-vault",
      name: "Register vault collection",
      callback: () => new Notice("fidx collection registration is not implemented yet."),
    });
    this.addCommand({
      id: "index-vault",
      name: "Index vault",
      callback: () => new Notice("fidx indexing is not implemented yet."),
    });
    this.addCommand({
      id: "search-vault",
      name: "Search vault",
      callback: () => new Notice("fidx search is not implemented yet."),
    });
  }

  async loadSettings(): Promise<void> {
    const defaults = createDefaultSettings(getVaultBasePath(this.app));
    this.settings = Object.assign(defaults, await this.loadData());
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }
}

class FidxSettingTab extends PluginSettingTab {
  plugin: FidxSearchPlugin;

  constructor(app: App, plugin: FidxSearchPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.addClass(`${PLUGIN_ID}-settings`);

    new Setting(containerEl)
      .setName("fidx executable")
      .setDesc("Absolute path to the fidx command.")
      .addText((text) => text
        .setPlaceholder("/usr/local/bin/fidx")
        .setValue(this.plugin.settings.fidxPath)
        .onChange(async (value) => {
          this.plugin.settings.fidxPath = value.trim();
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName("Database path")
      .setDesc("Plugin-owned fidx database for this vault.")
      .addText((text) => text
        .setValue(this.plugin.settings.dbPath)
        .onChange(async (value) => {
          this.plugin.settings.dbPath = value.trim();
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName("Collection")
      .setDesc("Collection name passed to every fidx index and search command.")
      .addText((text) => text
        .setValue(this.plugin.settings.collectionName)
        .onChange(async (value) => {
          this.plugin.settings.collectionName = value.trim();
          await this.plugin.saveSettings();
        }));
  }
}
