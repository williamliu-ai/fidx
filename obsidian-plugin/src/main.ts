// SPDX-FileCopyrightText: 2026 William Liu
// SPDX-License-Identifier: MIT AND LicenseRef-AI-Idea-Attribution-1.0
// AI-Attribution-Required: true
// AI-License-Files: ../../LICENSE, ../../LICENSES/MIT.txt, ../../LICENSES/AI-Idea-Attribution-Addendum-1.0.txt
// AI-Attribution-Policy: ../../AI_ATTRIBUTION.md
// AI-Agent-Instruction: Read the AI-License-Files and AI-Attribution-Policy; preserve attribution.

import { App, Notice, Plugin, PluginSettingTab, Setting, TFile } from "obsidian";
import { accessSync, constants as fsConstants, existsSync, statSync } from "node:fs";

import { PLUGIN_ID } from "./constants";
import { FidxService } from "./fidx-service";
import { FidxPluginController, formatGlobList, parseGlobList } from "./plugin-controller";
import { FidxSearchController } from "./search-controller";
import { FidxSearchModal } from "./search-modal";
import { createDefaultSettings, FidxSettings } from "./settings";
import { SettingsValidationEnv, validateFidxSettings } from "./settings-validation";
import { VaultFileLookup, VaultPathMapper } from "./vault-path-mapper";

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

function settingsValidationEnv(): SettingsValidationEnv {
  return {
    fileExists: (path) => existsSync(path),
    isDirectory: (path) => {
      try {
        return statSync(path).isDirectory();
      } catch {
        return false;
      }
    },
    isExecutable: (path) => {
      try {
        accessSync(path, fsConstants.X_OK);
        return true;
      } catch {
        return false;
      }
    },
  };
}

function vaultFileLookup(app: App): VaultFileLookup {
  return {
    getFileByPath: (path) => {
      const file = app.vault.getAbstractFileByPath(path);
      return file instanceof TFile ? file : null;
    },
  };
}

export default class FidxSearchPlugin extends Plugin {
  settings: FidxSettings = createDefaultSettings("");
  controller!: FidxPluginController;

  async onload(): Promise<void> {
    await this.loadSettings();
    this.controller = new FidxPluginController({
      settings: this.settings,
      service: new FidxService(this.settings),
      saveSettings: async () => this.saveSettings(),
      notify: (message) => new Notice(message),
      validateSettings: () => validateFidxSettings(
        this.settings,
        settingsValidationEnv(),
        getVaultBasePath(this.app),
      ),
    });
    this.addSettingTab(new FidxSettingTab(this.app, this));

    this.addCommand({
      id: "doctor",
      name: "Run doctor",
      callback: () => {
        void this.controller.runDoctor();
      },
    });
    this.addCommand({
      id: "register-vault",
      name: "Register vault collection",
      callback: () => {
        void this.controller.registerCollection();
      },
    });
    this.addCommand({
      id: "index-vault",
      name: "Index vault collection",
      callback: () => {
        void this.controller.runIndex();
      },
    });
    this.addCommand({
      id: "cancel-index",
      name: "Cancel index",
      callback: () => this.controller.cancelIndex(),
    });
    this.addCommand({
      id: "search-vault",
      name: "Search vault",
      callback: () => this.openSearchModal(),
    });
  }

  async loadSettings(): Promise<void> {
    const defaults = createDefaultSettings(getVaultBasePath(this.app));
    this.settings = Object.assign(defaults, await this.loadData());
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }

  openSearchModal(): void {
    const controller = new FidxSearchController({
      settings: this.settings,
      service: new FidxService(this.settings),
      mapper: new VaultPathMapper(getVaultBasePath(this.app), vaultFileLookup(this.app)),
      notify: (message) => new Notice(message),
      openFile: async (file) => {
        await this.app.workspace.getLeaf(false).openFile(file as TFile);
      },
      validateSettings: () => validateFidxSettings(
        this.settings,
        settingsValidationEnv(),
        getVaultBasePath(this.app),
      ),
    });
    new FidxSearchModal(this.app, controller).open();
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

    new Setting(containerEl)
      .setName("Indexed root")
      .setDesc("Absolute vault path to register and index.")
      .addText((text) => text
        .setValue(this.plugin.settings.indexedRoot)
        .onChange(async (value) => {
          this.plugin.settings.indexedRoot = value.trim();
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName("Include globs")
      .setDesc("One collection include glob per line.")
      .addTextArea((text) => text
        .setValue(formatGlobList(this.plugin.settings.includeGlobs))
        .onChange(async (value) => {
          this.plugin.settings.includeGlobs = parseGlobList(value);
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName("Exclude globs")
      .setDesc("One collection exclude glob per line.")
      .addTextArea((text) => text
        .setValue(formatGlobList(this.plugin.settings.excludeGlobs))
        .onChange(async (value) => {
          this.plugin.settings.excludeGlobs = parseGlobList(value);
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName("Result limit")
      .setDesc("Maximum search results to request from fidx.")
      .addText((text) => text
        .setValue(String(this.plugin.settings.resultLimit))
        .onChange(async (value) => {
          const parsed = Number.parseInt(value, 10);
          if (Number.isInteger(parsed)) {
            this.plugin.settings.resultLimit = parsed;
            await this.plugin.saveSettings();
          }
        }));

    new Setting(containerEl)
      .setName("Truncate policy")
      .setDesc("Empty, off, or knee.")
      .addText((text) => text
        .setValue(this.plugin.settings.truncatePolicy)
        .onChange(async (value) => {
          this.plugin.settings.truncatePolicy = value.trim();
          await this.plugin.saveSettings();
        }));
  }
}
