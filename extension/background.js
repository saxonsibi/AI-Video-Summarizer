const SETTINGS_KEY = "videoiq_extension_settings";
const DEFAULT_SETTINGS = {
  backendUrl: "http://127.0.0.1:8000"
};

function normalizeBackendUrl(url) {
  const raw = String(url || "").trim();
  if (!raw) return DEFAULT_SETTINGS.backendUrl;
  return raw.replace(/\/+$/, "");
}

async function getSettings() {
  const data = await chrome.storage.sync.get(SETTINGS_KEY);
  const row = data[SETTINGS_KEY] || DEFAULT_SETTINGS;
  return {
    backendUrl: normalizeBackendUrl(row.backendUrl)
  };
}

async function ensureDefaults() {
  const settings = await getSettings();
  await chrome.storage.sync.set({ [SETTINGS_KEY]: settings });
}

chrome.runtime.onInstalled.addListener(async () => {
  await ensureDefaults();
  await chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

chrome.runtime.onStartup.addListener(async () => {
  await ensureDefaults();
});

chrome.action.onClicked.addListener(async (tab) => {
  try {
    if (tab && tab.windowId) {
      await chrome.sidePanel.open({ windowId: tab.windowId });
    } else {
      const current = await chrome.windows.getCurrent();
      if (current.id) {
        await chrome.sidePanel.open({ windowId: current.id });
      }
    }
  } catch (error) {
    console.warn("Failed to open side panel:", error);
  }
});
