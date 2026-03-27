const SETTINGS_KEY = "videoiq_extension_settings";
const DEFAULT_SETTINGS = {
  backendUrl: "http://127.0.0.1:8000"
};

const appFrame = document.getElementById("appFrame");
const errorBox = document.getElementById("errorBox");
const reloadBtn = document.getElementById("reloadBtn");
const openTabBtn = document.getElementById("openTabBtn");
const settingsBtn = document.getElementById("settingsBtn");

function normalizeBackendUrl(url) {
  const raw = String(url || "").trim();
  if (!raw) return DEFAULT_SETTINGS.backendUrl;
  return raw.replace(/\/+$/, "");
}

function showError(message) {
  if (!message) {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
    return;
  }
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

async function getSettings() {
  const data = await chrome.storage.sync.get(SETTINGS_KEY);
  const row = data[SETTINGS_KEY] || DEFAULT_SETTINGS;
  return {
    backendUrl: normalizeBackendUrl(row.backendUrl)
  };
}

async function mountIframe() {
  try {
    const settings = await getSettings();
    const appUrl = `${settings.backendUrl}/`;
    appFrame.src = appUrl;
    showError("");
  } catch (error) {
    showError(`Failed to load settings: ${error?.message || error}`);
  }
}

reloadBtn.addEventListener("click", () => {
  appFrame.src = appFrame.src;
});

openTabBtn.addEventListener("click", async () => {
  const settings = await getSettings();
  const appUrl = `${settings.backendUrl}/`;
  chrome.tabs.create({ url: appUrl });
});

settingsBtn.addEventListener("click", () => {
  chrome.runtime.openOptionsPage();
});

appFrame.addEventListener("load", () => {
  // If iframe embedding is blocked by server headers, this still "loads" a browser error page.
  // We keep an explicit hint visible for quick diagnosis.
  showError("");
});

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== "sync") return;
  if (changes[SETTINGS_KEY]) {
    mountIframe();
  }
});

mountIframe();
