const SETTINGS_KEY = "videoiq_extension_settings";
const DEFAULT_SETTINGS = {
  backendUrl: "http://127.0.0.1:8000"
};

const form = document.getElementById("settingsForm");
const backendUrlInput = document.getElementById("backendUrl");
const messageEl = document.getElementById("message");
const openPanelBtn = document.getElementById("openPanelBtn");

function normalizeBackendUrl(url) {
  const raw = String(url || "").trim();
  if (!raw) return DEFAULT_SETTINGS.backendUrl;
  return raw.replace(/\/+$/, "");
}

function showMessage(text) {
  messageEl.textContent = text;
  messageEl.classList.remove("hidden");
  setTimeout(() => messageEl.classList.add("hidden"), 2200);
}

async function loadSettings() {
  const data = await chrome.storage.sync.get(SETTINGS_KEY);
  const row = data[SETTINGS_KEY] || DEFAULT_SETTINGS;
  backendUrlInput.value = normalizeBackendUrl(row.backendUrl);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const backendUrl = normalizeBackendUrl(backendUrlInput.value);
  await chrome.storage.sync.set({ [SETTINGS_KEY]: { backendUrl } });
  showMessage("Saved.");
});

openPanelBtn.addEventListener("click", async () => {
  const backendUrl = normalizeBackendUrl(backendUrlInput.value);
  await chrome.storage.sync.set({ [SETTINGS_KEY]: { backendUrl } });
  chrome.tabs.create({ url: `${backendUrl}/?layout=panel` });
});

loadSettings();
