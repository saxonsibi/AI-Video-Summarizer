# VideoIQ Side Panel Extension (No Build)

This is a lightweight MV3 wrapper that opens your existing frontend in Chrome Side Panel:

- Loads: `http://127.0.0.1:8000/?layout=panel`
- Uses your current frontend components/layout (panel mode)
- No React/Vite build needed for extension

## Files

- `manifest.json`
- `background.js`
- `sidepanel.html`, `sidepanel.css`, `sidepanel.js`
- `options.html`, `options.css`, `options.js`

## Load in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select this folder: `extension/`

## Backend requirement

Run Django:

```powershell
cd backend
python manage.py runserver
```

The extension opens your frontend in iframe mode.  
In Django settings this is controlled by:

- `ALLOW_EXTENSION_IFRAME=True` (enabled by default in DEBUG)

## Settings

Open extension options page and set backend URL if needed:

- default: `http://127.0.0.1:8000`

