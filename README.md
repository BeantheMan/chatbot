# 🥦 USDA FOB Market Analyst

A Streamlit app that lets you chat with an AI analyst about daily USDA FOB shipping-point prices. It downloads the latest USDA FOB report, parses it into structured data, and answers questions via Anthropic Claude.

## Running locally for development

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your API key

Create `.streamlit/secrets.toml` (it is git-ignored) and add your Anthropic API key:

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-..."
```

### 3. Start the dev server

**Windows:**

```bash
scripts\run.bat
```

**macOS / Linux:**

```bash
./scripts/run.sh
```

Or run Streamlit directly:

```bash
streamlit run streamlit_app.py
```

The app will open at http://localhost:8501.

### Dev settings

The `.streamlit/config.toml` file enables:

- `runOnSave = true` — the app reloads automatically when you edit code.
- `fastReruns = true` — faster rerun during dev.
- `gatherUsageStats = false` — disables Streamlit usage telemetry.
