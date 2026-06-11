import streamlit as st
import anthropic
import pandas as pd
import importlib
import sys

# Force reload of usda_fob_parser on every Streamlit execution.
# Without this, Streamlit keeps the module cached in sys.modules and does not
# reload changes even if the file was updated on disk
# (for example, after a git pull).
if "usda_fob_parser" in sys.modules:
    importlib.reload(sys.modules["usda_fob_parser"])

from usda_fob_parser import get_fob_dataframe


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="USDA FOB Market Analyst",
    page_icon="🥦",
    layout="wide",
)

ANTHROPIC_CLIENT = anthropic.Anthropic(
    api_key=st.secrets["ANTHROPIC_API_KEY"]
)
MODEL            = "claude-opus-4-20250514"


# ─────────────────────────────────────────────
# LOAD REPORT (cached 1 hour)
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_report():
    df, context = get_fob_dataframe()
    return df, context


# ─────────────────────────────────────────────
# SIDEBAR — REPORT STATUS PANEL
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Report Status")

    if st.button("🔄 Update report"):
        st.cache_data.clear()
        st.rerun()

    try:
        with st.spinner("📥 Downloading..."):
            df, context = load_report()

        # Auto-clear if cached result is empty
        if df.empty:
            st.cache_data.clear()
            df, context = load_report()

        if not df.empty:
            st.success(f"✅ {len(df):,} entries")
            st.write(f"📅 Date: **{df['date'].iloc[0].date()}**")
            st.write(f"📦 Commodities: **{df['commodity'].nunique()}**")
            st.write(f"🗺️ Areas: **{df['region'].nunique()}**")

            st.divider()
            st.subheader("Categories")
            for cat, cnt in df.groupby("category")["commodity"].count().items():
                st.write(f"• {cat}: {cnt}")

            st.divider()
            st.subheader("Origins")
            for orig, cnt in (
                df.groupby("origin")["commodity"].count()
                .sort_values(ascending=False).items()
            ):
                st.write(f"• {orig}: {cnt}")
        else:
            st.error("⚠️ No data found — check connection or PDF format")
            context = ""

    except Exception as e:
        st.error(f"Error: {e}")
        df      = pd.DataFrame()
        context = ""


# ─────────────────────────────────────────────
# MAIN AREA — CHAT WITH THE AGENT
# ─────────────────────────────────────────────

st.title("🥦 USDA FOB Market Analyst")
st.caption("Ask questions about today's FOB shipping point prices.")

# Conversation history in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about prices, commodities, origins…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build system prompt with the report data
    system_prompt = f"""You are a produce market analyst specializing in USDA FOB shipping point prices.
You have access to today's USDA FOB report data. Use this data to answer questions accurately.

REPORT DATA:
{context if context else "No report data available — please update the report."}

Guidelines:
- Be concise and specific with price data.
- When comparing prices, reference the package type and origin.
- If asked about a commodity not in the data, say so clearly.
- Format prices as $X.XX-$Y.YY per package.
- Always mention the report date when discussing prices.
"""

    # Build message history for the API
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = ANTHROPIC_CLIENT.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system_prompt,
                messages=api_messages,
            )
            answer = response.content[0].text

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
