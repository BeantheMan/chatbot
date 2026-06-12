import streamlit as st
import anthropic
import pandas as pd
import importlib
import sys

from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from collections import Counter

# Force reload of usda_fob_parser on every Streamlit execution.
# Without this, Streamlit keeps the module cached in sys.modules and does not
# reload changes even if the file was updated on disk
# (for example, after a git pull).
if "usda_fob_parser" in sys.modules:
    importlib.reload(sys.modules["usda_fob_parser"])
if "usda_extractor" in sys.modules:
    importlib.reload(sys.modules["usda_extractor"])
from usda_extractor import get_usda_data


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
# claude-sonnet-4-6             claude-haiku-4-5-20251001
MODEL            = "claude-haiku-4-5-20251001"


# ─────────────────────────────────────────────
# LOAD REPORT (cached 1 hour)
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_report():
    df, context = get_usda_data()#get_fob_dataframe()
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
            pacific_now = datetime.now(ZoneInfo("America/Los_Angeles"))
            today: str = pacific_now.strftime("%m/%d/%Y")
            yesterday: str = (pacific_now - timedelta(days=1)).strftime("%m/%d/%Y")

            st.download_button(
                label="📥 Download Report CSV",
                data=context,
                file_name=f"usda_data_{today.replace("/", "_")}.csv",
                mime="text/csv"
            )

            st.success(f"✅ {len(df):,} entries")
            st.write(f"📅 Today: **{today}**")
            st.write(f"📦 Commodities: **{df['commodity'].nunique()}**")
            st.write(f"🗺️ Areas: **{df['origin'].nunique()}**")

            st.divider()
            st.subheader("Entries")
            st.write(f"• {today}: {(df["report_end_date"] == today).sum()}")
            st.write(f"• {yesterday}: {(df["report_end_date"] == yesterday).sum()}")

            st.divider()
            st.subheader("Categories")
            for cat, cnt in df.groupby("category")["commodity"].count().items():
                st.write(f"• {cat}: {cnt}")

            st.divider()
            st.subheader("Origins")
            
            origins = Counter()
            for orig, cnt in (
                df.groupby("origin")["commodity"].count()
                .sort_values(ascending=False).items()
            ):
                substrs = [
                    "IMPERIAL VALLEY CALIFORNIA AND CENTRAL AND WESTERN ARIZONA", 
                    "Mexico crossing", 
                    "California", 
                    "Washington",
                    "Texas",
                    "North Carolina"
                ]
                orig_str = str(orig).split(',', 1)[0]
                if len(orig_str) > 1 and orig_str[1].isupper():
                    orig_str = orig_str[:1] + orig_str[1:].lower()
                matched = next((sub for sub in substrs if sub.casefold() in orig_str.casefold()), None)
                match matched:
                    case "Mexico crossing":
                        origins["Mexico"] += cnt
                    case "IMPERIAL VALLEY CALIFORNIA AND CENTRAL AND WESTERN ARIZONA":
                        origins["Arizona-California"] += cnt
                    case _:
                        if matched:
                            origins[matched] += cnt
                        else:
                            origins[orig_str] += cnt
                
            for place, count in sorted(origins.items()):
                st.write(f"• {place}: {count}")
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
    system_prompt = f"""
You are a produce market analyst specializing in USDA FOB shipping point prices.

Use only the CSV REPORT DATA below to answer questions.

Rules:
- Be as accurate as possible.
- If an item or commodity is not in the data, concisely say so.
- Format prices as: $X.XX-$Y.YY per package.
- Always include report_end_date, grade and quality  when discussing prices.
- For price comparisons, reference relevant fields when available:
  variety, properties, package, item_size, category, organic,
  grade, quality, origin, market_location_name, report_end_date.
- If a prompt asks for a precision check, recheck the accuracy of your last response

CSV REPORT DATA:
{context if context else "No report data available. Please update the report."}
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
