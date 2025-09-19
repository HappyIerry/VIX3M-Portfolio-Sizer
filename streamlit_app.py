import streamlit as st
import yfinance as yf
import pandas as pd

# ----------------------------
# Functions
# ----------------------------
def fetch_vix3m_latest() -> float:
    hist = yf.Ticker("^VIX3M").history(period="5d", interval="1d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError("No recent data returned for ^VIX3M.")
    return float(hist["Close"].iloc[-1])

def fetch_vix3m_ref_one_year() -> float:
    ref_dt = (pd.Timestamp.today() - pd.Timedelta(days=365)).normalize()
    start = (ref_dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (ref_dt + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    df = yf.download("^VIX3M", start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("Failed to fetch historical VIX3M.")
    # nearest prior value
    dates = df.index[df.index <= ref_dt]
    if dates.empty:
        return float(df["Close"].iloc[0])
    return float(df.loc[dates[-1], "Close"])

def compute_multiplier(vix3m_now: float, vix3m_ref: float, min_mult: float = 0.1) -> float:
    if vix3m_now <= 0 or vix3m_ref <= 0:
        raise ValueError("VIX3M values must be positive.")
    return max(vix3m_ref / vix3m_now, min_mult)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="VIX3M Portfolio Sizer", layout="centered")
st.title("📊 VIX3M Portfolio Sizer Dashboard")

st.markdown("""
This dashboard fetches the **latest VIX3M**, computes a **1-year-ago reference**, and calculates
a **portfolio sizing multiplier** (`ref / now`). You can also override the reference value or date.
""")

# User inputs
override_ref = st.number_input("Override numeric reference (optional)", min_value=0.0, value=0.0, step=0.1)
override_date = st.date_input("Override reference date (optional)", value=None)

# Fetch data
try:
    v_now = fetch_vix3m_latest()
except Exception as e:
    st.error(f"Failed to fetch latest VIX3M: {e}")
    st.stop()

# Determine reference
if override_ref and override_ref > 0:
    v_ref = override_ref
    ref_source = "User-supplied"
else:
    try:
        if override_date:
            ref_dt = pd.Timestamp(override_date)
            start = (ref_dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            end = (ref_dt + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            df = yf.download("^VIX3M", start=start, end=end, progress=False, auto_adjust=False)
            if df.empty:
                st.warning(f"No historical data for {ref_dt.date()}, using 1-year-ago default.")
                v_ref = fetch_vix3m_ref_one_year()
            else:
                dates = df.index[df.index <= ref_dt]
                if len(dates) == 0:
                    v_ref = float(df["Close"].iloc[0])
                else:
                    v_ref = float(df.loc[dates[-1], "Close"])
            ref_source = f"Historical ({ref_dt.date()})"
        else:
            v_ref = fetch_vix3m_ref_one_year()
            ref_source = "1-year-ago default"
    except Exception as e:
        st.error(f"Failed to fetch reference VIX3M: {e}")
        st.stop()

# Compute multiplier
mult = compute_multiplier(v_now, v_ref)

# Display results
st.subheader("Multiplier Result")
st.metric(label="Latest VIX3M", value=f"{v_now:.2f}")
st.metric(label=f"Reference VIX3M ({ref_source})", value=f"{v_ref:.2f}")
st.metric(label="Portfolio Multiplier", value=f"{mult:.4f}")

st.markdown("---")
st.markdown("💡 **Note:** Multiplier = `ref / latest`. Apply to scale your portfolio exposure accordingly.")

