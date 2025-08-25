# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

from brokers.schwab_client import SchwabClient, LoadMode

st.set_page_config(page_title="Schwab Portfolio Dashboard", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("Schwab Portfolio")
load_mode = st.sidebar.radio(
    "Data source",
    [LoadMode.CSV.value, LoadMode.API.value],
    index=0,
    help="Start with CSV uploads. You can switch to the API later."
)

# ---------------- Data loader ----------------
client = SchwabClient(load_mode=load_mode)

if client.mode_is_csv:
    st.sidebar.subheader("Upload CSVs")
    pos_file = st.sidebar.file_uploader("Positions CSV", type=["csv"])
    txn_file = st.sidebar.file_uploader("Transactions CSV", type=["csv"])
    acct_file = st.sidebar.file_uploader("Accounts/Balances CSV (optional)", type=["csv"])
    if st.sidebar.button("Load CSVs"):
        client.load_from_csv_files(pos_file, txn_file, acct_file)
else:
    st.sidebar.subheader("Schwab API (optional)")
    st.sidebar.info("Set API secrets in your deploy environment to enable. This demo uses CSV by default.")
    if st.sidebar.button("Connect (if configured)"):
        client.load_from_api()

# ---------------- Fetch portfolio data ----------------
positions, transactions, accounts = client.get_positions(), client.get_transactions(), client.get_accounts()

# guard rails
if positions is None and transactions is None and accounts is None:
    st.info("Upload your CSVs (or connect API) to get started.")
    st.stop()

# ---------------- Overview ----------------
st.title("Portfolio Overview")

col1, col2, col3 = st.columns(3)
total_mv = float(positions["market_value"].sum()) if positions is not None and "market_value" in positions else 0.0
cash = float(accounts["cash"].sum()) if accounts is not None and "cash" in accounts else 0.0
today_val = total_mv + cash
with col1:
    st.metric("Total Market Value", f"${total_mv:,.2f}")
with col2:
    st.metric("Cash", f"${cash:,.2f}")
with col3:
    st.metric("Total (MV + Cash)", f"${today_val:,.2f}")

# ---------------- Positions table ----------------
st.header("Positions")
if positions is not None and not positions.empty:
    # fetch live/last prices with yfinance
    tickers = positions["symbol"].dropna().unique().tolist()
    if tickers:
        try:
            data = yf.download(tickers=tickers, period="5d", interval="1d", auto_adjust=True, progress=False, threads=True)
            # handle single vs multi
            if isinstance(data, pd.DataFrame) and "Close" in data.columns:
                last_prices = data["Close"].ffill().iloc[-1]
                def last_px(sym):
                    try:
                        return float(last_prices[sym])
                    except Exception:
                        # single-ticker shape fallback
                        return float(data["Close"].ffill().iloc[-1])
                positions["last_price_yf"] = positions["symbol"].apply(last_px)
            else:
                positions["last_price_yf"] = np.nan
        except Exception:
            positions["last_price_yf"] = np.nan

    # compute P&L vs cost basis if available
    if "cost_basis_total" in positions.columns and "quantity" in positions.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            positions["pnl_unrealized"] = positions["market_value"] - positions["cost_basis_total"]
            positions["pnl_unrealized_pct"] = np.where(
                positions["cost_basis_total"] != 0,
                positions["pnl_unrealized"] / positions["cost_basis_total"] * 100.0,
                np.nan
            )

    st.dataframe(
        positions.sort_values("market_value", ascending=False)
                 .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No positions found (yet).")

# ---------------- Performance (basic) ----------------
st.header("Performance (Price-only, non-IRR)")
st.caption("Quick price-based performance view using Yahoo Finance (does not include dividends unless adjusted, nor cashflows).")

# Choose a subset tickers to plot
if positions is not None and not positions.empty:
    sample = st.multiselect(
        "Select tickers to chart",
        options=positions["symbol"].dropna().unique().tolist(),
        default=positions["symbol"].dropna().unique().tolist()[:6]
    )

    rng = st.selectbox("Range", ["1M","3M","6M","1Y","YTD","Max"], index=2)
    end = datetime.today().date()
    if   rng=="1M": start = end - timedelta(days=30)
    elif rng=="3M": start = end - timedelta(days=90)
    elif rng=="6M": start = end - timedelta(days=180)
    elif rng=="1Y": start = end - timedelta(days=365)
    elif rng=="YTD": start = datetime(end.year,1,1).date()
    else: start = end - timedelta(days=3650)

    if sample:
        hist = yf.download(sample, start=start, end=end + timedelta(days=1), auto_adjust=True, progress=False, threads=True)
        fig = go.Figure()
        if "Close" in hist.columns:
            closes = hist["Close"]
        else:
            # single ticker shape
            closes = hist

        # normalize to 100
        if isinstance(closes, pd.DataFrame):
            for sym in closes.columns:
                s = closes[sym].dropna()
                base = s.iloc[0]
                idx = s / base * 100.0
                fig.add_trace(go.Scatter(x=idx.index, y=idx.values, mode="lines", name=sym))
        else:
            s = closes.dropna()
            base = s.iloc[0]
            idx = s / base * 100.0
            fig.add_trace(go.Scatter(x=idx.index, y=idx.values, mode="lines", name=sample[0]))

        fig.update_layout(
            title="Indexed to 100 (price-only)",
            xaxis_title="Date",
            yaxis_title="Index (100 = start)",
            template="plotly_white",
            height=420,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Add positions to view charts.")

# ---------------- Transactions (optional) ----------------
st.header("Transactions")
if transactions is not None and not transactions.empty:
    st.dataframe(
        transactions.sort_values("date", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True
    )
else:
    st.info("No transactions loaded.")
