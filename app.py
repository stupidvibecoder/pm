import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import date, timedelta

st.set_page_config(page_title="Manual Portfolio Tracker", layout="wide")
st.title("Manual Portfolio Tracker (No Broker API)")

# -------------------------- Helpers --------------------------
@st.cache_data(ttl=300)
def fetch_prices(tickers, start, end):
    """Download adjusted close prices for a list of tickers, daily frequency."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end + timedelta(days=1),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        closes = data["Close"].copy()
    else:
        closes = data.copy()
        if isinstance(tickers, list) and tickers:
            closes.name = tickers[0]
        closes = pd.DataFrame(closes)
    closes.columns = [str(c) for c in closes.columns]
    if isinstance(closes.index, pd.DatetimeIndex) and closes.index.tz is not None:
        closes.index = closes.index.tz_convert(None)
    return closes.sort_index().dropna(how="all")

def validate_lots(df):
    """Basic validation + coercion of manually entered/CSV lots."""
    df = df.copy()
    rename = {
        "ticker": "ticker",
        "symbol": "ticker",
        "buy_date": "buy_date",
        "date": "buy_date",
        "shares": "shares",
        "qty": "shares",
        "cost_basis_per_share": "cost_basis_per_share",
        "price": "cost_basis_per_share",
        "cost_basis": "cost_basis_per_share",
    }
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})

    needed = ["ticker", "buy_date", "shares", "cost_basis_per_share"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce").dt.date
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["cost_basis_per_share"] = pd.to_numeric(df["cost_basis_per_share"], errors="coerce")
    df = df.dropna(subset=["ticker", "buy_date", "shares", "cost_basis_per_share"])
    df = df[df["shares"] > 0]
    return df

def build_share_matrix(lots_df, prices):
    """Time-by-ticker matrix of shares held each day (step function from buy_date)."""
    if prices.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    dates = prices.index
    tickers = prices.columns.tolist()
    shares_mat = pd.DataFrame(0.0, index=dates, columns=tickers)
    for tkr in tickers:
        impulses = pd.Series(0.0, index=dates)
        lots_t = lots_df[lots_df["ticker"] == tkr]
        for _, r in lots_t.iterrows():
            d = pd.Timestamp(r["buy_date"])
            loc = dates.searchsorted(d)
            if loc < len(dates):
                impulses.iloc[loc] += float(r["shares"])
        shares_mat[tkr] = impulses.cumsum()
    return shares_mat

def cumulative_invested(lots_df, date_index):
    """Cumulative cash invested over time from the lots."""
    if date_index.empty or lots_df.empty:
        return pd.Series(dtype=float)
    inv = pd.Series(0.0, index=date_index)
    for _, r in lots_df.iterrows():
        d = pd.Timestamp(r["buy_date"])
        amount = float(r["shares"] * r["cost_basis_per_share"])
        loc = date_index.searchsorted(d)
        if loc < len(inv):
            inv.iloc[loc] += amount
    return inv.cumsum()

@st.cache_data(ttl=3600)
def get_logo_url(ticker: str) -> str:
    """Best-effort company logo via yfinance (gracefully falls back to '')."""
    try:
        info = yf.Ticker(ticker).info or {}
        url = info.get("logo_url") or ""
        return url if isinstance(url, str) else ""
    except Exception:
        return ""

# -------------------------- Inputs --------------------------
st.sidebar.header("Holdings input")

tab_manual, tab_csv = st.sidebar.tabs(["Type lots", "Upload CSV"])

with tab_manual:
    st.sidebar.caption("Enter one row per lot (per buy date).")
    default_rows = pd.DataFrame(
        [
            {
                "ticker": "CPRT",
                "buy_date": date.today() - timedelta(days=400),
                "shares": 100,
                "cost_basis_per_share": 40.00,
            },
        ]
    )
    lots_editor = st.sidebar.data_editor(
        default_rows,
        key="lots_editor",
        num_rows="dynamic",
        use_container_width=True,
    )
    lots_source = "manual"

with tab_csv:
    up = st.sidebar.file_uploader("Upload lots CSV", type=["csv"])
    if up is not None:
        lots_source = "csv"

if lots_source == "csv" and up is not None:
    try:
        lots_df = pd.read_csv(up)
        lots_df = validate_lots(lots_df)
    except Exception as e:
        st.error(f"CSV error: {e}")
        st.stop()
else:
    try:
        lots_df = validate_lots(pd.DataFrame(lots_editor))
    except Exception as e:
        st.error(f"Input error: {e}")
        st.stop()

if lots_df.empty:
    st.info("Add at least one lot to proceed.")
    st.stop()

# Date range to chart
min_buy = min(lots_df["buy_date"])
start_default = max(min_buy - timedelta(days=5), date(1990, 1, 1))
end_default = date.today()

colA, colB = st.columns(2)
with colA:
    start_date = st.date_input("Start date", value=start_default, min_value=date(1990, 1, 1), max_value=end_default)
with colB:
    end_date = st.date_input("End date", value=end_default, min_value=start_date, max_value=end_default)

# Optional benchmark
bm = st.selectbox("Optional benchmark (scaled to portfolio start value)", ["None", "SPY", "^GSPC", "QQQ"], index=0)

# -------------------------- Load prices --------------------------
tickers = sorted(lots_df["ticker"].unique().tolist())
with st.spinner(f"Downloading daily prices for {', '.join(tickers)}…"):
    prices = fetch_prices(tickers, start_date, end_date)
if prices.empty:
    st.error("No price data available for the selected range.")
    st.stop()

# Shares matrix & portfolio value
shares_mat = build_share_matrix(lots_df, prices)
port_val = (shares_mat * prices).sum(axis=1)
invested = cumulative_invested(lots_df, prices.index)

# -------------------------- Charts --------------------------
st.subheader("Portfolio value over time")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=port_val.index,
        y=port_val.values,
        mode="lines",
        name="Portfolio value",
        hovertemplate="Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
    )
)
if not invested.empty:
    fig.add_trace(
        go.Scatter(
            x=invested.index,
            y=invested.values,
            mode="lines",
            name="Cumulative invested",
            hovertemplate="Date: %{x}<br>Invested: $%{y:,.0f}<extra></extra>",
        )
    )

# Benchmark scaled to portfolio's start value (not index=100)
if bm != "None":
    try:
        bm_px = fetch_prices([bm], start_date, end_date).iloc[:, 0].dropna()
        if not bm_px.empty and not port_val.empty:
            scale_to = float(port_val.iloc[0])
            bm_scaled = (bm_px / bm_px.iloc[0]) * scale_to
            fig.add_trace(
                go.Scatter(
                    x=bm_scaled.index,
                    y=bm_scaled.values,
                    mode="lines",
                    name=f"{bm} (scaled to portfolio start)",
                    hovertemplate="Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
                    line=dict(dash="dash"),
                )
            )
    except Exception:
        pass

fig.update_layout(
    template="plotly_white",
    height=520,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Dollars ($)",
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------- Positions & P&L --------------------------
st.subheader("Positions & P&L (current)")
cur_shares = shares_mat.iloc[-1].rename("shares").to_frame()
cur_shares = cur_shares[cur_shares["shares"] > 0]
if cur_shares.empty:
    st.info("No open positions at the end date.")
else:
    latest_px = prices.ffill().iloc[-1].rename("last_price")
    lot_costs = (
        lots_df.groupby("ticker")
        .apply(lambda g: pd.Series({
            "total_shares": g["shares"].sum(),
            "total_cost": (g["shares"] * g["cost_basis_per_share"]).sum(),
        }))
    )
    lot_costs["avg_cost_ps"] = lot_costs["total_cost"] / lot_costs["total_shares"]

    pos = cur_shares.join(latest_px, how="left")
    pos = pos.join(lot_costs[["avg_cost_ps"]], how="left")
    pos["market_value"] = pos["shares"] * pos["last_price"]
    pos["cost_basis_total"] = pos["shares"] * pos["avg_cost_ps"]
    pos["unrealized_pnl"] = pos["market_value"] - pos["cost_basis_total"]
    with np.errstate(divide="ignore", invalid="ignore"):
        pos["unrealized_pnl_pct"] = np.where(
            pos["cost_basis_total"] > 0,
            pos["unrealized_pnl"] / pos["cost_basis_total"] * 100.0,
            np.nan,
        )
    pos = pos.reset_index().rename(columns={"index": "ticker"})

    # Show the table (kept for reference)
    st.dataframe(pos.sort_values("market_value", ascending=False), use_container_width=True, hide_index=True)

    # -------------------- Per-ticker cards (not a table) --------------------
    st.subheader("Per-ticker summary")
    st.caption("Factor/individual return cards — logo, ticker, MV, shares, price, cost basis, cost/share, total return.")
    # card grid
    cards_per_row = 3
    sorted_pos = pos.sort_values("market_value", ascending=False).reset_index(drop=True)
    rows = [sorted_pos.iloc[i:i+cards_per_row] for i in range(0, len(sorted_pos), cards_per_row)]

    for chunk in rows:
        cols = st.columns(len(chunk))
        for col, (_, r) in zip(cols, chunk.iterrows()):
            tkr = str(r["ticker"])
            mv  = float(r["market_value"])
            sh  = float(r["shares"])
            px  = float(r["last_price"])
            cbT = float(r["cost_basis_total"]) if not pd.isna(r["cost_basis_total"]) else np.nan
            cbP = float(r["avg_cost_ps"]) if not pd.isna(r["avg_cost_ps"]) else np.nan
            ret = float(r["unrealized_pnl_pct"]) if not pd.isna(r["unrealized_pnl_pct"]) else np.nan

            logo = get_logo_url(tkr)
            # simple card using HTML; safe minimal styling
            html = f"""
<div style="border:1px solid #e6e6e6;border-radius:12px;padding:14px;margin-bottom:10px">
  <div style="display:flex;align-items:center;gap:10px;">
    {'<img src="'+logo+'" style="width:28px;height:28px;border-radius:6px;">' if logo else ''}
    <div style="font-weight:700;font-size:18px;">{tkr}</div>
  </div>
  <div style="margin-top:8px;line-height:1.6;">
    <div><b>Market Value:</b> ${mv:,.2f}</div>
    <div><b>Shares:</b> {sh:,.4f}</div>
    <div><b>Last Price:</b> ${px:,.2f}</div>
    <div><b>Cost Basis (total):</b> {'$'+format(cbT,',.2f') if not np.isnan(cbT) else '—'}</div>
    <div><b>Cost/Share:</b> {'$'+format(cbP,',.4f') if not np.isnan(cbP) else '—'}</div>
    <div><b>Total Return:</b> {('+' if ret>=0 else '')+format(ret,'.2f')}%</div>
  </div>
</div>
"""
            with col:
                st.markdown(html, unsafe_allow_html=True)

# -------------------------- Lots ledger --------------------------
st.subheader("Lots ledger (input)")
st.caption("Exact data used to compute shares and invested cash.")
st.dataframe(lots_df.sort_values(["ticker", "buy_date"]).reset_index(drop=True),
             use_container_width=True, hide_index=True)

# -------------------------- Downloads --------------------------
st.subheader("Download")
colx, coly = st.columns(2)
with colx:
    out_ledger = lots_df.sort_values(["ticker","buy_date"]).reset_index(drop=True)
    st.download_button(
        label="Download lots CSV",
        data=out_ledger.to_csv(index=False).encode("utf-8"),
        file_name="lots.csv",
        mime="text/csv",
    )
with coly:
    pv = pd.DataFrame({"date": port_val.index.strftime("%Y-%m-%d"), "portfolio_value": port_val.values})
    st.download_button(
        label="Download portfolio value CSV",
        data=pv.to_csv(index=False).encode("utf-8"),
        file_name="portfolio_value.csv",
        mime="text/csv",
    )

# -------------------------- CSV format hint --------------------------
with st.expander("CSV format expected"):
    st.markdown("Columns (case-insensitive): ticker, buy_date (YYYY-MM-DD), shares, cost_basis_per_share")
    st.code(
        "ticker,buy_date,shares,cost_basis_per_share\n"
        "CPRT,2023-03-01,100,40.00\n"
        "AAPL,2023-08-15,10,180.50\n",
        language="csv",
    )
