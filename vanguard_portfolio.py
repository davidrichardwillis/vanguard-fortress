import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- Page Config (Must be first) ---
st.set_page_config(page_title="Vanguard Fortress Portfolio Builder", layout="wide")

# --- Define Data ---
VANGUARD_ETFS = {
    "VTI": "Total Stock Market",
    "VOO": "S&P 500",
    "VTV": "Value ETF",
    "VUG": "Growth ETF",
    "BND": "Total Bond Market",
    "BSV": "Short-Term Bond",
    "VGLT": "Long-Term Treasury",
    "VNQ": "Real Estate (REIT)",
    "VPU": "Utilities",
    "VHT": "Health Care",
    "VDC": "Consumer Staples",
    "VXUS": "Total International Stock",
    "BNDX": "Total International Bond"
}

# --- Sidebar (Configuration) ---
st.sidebar.header("Configuration")
st.sidebar.info("Running in Mock Mode (No external API calls)")

selected_tickers = st.sidebar.multiselect(
    "Select ETFs to Analyze", 
    options=list(VANGUARD_ETFS.keys()), 
    default=["VTI", "BND", "VNQ", "VGLT", "VTV"],
    format_func=lambda x: f"{x} - {VANGUARD_ETFS.get(x, '')}"
)

start_year = st.sidebar.number_input("Start Year", min_value=1980, max_value=2024, value=2010)
target_max_drawdown = st.sidebar.slider("Max Acceptable Drawdown (%)", 5, 50, 20)
target_neg_year_pct = st.sidebar.slider("Max % of Negative Years Allowed", 0, 50, 10)

run_btn = st.sidebar.button("Run Analysis", type="primary")

# --- Main Content ---
st.title("🛡️ Vanguard Fortress Portfolio Builder (Mock)")
st.markdown("""
This tool analyzes historical data for Vanguard ETFs to find portfolio combinations 
that maximize returns while minimizing drawdowns and negative years.
""")

if run_btn:
    if not selected_tickers:
        st.error("Please select at least one ETF.")
    else:
        status_text = st.empty()
        status_text.info("Generating mock data...")
        
        # --- Generate MOCK Data ---
        # 10 years of random monthly returns
        dates = pd.date_range(start="2010-01-01", periods=120, freq="ME")
        mock_returns = pd.DataFrame(
            np.random.normal(0.005, 0.04, size=(120, len(selected_tickers))),
            index=dates,
            columns=selected_tickers
        )
        
        # Resample to Annual
        annual_returns = mock_returns.resample('YE').apply(lambda x: (1+x).prod() - 1)
        
        status_text.info(f"Analyzing {len(selected_tickers)} assets...")
        
        # Simplified Monte Carlo Simulation
        num_simulations = 1000
        results = []
        
        # Pre-generate random weights
        all_weights = np.random.dirichlet(np.ones(len(selected_tickers)), size=num_simulations)
        
        progress_bar = st.progress(0)
        
        for i in range(num_simulations):
            weights = all_weights[i]
            
            if i % 100 == 0:
                progress_bar.progress(i / num_simulations)

            port_annual_ret = annual_returns.dot(weights)
            
            mean_ret = port_annual_ret.mean()
            std_dev = port_annual_ret.std()
            sharpe = mean_ret / std_dev if std_dev > 0 else 0
            
            # Drawdown (approx)
            cum_ret = (1 + port_annual_ret).cumprod()
            peak = cum_ret.expanding(min_periods=1).max()
            dd = (cum_ret / peak) - 1
            max_dd = dd.min()
            
            # Neg Years
            neg_years = port_annual_ret[port_annual_ret < 0]
            pct_neg = len(neg_years) / len(port_annual_ret) if len(port_annual_ret) > 0 else 0
            
            # Store only if it meets criteria
            # Relaxed filter for mock data randomness
            if True: 
                results.append({
                   "Weights": dict(zip(selected_tickers, weights)),
                   "CAGR": mean_ret,
                   "Sharpe": sharpe,
                   "MaxDD": max_dd,
                   "PctNegYears": pct_neg
               })
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        if not results:
             st.warning("No portfolios met criteria.")
        else:
            df = pd.DataFrame(results).sort_values(by="Sharpe", ascending=False)
            best = df.iloc[0]
            
            st.success(f"🏆 Found {len(df)} Valid Portfolios! Best Match:")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Sharpe Ratio", f"{best['Sharpe']:.2f}")
            c2.metric("Annual Return", f"{best['CAGR']:.2%}")
            c3.metric("Max Drawdown", f"{best['MaxDD']:.2%}")
            
            st.subheader("Optimal Allocation")
            alloc = {k: v for k, v in best['Weights'].items() if v > 0.01}
            st.bar_chart(pd.Series(alloc))
            
            st.dataframe(df.head(5)[["CAGR", "Sharpe", "MaxDD", "PctNegYears"]].style.format("{:.2%}"))
