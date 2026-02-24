import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
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

selected_tickers = st.sidebar.multiselect(
    "Select Universe to Test", 
    options=list(VANGUARD_ETFS.keys()), 
    default=["VTI", "BND", "VNQ", "VGLT", "VTV"],
    format_func=lambda x: f"{x} - {VANGUARD_ETFS.get(x, '')}"
)

start_year = st.sidebar.number_input("Start Year", min_value=1980, max_value=2024, value=2010)
end_year = datetime.now().year

target_max_drawdown = st.sidebar.slider("Max Acceptable Drawdown (%)", 5, 50, 20)
target_neg_year_pct = st.sidebar.slider("Max % of Negative Years Allowed", 0, 50, 10)

allow_sparse = st.sidebar.checkbox("Try Concentrated Subsets?", value=True, help="If checked, the algorithm will test portfolios that drop some ETFs entirely to find a better 2-5 asset mix.")

run_btn = st.sidebar.button("Run Analysis", type="primary")

# --- Main Content ---
st.title("🛡️ Vanguard Fortress Portfolio Builder")
st.markdown("""
This tool analyzes historical data for Vanguard ETFs to find portfolio combinations 
that maximize returns while minimizing drawdowns and negative years.
""")

if run_btn:
    if not selected_tickers:
        st.error("Please select at least one ETF.")
    else:
        status_text = st.empty()
        status_text.info("Fetching market data...")
        
        try:
            # Fetch Data
            df = yf.download(selected_tickers, start=f"{start_year}-01-01", end=f"{end_year}-12-31", progress=False)
            
            # Extract Adjusted Close
            if 'Adj Close' in df.columns:
                data = df['Adj Close']
            elif 'Close' in df.columns:
                data = df['Close']
            else:
                data = df
            
            if data.empty:
                st.error("No data found.")
            else:
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=selected_tickers[0])
                
                # Resample to Annual
                annual_data = data.resample('YE').last().ffill()
                annual_returns = annual_data.pct_change().dropna()
                
                status_text.info(f"Analyzing {len(selected_tickers)} assets...")
                
                # Simulation Parameters
                num_simulations = 5000 if allow_sparse else 2000
                results = []
                
                progress_bar = st.progress(0)
                
                # Pre-generate weights
                if allow_sparse:
                    # Sparse Logic: Generate weights where some are explicitly zero
                    # We do this by generating Dirichlet weights on a random SUBSET of indices
                    all_weights = np.zeros((num_simulations, len(selected_tickers)))
                    
                    for i in range(num_simulations):
                        # Pick a random number of assets to include (2 to N)
                        n_assets = len(selected_tickers)
                        k = np.random.randint(2, n_assets + 1)
                        
                        # Pick k random indices
                        indices = np.random.choice(n_assets, k, replace=False)
                        
                        # Generate weights for these k assets
                        sub_weights = np.random.dirichlet(np.ones(k))
                        
                        # Assign to the full weight array
                        all_weights[i, indices] = sub_weights
                else:
                    # Standard Dense Logic (All assets included)
                    all_weights = np.random.dirichlet(np.ones(len(selected_tickers)), size=num_simulations)

                # Simulation Loop
                for i in range(num_simulations):
                    weights = all_weights[i]
                    
                    if i % 200 == 0:
                        progress_bar.progress(i / num_simulations)

                    # Calculate Portfolio Annual Returns
                    port_annual_ret = annual_returns.dot(weights)
                    
                    mean_ret = port_annual_ret.mean()
                    std_dev = port_annual_ret.std()
                    sharpe = mean_ret / std_dev if std_dev > 0 else 0
                    
                    cum_ret = (1 + port_annual_ret).cumprod()
                    peak = cum_ret.expanding(min_periods=1).max()
                    dd = (cum_ret / peak) - 1
                    max_dd = dd.min()
                    
                    neg_years = port_annual_ret[port_annual_ret < 0]
                    pct_neg = len(neg_years) / len(port_annual_ret) if len(port_annual_ret) > 0 else 0
                    
                    if (abs(max_dd) * 100 <= target_max_drawdown) and (pct_neg * 100 <= target_neg_year_pct):
                        results.append({
                           "Weights": dict(zip(selected_tickers, weights)),
                           "CAGR": mean_ret,
                           "Sharpe": sharpe,
                           "MaxDD": max_dd,
                           "PctNegYears": pct_neg,
                           "RawWeights": weights
                       })
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                if not results:
                     st.warning("No portfolios met criteria.")
                else:
                    df = pd.DataFrame(results).sort_values(by="Sharpe", ascending=False)
                    best = df.iloc[0]
                    
                    st.success(f"🏆 Best Found Portfolio (Sharpe: {best['Sharpe']:.2f})")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Annual Return", f"{best['CAGR']:.2%}")
                    c2.metric("Max Drawdown", f"{best['MaxDD']:.2%}")
                    c3.metric("Negative Years %", f"{best['PctNegYears']:.1%}")
                    
                    st.subheader("Optimal Allocation")
                    # Filter out tiny weights (< 1%)
                    alloc = {k: v for k, v in best['Weights'].items() if v > 0.01}
                    
                    col_chart, col_list = st.columns([2, 1])
                    with col_chart:
                        st.bar_chart(pd.Series(alloc))
                    with col_list:
                        st.markdown("### Assets Included:")
                        for ticker, weight in alloc.items():
                            st.markdown(f"**{ticker}:** {weight:.1%}")
                    
                    st.subheader("📅 Annual Performance History")
                    best_weights = best['RawWeights']
                    best_annual_series = annual_returns.dot(best_weights)
                    
                    annual_df = best_annual_series.to_frame(name="Return")
                    annual_df.index = annual_df.index.year.astype(str)
                    annual_df = annual_df.sort_index(ascending=False)
                    
                    def color_negative_red(val):
                        color = '#ffcccb' if val < 0 else '#90ee90'
                        return f'background-color: {color}; color: black'

                    st.dataframe(annual_df.style.format("{:.2%}").applymap(color_negative_red), use_container_width=True)

        except Exception as e:
            st.error(f"Analysis Error: {e}")
