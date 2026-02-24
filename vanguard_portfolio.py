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
    "Select ETFs to Analyze", 
    options=list(VANGUARD_ETFS.keys()), 
    default=["VTI", "BND", "VNQ", "VGLT", "VTV"],
    format_func=lambda x: f"{x} - {VANGUARD_ETFS.get(x, '')}"
)

start_year = st.sidebar.number_input("Start Year", min_value=1980, max_value=2024, value=2010)
end_year = datetime.now().year

target_max_drawdown = st.sidebar.slider("Max Acceptable Drawdown (%)", 5, 50, 20)
target_neg_year_pct = st.sidebar.slider("Max % of Negative Years Allowed", 0, 50, 10)

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
            # auto_adjust=True makes it return just Close, Open, etc. adjusted.
            # We just want the 'Close' price which is adjusted by default in new yfinance
            df = yf.download(selected_tickers, start=f"{start_year}-01-01", end=f"{end_year}-12-31", progress=False)
            
            # Extract Adjusted Close
            # Handle different yfinance return structures
            if 'Adj Close' in df.columns:
                data = df['Adj Close']
            elif 'Close' in df.columns:
                data = df['Close']
            else:
                # If single ticker and flat, it might just be the dataframe itself
                data = df
            
            # Check if empty
            if data.empty:
                st.error("No data found for the selected tickers/dates.")
            else:
                # Ensure it's a DataFrame (if single ticker Series)
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=selected_tickers[0])
                
                # Resample to Annual
                # Using 'YE' for Year End (pandas 2.2+ compliant)
                # Ensure we handle potential NA from the resampling
                annual_data = data.resample('YE').last().ffill()
                
                # Calculate Returns
                annual_returns = annual_data.pct_change().dropna()
                
                status_text.info(f"Analyzing {len(selected_tickers)} assets...")
                
                # Simplified Monte Carlo Simulation
                num_simulations = 2000
                results = []
                
                # Pre-generate random weights (Dirichlet distribution for simplex sum=1)
                all_weights = np.random.dirichlet(np.ones(len(selected_tickers)), size=num_simulations)
                
                progress_bar = st.progress(0)
                
                for i in range(num_simulations):
                    weights = all_weights[i]
                    
                    if i % 200 == 0:
                        progress_bar.progress(i / num_simulations)

                    # Calculate Portfolio Annual Returns
                    # annual_returns is (Years x Assets), weights is (Assets,)
                    # result is (Years,)
                    port_annual_ret = annual_returns.dot(weights)
                    
                    mean_ret = port_annual_ret.mean()
                    std_dev = port_annual_ret.std()
                    sharpe = mean_ret / std_dev if std_dev > 0 else 0
                    
                    # Drawdown Calculation (Approximate using Annual Returns)
                    cum_ret = (1 + port_annual_ret).cumprod()
                    peak = cum_ret.expanding(min_periods=1).max()
                    dd = (cum_ret / peak) - 1
                    max_dd = dd.min()
                    
                    # Neg Years
                    neg_years = port_annual_ret[port_annual_ret < 0]
                    pct_neg = len(neg_years) / len(port_annual_ret) if len(port_annual_ret) > 0 else 0
                    
                    # Store only if it meets criteria
                    if (abs(max_dd) * 100 <= target_max_drawdown) and (pct_neg * 100 <= target_neg_year_pct):
                        results.append({
                           "Weights": dict(zip(selected_tickers, weights)),
                           "CAGR": mean_ret,
                           "Sharpe": sharpe,
                           "MaxDD": max_dd,
                           "PctNegYears": pct_neg,
                           "RawWeights": weights # Store raw array for later use
                       })
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                if not results:
                     st.warning("No portfolios met your strict criteria. Try relaxing the Max Drawdown (e.g., 25%) or Negative Year constraints.")
                else:
                    df = pd.DataFrame(results).sort_values(by="Sharpe", ascending=False)
                    best = df.iloc[0]
                    
                    st.success(f"🏆 Found {len(df)} Valid Portfolios! Best Match:")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Sharpe Ratio", f"{best['Sharpe']:.2f}")
                    c2.metric("Avg Annual Return", f"{best['CAGR']:.2%}")
                    c3.metric("Max Drawdown", f"{best['MaxDD']:.2%}")
                    
                    st.subheader("Optimal Allocation")
                    alloc = {k: v for k, v in best['Weights'].items() if v > 0.01}
                    st.bar_chart(pd.Series(alloc))
                    
                    # --- NEW: Annual Returns Table ---
                    st.subheader("📅 Annual Performance History")
                    
                    # Re-calculate the returns series for the best portfolio
                    best_weights = best['RawWeights']
                    best_annual_series = annual_returns.dot(best_weights)
                    
                    # Create a clean DataFrame for display
                    annual_df = best_annual_series.to_frame(name="Annual Return")
                    annual_df.index = annual_df.index.year.astype(str) # Convert year to string for cleaner display
                    annual_df = annual_df.sort_index(ascending=False)
                    
                    # Add numeric coloring
                    st.bar_chart(annual_df)
                    st.dataframe(annual_df.style.format("{:.2%}"), use_container_width=True)

                    with st.expander("See Top 10 Alternative Portfolios"):
                        st.dataframe(df.head(10)[["CAGR", "Sharpe", "MaxDD", "PctNegYears"]].style.format("{:.2%}"))

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            # st.exception(e) # Uncomment for stack trace
