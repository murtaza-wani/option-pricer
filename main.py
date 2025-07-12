import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models.black_scholes import black_scholes
from models.monte_carlo import monte_carlo
from models.binomial import binomial_model

st.set_page_config(page_title="Option Pricing Model", layout="wide")
st.sidebar.title("Option Pricing Inputs")

# Model selection
model = st.sidebar.selectbox("Choose Pricing Model", ["Black-Scholes", "Monte Carlo", "Binomial"])

# Common inputs
S = st.sidebar.number_input("Stock Price (S)", min_value=0.0, value=100.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0)
T = st.sidebar.number_input("Time to Expiry (T, in years)", min_value=0.01, value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.2)

# Model-specific inputs
if model == "Monte Carlo":
    num_simulations = st.sidebar.number_input("Number of Simulations", min_value=1000, step=1000, value=10000)
if model == "Binomial":
    N = st.sidebar.number_input("Number of Steps (N)", min_value=1, max_value=1000, value=100)

calculate = st.sidebar.button("Calculate Option Price")

# Heatmap parameters
st.sidebar.markdown("---")
st.sidebar.header("Heatmap Parameters")
min_spot = st.sidebar.number_input("Minimum Spot Price", value=80.0)
max_spot = st.sidebar.number_input("Maximum Spot Price", value=120.0)

min_vol = st.sidebar.slider("Minimum Volatility", min_value=0.01, max_value=1.0, value=0.1)
max_vol = st.sidebar.slider("Maximum Volatility", min_value=0.01, max_value=1.0, value=0.5)

st.title("Option Pricing App")

# Session state to track if calculation has been done at least once
if "calculated" not in st.session_state:
    st.session_state.calculated = False

# Handle calculate button
if calculate:
    st.session_state.calculated = True

# Only display results if calculated at least once
if st.session_state.calculated:

    # --- Main call/put price ---
    if model == "Black-Scholes":
        call_price, put_price = black_scholes(S, K, T, r, sigma)
    elif model == "Monte Carlo":
        call_price, put_price = monte_carlo(S, K, T, r, sigma, num_simulations)
    elif model == "Binomial":
        call_price, put_price = binomial_model(S, K, T, r, sigma, N)
    
    st.markdown(f"### {model}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Call Option Price")
        st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:#e6f4ea; font-size:24px;'>${call_price:.2f}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("### 📉 Put Option Price")
        st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:#fdecea; font-size:24px;'>${put_price:.2f}</div>", unsafe_allow_html=True)

    # Add vertical spacing
    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # --- Heatmap ---
    if min_spot >= max_spot:
        st.error("Minimum Spot Price should be less than Maximum Spot Price.")
    elif min_vol >= max_vol:
        st.error("Minimum Volatility should be less than Maximum Volatility.")
    
    elif min_spot < max_spot and min_vol < max_vol:
            spot_prices = np.linspace(min_spot, max_spot, 10)
            volatilities = np.linspace(min_vol, max_vol, 10)

            call_matrix = np.zeros((len(volatilities), len(spot_prices)))
            put_matrix = np.zeros((len(volatilities), len(spot_prices)))

            for i, vol in enumerate(volatilities):
                for j, spot in enumerate(spot_prices):
                    if model == "Black-Scholes":
                        c, p = black_scholes(spot, K, T, r, vol)
                    elif model == "Monte Carlo":
                        c, p = monte_carlo(spot, K, T, r, vol, num_simulations)
                    elif model == "Binomial":
                        c, p = binomial_model(spot, K, T, r, vol, N)
                    call_matrix[i, j] = c
                    put_matrix[i, j] = p

            st.subheader("Option Price Sensitivity Heatmap")

            fig1, ax1 = plt.subplots(figsize=(8, 6))
            sns.heatmap(call_matrix, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2),
                        cmap="YlGnBu", ax=ax1, annot=True, fmt=".1f", annot_kws={"size": 8}, cbar_kws={'label': 'Call Price'})
            ax1.set_xlabel("Spot Price")
            ax1.set_ylabel("Volatility")
            ax1.set_title("Call Option Heatmap")
            plt.xticks(rotation=45)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(put_matrix, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2),
                        cmap="YlGnBu", ax=ax2, annot=True, fmt=".1f", annot_kws={"size": 8}, cbar_kws={'label': 'Put Price'})
            ax2.set_xlabel("Spot Price")
            ax2.set_ylabel("Volatility")
            ax2.set_title("Put Option Heatmap")
            plt.xticks(rotation=45)

            col3, col4 = st.columns(2)
            with col3:
                st.pyplot(fig1)
            with col4:
                st.pyplot(fig2)
    else:
        st.info("Enter inputs in the sidebar and click **Calculate Option Price** to begin.")
