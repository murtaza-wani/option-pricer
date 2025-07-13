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

# PnL parameters
st.sidebar.markdown("---")
st.sidebar.header("PnL Parameters")
call_purchase_price = st.sidebar.number_input("Call Purchase Price", min_value=0.0, value=0.0)
put_purchase_price = st.sidebar.number_input("Put Purchase Price", min_value=0.0, value=0.0)


st.title("Option Pricing App")

# Session state to track if calculation has been done at least once
if "calculated" not in st.session_state:
    st.session_state.calculated = False

# Handle calculate button
if calculate:
    st.session_state.calculated = True

# Only display results if calculated at least once
if st.session_state.calculated:
    
    epsilon = 0.1
    spot_prices = []
    
    call_deltas = []
    put_deltas = []
    call_gammas = []
    put_gammas = []
    call_thetas = []
    put_thetas = []
    call_vegas = []
    put_vegas = []
    call_rhos = []
    put_rhos = []
    
    for y in range(-1000, 1001, 1):
        S_i = S + y * epsilon
        if S_i <=epsilon:
            continue
        spot_prices.append(S_i)

    # --- Main call/put price ---
    if model == "Black-Scholes":
        call_price, put_price = black_scholes(S, K, T, r, sigma)
        #Delta
        for s in spot_prices:
            call_up, put_up = black_scholes(s + epsilon, K, T, r, sigma)
            call_down, put_down = black_scholes(s - epsilon, K, T, r, sigma)
            call_delta = (call_up - call_down) / (2 * epsilon)
            put_delta = (put_up - put_down) / (2 * epsilon)
            call_deltas.append(call_delta)
            put_deltas.append(put_delta)
        #Gamma
        for s in spot_prices:
            call_up, put_up = black_scholes(s + epsilon, K, T, r, sigma)
            call_down, put_down = black_scholes(s - epsilon, K, T, r, sigma)
            call, put = black_scholes(s, K, T, r, sigma)
            call_gamma = (call_up - 2 * call + call_down) / (epsilon ** 2)
            put_gamma = (put_up - 2 * put + put_down) / (epsilon ** 2)
            call_gammas.append(call_gamma)
            put_gammas.append(put_gamma)
        #Theta
        for s in spot_prices:
            call_down, put_down = black_scholes(s, K, T - epsilon , r, sigma)
            call, put = black_scholes(s, K, T, r, sigma)
            call_theta = (call_down - call) / epsilon
            put_theta = (put_down -  put) / epsilon
            call_thetas.append(call_theta)
            put_thetas.append(put_theta)
       #Sigma
        for s in spot_prices:
            call_up, put_up = black_scholes(s, K, T, r, sigma + 0.01)
            call_down, put_down = black_scholes(s, K, T, r, sigma - 0.01)
            call_vega = (call_up - call_down) / (2 * 0.01)
            put_vega = (put_up - put_down) / (2 * 0.01)
            call_vegas.append(call_vega)
            put_vegas.append(put_vega)
       #Rho
        for s in spot_prices:
            call_up, put_up = black_scholes(s, K, T, r + 0.01, sigma)
            call_down, put_down = black_scholes(s, K, T, r - 0.01, sigma)
            call_rho = (call_up - call_down) / (2 * 0.01)
            put_rho = (put_up - put_down) / (2 * 0.01)
            call_rhos.append(call_rho)
            put_rhos.append(put_rho)
            
    elif model == "Monte Carlo":
        call_price, put_price = monte_carlo(S, K, T, r, sigma, num_simulations)     
        #Delta
        for s in spot_prices:
            call_up, put_up = monte_carlo(s + epsilon, K, T, r, sigma)
            call_down, put_down = monte_carlo(s - epsilon, K, T, r, sigma)
            call_delta = (call_up - call_down) / (2 * epsilon)
            put_delta = (put_up - put_down) / (2 * epsilon)
            call_deltas.append(call_delta)
            put_deltas.append(put_delta)
        #Gamma
        for s in spot_prices:
            call_up, put_up = monte_carlo(s + epsilon, K, T, r, sigma)
            call_down, put_down = monte_carlo(s - epsilon, K, T, r, sigma)
            call, put = monte_carlo(s, K, T, r, sigma)
            call_gamma = (call_up - 2 * call + call_down) / (epsilon ** 2)
            put_gamma = (put_up - 2 * put + put_down) / (epsilon ** 2)
            call_gammas.append(call_gamma)
            put_gammas.append(put_gamma)
        #Theta
        for s in spot_prices:
            call_down, put_down = monte_carlo(s, K, T - epsilon , r, sigma)
            call, put = monte_carlo(s, K, T, r, sigma )
            call_theta = (call_down - call) / epsilon
            put_theta = (put_down - put) / epsilon
            call_thetas.append(call_theta)
            put_thetas.append(put_theta)
       #Sigma
        for s in spot_prices:
            call_up, put_up = monte_carlo(s, K, T, r, sigma + 0.01)
            call_down, put_down = monte_carlo(s, K, T, r, sigma - 0.01)
            call_vega = (call_up - call_down) / (2 * 0.01)
            put_vega = (put_up - put_down) / (2 * 0,.01)
            call_vegas.append(call_vega)
            put_vegas.append(put_vega)
       #Rho
        for s in spot_prices:
            call_up, put_up = monte_carlo(s, K, T, r + 0.01, sigma)
            call_down, put_down = monte_carlo(s, K, T, r - 0.01, sigma)
            call_rho = (call_up - call_down) / (2 * 0.01)
            put_rho = (put_up - put_down) / (2 * 0.01)
            call_rhos.append(call_rho)
            put_rhos.append(put_rho)
    elif model == "Binomial":
        call_price, put_price = binomial_model(S, K, T, r, sigma, N)
        #Delta
        for s in spot_prices:
            call_up, put_up = binomial_model(s + epsilon, K, T, r, sigma)
            call_down, put_down = binomial_model(s - epsilon, K, T, r, sigma)
            call_delta = (call_up - call_down) / (2 * epsilon)
            put_delta = (put_up - put_down) / (2 * epsilon)
            call_deltas.append(call_delta)
            put_deltas.append(put_delta)
        #Gamma
        for s in spot_prices:
            call_up, put_up = binomial_model(s + epsilon, K, T, r, sigma)
            call_down, put_down = binomial_model(s - epsilon, K, T, r, sigma)
            call, put = binomial_model(s, K, T, r, sigma)
            call_gamma = (call_up - 2 * call + call_down) / (epsilon ** 2)
            put_gamma = (put_up - 2 * put + put_down) / (epsilon ** 2)
            call_gammas.append(call_gamma)
            put_gammas.append(put_gamma)
        #Theta
        for s in spot_prices:
            call_down, put_down = binomial_model(s, K, T - epsilon , r, sigma)
            call, put = binomial_model(s, K, T, r, sigma )
            call_theta = (call_down - call) / epsilon
            put_theta = (put_down - put) / epsilon
            call_thetas.append(call_theta)
            put_thetas.append(put_theta)
       #Sigma
        for s in spot_prices:
            call_up, put_up = binomial_model(s, K, T, r, sigma + 0.01)
            call_down, put_down = binomial_model(s, K, T, r, sigma - 0.01)
            call_vega = (call_up - call_down) / (2 * 0.01)
            put_vega = (put_up - put_down) / (2 * 0.01)
            call_vegas.append(call_vega)
            put_vegas.append(put_vega)
       #Rho
        for s in spot_prices:
            call_up, put_up = binomial_model(s, K, T, r + 0.01, sigma)
            call_down, put_down = binomial_model(s, K, T, r - 0.01, sigma)
            call_rho = (call_up - call_down) / (2 * 0.01)
            put_rho = (put_up - put_down) / (2 * 0.01)
            call_rhos.append(call_rho)
            put_rhos.append(put_rho)
            
    st.markdown(f"### Model : {model}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("###### Call Option Price 📈 ")
        st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:#e6f4ea; font-size:24px;'>${call_price:.2f}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("###### Put Option Price 📉 ")
        st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:#fdecea; font-size:24px;'>${put_price:.2f}</div>", unsafe_allow_html=True)
   
   # Add vertical spacing
    st.markdown("<br><hr>", unsafe_allow_html=True)  
    st.markdown("### Greeks(using finite differences)")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("###### Delta vs Spot Price (Call)")
        fig_call, ax_call = plt.subplots(figsize=(5, 3))
        ax_call.plot(spot_prices, call_deltas, label="Call Delta", color="blue")
        ax_call.axhline(0, color="gray", linestyle="--")
        ax_call.set_xlabel("Spot Price")
        ax_call.set_ylabel("Delta")
        ax_call.grid(True)
        st.pyplot(fig_call)
        
        st.markdown("<br><hr>", unsafe_allow_html=True) 
        
        st.markdown("###### Gamma vs Spot Price (Call)")
        fig_call, ax_call = plt.subplots(figsize=(5, 3))
        ax_call.plot(spot_prices, call_gammas, label="Call Gamma", color="blue")
        ax_call.axhline(0, color="gray", linestyle="--")
        ax_call.set_xlabel("Spot Price")
        ax_call.set_ylabel("Gamma")
        ax_call.grid(True)
        st.pyplot(fig_call)
        
        st.markdown("<br><hr>", unsafe_allow_html=True) 
        
        st.markdown("###### Theta vs Time (Call)")
        fig_call, ax_call = plt.subplots(figsize=(5, 3))
        ax_call.plot(spot_prices, call_thetas, label="Call Theta", color="blue")
        ax_call.axhline(0, color="gray", linestyle="--")
        ax_call.set_xlabel("Time")
        ax_call.set_ylabel("Theta")
        ax_call.grid(True)
        st.pyplot(fig_call)

        st.markdown("<br><hr>", unsafe_allow_html=True) 
        
        st.markdown("###### Vega vs Spot price (Call)")
        fig_call, ax_call = plt.subplots(figsize=(5, 3))
        ax_call.plot(spot_prices, call_vegas, label="Call Vega", color="blue")
        ax_call.axhline(0, color="gray", linestyle="--")
        ax_call.set_xlabel("Spot Price")
        ax_call.set_ylabel("Vega")
        ax_call.grid(True)
        st.pyplot(fig_call)
        
        st.markdown("<br><hr>", unsafe_allow_html=True) 
        
        st.markdown("###### Rho vs Spot price (Call)")
        fig_call, ax_call = plt.subplots(figsize=(5, 3))
        ax_call.plot(spot_prices, call_rhos, label="Call Rho", color="blue")
        ax_call.axhline(0, color="gray", linestyle="--")
        ax_call.set_xlabel("Spot Price")
        ax_call.set_ylabel("Rho")
        ax_call.grid(True)
        st.pyplot(fig_call)
        
        st.markdown("<br><hr>", unsafe_allow_html=True) 
    with col4: 
        st.markdown("###### Delta vs Spot Price (Put)")
        fig_put, ax_put = plt.subplots(figsize=(5, 3))
        ax_put.plot(spot_prices, put_deltas, label="Put Delta", color="red")
        ax_put.axhline(0, color="gray", linestyle="--")
        ax_put.set_xlabel("Spot Price")
        ax_put.set_ylabel("Delta")
        ax_put.grid(True)
        st.pyplot(fig_put)
        
        st.markdown("<br><hr>", unsafe_allow_html=True) 
        
        st.markdown("###### Gamma vs Spot Price (Put)")
        fig_put, ax_put = plt.subplots(figsize=(5, 3))
        ax_put.plot(spot_prices, put_gammas, label="Put Gamma", color="red")
        ax_put.axhline(0, color="gray", linestyle="--")
        ax_put.set_xlabel("Spot Price")
        ax_put.set_ylabel("Gamma")
        ax_put.grid(True)
        st.pyplot(fig_put)
        
        st.markdown("<br><hr>", unsafe_allow_html=True) 
        
        st.markdown("###### Theta vs Time (Put)")
        fig_put, ax_put = plt.subplots(figsize=(5, 3))
        ax_put.plot(spot_prices, put_thetas, label="Put Theta", color="red")
        ax_put.axhline(0, color="gray", linestyle="--")
        ax_put.set_xlabel("Time")
        ax_put.set_ylabel("Theta")
        ax_put.grid(True)
        st.pyplot(fig_put)

        st.markdown("<br><hr>", unsafe_allow_html=True) 
        
        st.markdown("###### Vega vs Spot Price (Put)")
        fig_put, ax_put = plt.subplots(figsize=(5, 3))
        ax_put.plot(spot_prices, put_vegas, label="Put Vega", color="red")
        ax_put.axhline(0, color="gray", linestyle="--")
        ax_put.set_xlabel("Spot Price")
        ax_put.set_ylabel("Vega")
        ax_put.grid(True)
        st.pyplot(fig_put)
        
        st.markdown("<br><hr>", unsafe_allow_html=True) 
        
        st.markdown("###### Rho vs Spot Price (Put)")
        fig_put, ax_put = plt.subplots(figsize=(5, 3))
        ax_put.plot(spot_prices, put_rhos, label="Put Rho", color="red")
        ax_put.axhline(0, color="gray", linestyle="--")
        ax_put.set_xlabel("Spot Price")
        ax_put.set_ylabel("Rho")
        ax_put.grid(True)
        st.pyplot(fig_put)
        
        st.markdown("<br><hr>", unsafe_allow_html=True) 
        
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
            
                # --- PnL Heatmaps ---
            st.markdown("<br><hr><br>", unsafe_allow_html=True)
            st.subheader("Profit and Loss (PnL) Heatmap")

            call_pnl = call_matrix - call_purchase_price
            put_pnl = put_matrix - put_purchase_price

            fig3, ax3 = plt.subplots(figsize=(8, 6))
            sns.heatmap(call_pnl, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2),
                        cmap="RdYlGn", center=0, ax=ax3, annot=True, fmt=".1f", annot_kws={"size": 8}, 
                        cbar_kws={'label': 'Call PnL'})
            ax3.set_xlabel("Spot Price")
            ax3.set_ylabel("Volatility")
            ax3.set_title("Call Option PnL Heatmap")
            plt.xticks(rotation=45)

            fig4, ax4 = plt.subplots(figsize=(8, 6))
            sns.heatmap(put_pnl, xticklabels=np.round(spot_prices, 2), yticklabels=np.round(volatilities, 2),
                        cmap="RdYlGn", center=0, ax=ax4, annot=True, fmt=".1f", annot_kws={"size": 8}, 
                        cbar_kws={'label': 'Put PnL'})
            ax4.set_xlabel("Spot Price")
            ax4.set_ylabel("Volatility")
            ax4.set_title("Put Option PnL Heatmap")
            plt.xticks(rotation=45)

            col5, col6 = st.columns(2)
            with col5:
                st.pyplot(fig3)
            with col6:
                st.pyplot(fig4)

if not st.session_state.calculated:
    st.info("Enter option pricing inputs in the sidebar and click **Calculate Option Price** to get started.")