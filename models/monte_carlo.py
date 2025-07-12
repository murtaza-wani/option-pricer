import numpy as np

def monte_carlo(S, K, T, r, sigma, num_simulations=10000):
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # if option_type == 'call':
    #     payoffs = np.maximum(ST - K, 0)
    # else:
    #     payoffs = np.maximum(K - ST, 0)
    call_payoffs =  np.maximum(ST - K, 0)
    put_payoffs = np.maximum(K - ST, 0)

    call_option_price = np.exp(-r * T) * np.mean(call_payoffs)
    put_option_price = np.exp(-r * T) * np.mean(put_payoffs)
    return call_option_price, put_option_price