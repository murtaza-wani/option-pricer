import numpy as np

def binomial_model(S, K, T, r, sigma, N=100):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Terminal stock prices
    ST = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

    # Call and Put payoffs at maturity
    call_values = np.maximum(ST - K, 0)
    put_values = np.maximum(K - ST, 0)

    # Work backwards for call
    for i in range(N - 1, -1, -1):
        call_values = np.exp(-r * dt) * (p * call_values[1:i+2] + (1 - p) * call_values[0:i+1])
        put_values = np.exp(-r * dt) * (p * put_values[1:i+2] + (1 - p) * put_values[0:i+1])

    return call_values[0], put_values[0]