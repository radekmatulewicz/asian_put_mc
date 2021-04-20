import numpy as np
import scipy.stats as si


def asian_putMC(S0, r, sigma, T, K, n, M):
    dt = T / n
    A = np.zeros(M)
    G = np.zeros(M)


    # Monte Carlo
    for j in range(M):
        w = np.random.standard_normal(n - 1)
        Z = np.zeros(n)
        Z[0] = 0
        Z[1:] = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * w
        logS = np.log(S0) + np.cumsum(Z)

        arithmetic = np.mean(np.exp(logS))
        geometric = np.exp(np.mean(logS))

        A[j] = np.exp(-r * T) * max(0, K - arithmetic)
        G[j] = np.exp(-r * T) * max(0, K - geometric)

    # Cena azjatyckiej opcji put z dyskretną średnią geometryczną, analitycznie:
    muG = (r - 0.5 * sigma ** 2) * T * (n + 1) / (2 * n)
    sigmaG = (sigma ** 2 * T * (n + 1) * (2 * n + 1)) / (6 * n ** 2)
    b1 = (np.log(S0 / K) - muG) / np.sqrt(sigmaG)
    b2 = (np.log(S0 / K) - muG - sigmaG) / np.sqrt(sigmaG)
    V0 = np.exp(-r * T) * K * si.norm.cdf(b1) - S0 * np.exp(-r * T + muG + sigmaG / 2) * si.norm.cdf(b2)

    # Zmienne kontrolne, redukcja wariancji
    cov = np.cov(A, G)
    theta = cov[0, 1] / cov[1, 1]
    X = A - theta * (G - V0)

    # Obliczanie ceny i 95% przedzialu ufnosci z redukcja wariancji
    price = np.sum(X) / M
    bm2 = 1 / (M - 1) * sum((X - price) ** 2)
    ci = [price - 1.96 * np.sqrt(bm2) / np.sqrt(M), price + 1.96 * np.sqrt(bm2) / np.sqrt(M)]

    # Obliczanie ceny i 95% przedzialu ufnosci bez redukcji wariancji
    priceAM = np.sum(A) / M
    bm = 1 / (M - 1) * sum((A - priceAM) ** 2)
    ci_am = [priceAM - 1.96 * np.sqrt(bm) / np.sqrt(M), priceAM + 1.96 * np.sqrt(bm) / np.sqrt(M)]

    return price, ci, priceAM, ci_am
