# app_montecarlo.py
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.optimize import minimize

st.set_page_config(page_title="Simulación Monte Carlo de Acciones", layout="wide")

st.title("Simulación Monte Carlo con EWMA y Stress Adaptativo")

# -----------------------------
# Input del usuario
# -----------------------------
ticker = st.text_input("Ingrese el ticker (ej: WMT, AAPL, GGAL.BA):", value="WMT")
start_date = st.date_input("Fecha inicio", value=dt.date(2023,10,10))
end_date = st.date_input("Fecha fin", value=dt.date.today())

if start_date >= end_date:
    st.error("La fecha de inicio debe ser anterior a la fecha final.")
    st.stop()

# -----------------------------
# Descargar datos
# -----------------------------
with st.spinner("Descargando datos..."):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

if "Adj Close" in df.columns:
    prices = df["Adj Close"].dropna()
elif "Close" in df.columns:
    prices = df["Close"].dropna()
else:
    st.error("No se encontró columna de precios válida.")
    st.stop()

if prices.empty or len(prices) < 60:
    st.error("Pocos datos históricos (se recomiendan > 60 observaciones).")
    st.stop()

# -----------------------------
# Retornos logarítmicos
# -----------------------------
returns = np.log(prices / prices.shift(1)).dropna().values.ravel()

# -----------------------------
# Calcular lambda EWMA
# -----------------------------
def ewma_vol(returns, lam):
    vol2 = np.empty_like(returns)
    vol2[0] = returns[0]**2
    for t in range(1, len(returns)):
        vol2[t] = lam * vol2[t-1] + (1 - lam) * returns[t]**2
    return np.sqrt(vol2)

window = 21
vol_hist = np.array([np.std(returns[max(0, i-window+1):i+1]) for i in range(len(returns))])

def objective(lam_array):
    lam = lam_array[0]
    vol_ewma = ewma_vol(returns, lam)
    return np.sum((vol_hist - vol_ewma)**2)

res = minimize(objective, x0=[0.94], bounds=[(0.85, 0.99)])
lambda_opt = float(res.x[0])

# -----------------------------
# Calcular stress adaptativo
# -----------------------------
vol_daily = np.std(returns)
vol_annual = vol_daily * np.sqrt(252)
q01, q05, q95, q99 = np.percentile(returns, [1, 5, 95, 99])
eps = 1e-12
tail_ratio = abs(q05) / (abs(q95) + eps)
vol_factor = np.clip(vol_annual / 0.25, 0.5, 2.0)

stress_down = float(np.clip(1.0 + (tail_ratio * vol_factor * 0.35), 1.0, 2.0))
stress_up   = float(np.clip(1.0 - (1.0 / (tail_ratio + eps)) * vol_factor * 0.235, 0.4, 1.0))

# -----------------------------
# Simulación Monte Carlo
# -----------------------------
last_price = float(prices.iloc[-1])
days = 252
n_simulations = 5000
np.random.seed(42)

weights = np.array([(1 - lambda_opt) * (lambda_opt ** i) for i in reversed(range(len(returns)))])
weights /= weights.sum()
mu_ewma = float(np.sum(returns * weights))

rolling_vol = np.array([np.std(returns[max(0, i-window+1):i+1]) for i in range(len(returns))])
if len(rolling_vol) < days:
    rolling_vol = np.pad(rolling_vol, (0, days - len(rolling_vol)), 'edge')

sim_prices_up = np.zeros((days, n_simulations))
sim_prices_down = np.zeros((days, n_simulations))

for sim in range(n_simulations):
    daily_returns = np.random.choice(returns, size=days, replace=True)

    daily_returns_up = np.where(daily_returns > 0, daily_returns * stress_up, daily_returns)
    vol_choice = np.random.choice(rolling_vol, size=days, replace=True)
    vol_adjusted_up = mu_ewma + daily_returns_up * (vol_choice / (returns.std() + eps))
    sim_prices_up[:, sim] = last_price * np.exp(np.cumsum(vol_adjusted_up))

    daily_returns_down = np.where(daily_returns < 0, daily_returns * stress_down, daily_returns)
    vol_choice = np.random.choice(rolling_vol, size=days, replace=True)
    vol_adjusted_down = mu_ewma + daily_returns_down * (vol_choice / (returns.std() + eps))
    sim_prices_down[:, sim] = last_price * np.exp(np.cumsum(vol_adjusted_down))

p90_up = np.percentile(sim_prices_up, 90, axis=1)
p10_down = np.percentile(sim_prices_down, 10, axis=1)
p50_base = (p90_up + p10_down) / 2.0

# -----------------------------
# Gráfico
# -----------------------------
last_date = prices.index[-1]
future_dates = [last_date + dt.timedelta(days=i) for i in range(1, days+1)]

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(prices.index, prices.values, color="gray", label="Precio histórico")
ax.fill_between(future_dates, p10_down, p90_up, color="khaki", alpha=0.4)
ax.plot(future_dates, p50_base, color="orange", linewidth=2, label="Escenario base (promedio)")
ax.plot(future_dates, p90_up, color="green", linestyle="--", label="Optimista (p90)")
ax.plot(future_dates, p10_down, color="red", linestyle="--", label="Pesimista (p10)")

ax.set_title(f"{ticker} - Monte Carlo (λ={lambda_opt:.3f}, down={stress_down:.2f}, up={stress_up:.2f})", fontsize=14)
ax.set_xlabel("Fecha")
ax.set_ylabel("Precio (USD)")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# -----------------------------
# Resumen final
# -----------------------------
price_high = p90_up[-1]
price_base = p50_base[-1]
price_low = p10_down[-1]

pct_high = (price_high / last_price - 1) * 100
pct_base = (price_base / last_price - 1) * 100
pct_low = (price_low / last_price - 1) * 100

st.subheader("Resumen final")
st.write(f"Precio actual: {last_price:.2f} USD")
st.write(f"Optimista (p90): {price_high:.2f} USD  (+{pct_high:.2f}%)")
st.write(f"Base (promedio): {price_base:.2f} USD  ({pct_base:.2f}%)")
st.write(f"Pesimista (p10): {price_low:.2f} USD  ({pct_low:.2f}%)")
st.write(f"Lambda EWMA: {lambda_opt:.4f}")
st.write(f"Stress_down: {stress_down:.3f}   Stress_up: {stress_up:.3f}")
