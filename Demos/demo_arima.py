# %%
import pandas as pd

# %%
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# %%
# Simuleer tijdreeksgegevens

np.random.seed(42)
n = 100  # Aantal tijdstippen
time_series = pd.Series(
    50 + np.cumsum(np.random.normal(0, 1, n)),  # Geaggregeerde ruis om een trend te simuleren
    index=pd.date_range(start="2023-01-01", periods=n, freq="D")  # Dagelijkse data
)

# %%
# Plot de gegenereerde tijdreeks

plt.figure(figsize=(10, 4))
plt.plot(time_series, label="Simulated Time Series")
plt.title("Simulated Time Series")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# ACF en PACF plotten om de parameters te selecteren
plot_acf(time_series)
plot_pacf(time_series)
plt.show()

# %%
# ARIMA Model fitten

model = ARIMA(time_series, order=(1, 1, 1))  # ARIMA(p=1, d=1, q=1)
arima_result = model.fit()

# Toon samenvatting van het model
print(arima_result.summary())

# %%
# Voorspellingen maken

forecast_steps = 10
forecast = arima_result.forecast(steps=forecast_steps)

# Combineer oorspronkelijke data en voorspellingen
forecast_index = pd.date_range(start=time_series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq="D")
forecast_series = pd.Series(forecast, index=forecast_index)

# %%
# Plot de oorspronkelijke data en voorspellingen

plt.figure(figsize=(10, 4))
plt.plot(time_series, label="Original Time Series")
plt.plot(forecast_series, label="Forecast", color="red")
plt.title("ARIMA Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()