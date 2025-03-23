# %% imports
import pandas as pd

import numpy as np
from lifelines import KaplanMeierFitter

# %% Simuleer gegevens

np.random.seed(42)
n = 100  # Aantal patiënten
data = {
    "patient_id": range(1, n + 1),
    "time_to_event": np.random.exponential(scale=10, size=n),  # Tijd tot gebeurtenis
    "event_occurred": np.random.choice([1, 0], size=n, p=[0.7, 0.3])  # 1 = gebeurtenis, 0 = gecensureerd
}

# %% Maak een DataFrame
df = pd.DataFrame(data)

# Bekijk een voorbeeld van de data
print(df.head())

# %% Voer survival analysis uit
kmf = KaplanMeierFitter()

# Pas de Kaplan-Meier methode toe
kmf.fit(df["time_to_event"], event_observed=df["event_occurred"])

# %% Toon de resultaten
kmf.plot_survival_function()
kmf.print_summary()