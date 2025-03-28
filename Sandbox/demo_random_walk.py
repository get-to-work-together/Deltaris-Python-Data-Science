import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator
from matplotlib.dates import ConciseDateFormatter

import locale

locale.setlocale(locale.LC_ALL, 'pt_PT')

# %% generate data

n = 7
dates = pd.date_range('2025-03-28', periods=n, freq='d').date
random_walk = np.cumsum(np.random.randint(-2, 3, size=n))

df = pd.DataFrame({'dd': dates,
                   'values': random_walk}).set_index('dd')

# %% plot

locator = AutoDateLocator()
formatter = ConciseDateFormatter(locator)

fig, ax = plt.subplots()

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

ax.plot(df)
plt.show()
