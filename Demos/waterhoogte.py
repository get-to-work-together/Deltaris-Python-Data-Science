# %% imports
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm


# %% Read the data
#
# Getij metingen Terschelling

filename = '../Datasets/20250208_012.csv'


df = pd.read_csv(filename, sep=';', encoding='cp1252', low_memory=False, na_values=999999999)


# %% Exploratory Data Analysis

print(df.info())

print(df['MEETPUNT_IDENTIFICATIE'].unique())


# %% Select waardebepalingsmethode

df['WAARDEBEPALINGSMETHODE_OMSCHRIJVING'].value_counts()

waardebepalingsmethode = 'Astronomische waterhoogte mbv harmonische analyse'
df = df[df['WAARDEBEPALINGSMETHODE_OMSCHRIJVING']== waardebepalingsmethode]


# %% Remove columns NOT of interest

columns_of_interest = ['MEETPUNT_IDENTIFICATIE', 'WAARNEMINGDATUM', 'WAARNEMINGTIJD (MET/CET)', 'NUMERIEKEWAARDE', 'EENHEID_CODE']
columns_to_be_dropped = [col for col in df.columns if col not in columns_of_interest]

df.drop(columns_to_be_dropped, axis=1, inplace=True)


# %% Remove missing values. NUMERIEKE WAARDE == 999999999

NAN_VALUE = 999999999
df[df['NUMERIEKEWAARDE']==NAN_VALUE] = np.nan
df.dropna(inplace=True)


# %% Get timeseries with datetime as index

df['dt'] = pd.to_datetime(df['WAARNEMINGDATUM'] + ' ' + df['WAARNEMINGTIJD (MET/CET)'], dayfirst=True)
df.set_index('dt', inplace=True)
df.sort_index(inplace=True)


# %% Resample

sample_interval = 10 # minutes

waterhoogte = df['NUMERIEKEWAARDE'].resample(f'{sample_interval}min').mean()


# %% plot

def plot_waterhoogte(series):
    from_date = series.index.min()
    to_date = series.index.max()
    
    ax = series.plot()
    ax.set_ylabel('cm')
    ax.set_title(f'Waterhoogten van {from_date.strftime("%d-%m-%Y")} tot {to_date.strftime("%d-%m-%Y")}')
    ax.grid()
    plt.show()
    
plot_waterhoogte(waterhoogte)


# %% last year

year = '2024'
waterhoogte_year = waterhoogte.loc[year]
plot_waterhoogte(waterhoogte_year)


# %% Pandas lag_plot

from pandas.plotting import lag_plot

lag_plot(waterhoogte_year, lag=74)


# %% Format Timedelta to h:mm

def format_timedelta(lag, sample_interval):
    period = pd.Timedelta(lag * sample_interval, 'min')
    d, h, m, *_ = period.components
    return f'{h}:{m:02d} uur'
    

# %% Autoregression

acf = sm.tsa.stattools.acf(waterhoogte_year, nlags=100)

best_lag = np.argmax(acf[6:]) + 6 # start searching 1 hour later

str_period = format_timedelta(best_lag, sample_interval)

sm.graphics.tsaplots.plot_acf(waterhoogte_year, lags=100)
ax = plt.gca()
ax.set_title(f'Best lag: {best_lag} - Getij periode: {str_period} uur')
ax.axvline(best_lag, color='red', alpha=0.8)
plt.show()


# %% Autoregression model

from statsmodels.tsa.ar_model import AutoReg

model = AutoReg(waterhoogte_year, 100)
res = model.fit()
print(res.summary())

# %% Use scipy.signal

peaks, _ = scipy.signal.find_peaks(waterhoogte_year)

delays = peaks[1:] - peaks[:-1]
mean_delay = delays.mean()

str_period = format_timedelta(mean_delay, sample_interval)

print(f'Mean delay: {mean_delay} - Getij periode: {str_period} uur')

# %% FFT

from scipy.fft import fft, ifft

# x = waterhoogte_year.index - waterhoogte_year.index.min()
y = waterhoogte_year.values

n = len(y)
sample_rate = 1/6     # 10 minutes

yf = abs(fft(y))

freq = np.fft.fftfreq(n, sample_rate)

fig, ax = plt.subplots(1, 1, figsize = (14, 5))

ax.plot(freq, yf)

plt.show()

fig, ax = plt.subplots(1, 1, figsize = (14, 5))

ax.plot(freq[:3000], yf[:3000])

plt.show()

max_at = yf.argmax()
max_at

period = 1/freq[max_at]
period

print('Period', format_timedelta(period, 60), 'hours')

# %% Moving Average

window = 6 * 24 * 30

sma = waterhoogte.rolling(window=window).mean()  # 5-day Simple Moving Average
ema = waterhoogte.ewm(span=window).mean()        # Exponential Moving Average with span
# cma = waterhoogte.expanding().mean()             # Cumulative Moving Average

# plot_waterhoogte(waterhoogte)
sma.plot(label='SMA')
ema.plot(label='EMA')
# cma.plot(label='CMA')

ax = plt.gca()
ax.set_ylim(-20, 20)
ax.grid()
ax.legend()

plt.show()
