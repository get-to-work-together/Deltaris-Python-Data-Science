import pandas as pd
import locale

locale.setlocale(locale.LC_ALL, 'nl_NL')

# %%

url = 'https://wetenschap.infonu.nl/sterrenkunde/149193-wanneer-is-het-volle-maan-in-2020-tot-en-met-2025.html'

dfs = pd.read_html(url)
for year, df in enumerate(dfs, start=2020):    

    datums = df['Volle maan'].str.title().str.zfill(6) + '-' + str(year)
    tijdstippen = df['Tijdstip.2'].str.zfill(5)
    timestamps = datums + ' ' + tijdstippen

    df['volle_maan'] = pd.to_datetime(timestamps, format='%d-%b-%Y %H:%M') # requires correct locale nl_NL
    

series_volle_maan = pd.concat(dfs, axis=0)['volle_maan'].dropna()

series_volle_maan.to_csv('Sandbox/volle_maan.csv', index=False)

# %%

df_test = pd.DataFrame({'dd': pd.date_range('2025-03-20', periods=100)})

# %%

merged = pd.merge_asof(df_test.sort_values('dd'), 
                       series_volle_maan.sort_values(),
                       left_on='dd',
                       right_on='volle_maan', 
                       direction='backward')

merged['sinds_laatste_volle_maan'] = merged['dd'] - merged['volle_maan']

