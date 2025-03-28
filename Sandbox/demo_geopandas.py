import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# %% Get data (with shape file)

filename = 'Datasets/WijkBuurtkaart_2024_v1/wijkenbuurten_2024_v1.gpkg'

nederland = gpd.read_file(filename, layer='gemeenten')

# %%

nederland.plot()

# %%

nederland.info(verbose=True)

columns_of_interest = ['gemeentecode',
                       'gemeentenaam',
                       'aantal_inwoners',
                       'mannen',
                       'vrouwen',
                       'bevolkingsdichtheid_inwoners_per_km2',
                       'geometry']

# %%


nederland['water'].value_counts()

gemeenten = nederland.loc[nederland['water']=='NEE', columns_of_interest]

gemeenten.iloc[:10, :-1]

gemeenten.plot()

# %%

gemeenten.sort_values('bevolkingsdichtheid_inwoners_per_km2', ascending=False)[['gemeentenaam','bevolkingsdichtheid_inwoners_per_km2']].head(10)

gemeenten.sort_values('bevolkingsdichtheid_inwoners_per_km2', ascending=True)[['gemeentenaam','bevolkingsdichtheid_inwoners_per_km2']].head(10)


# %%

gemeenten.plot('bevolkingsdichtheid_inwoners_per_km2', legend=True)

gemeenten.plot('bevolkingsdichtheid_inwoners_per_km2', legend=True, cmap='Accent')

# %%

gemeenten['man_vrouw'] = gemeenten['mannen'] / gemeenten['vrouwen']
gemeenten.plot('man_vrouw', legend=True, cmap='PiYG')

gemeenten.sort_values('man_vrouw', ascending=False)[['gemeentenaam','man_vrouw']].head(10)
gemeenten.sort_values('man_vrouw', ascending=True)[['gemeentenaam','man_vrouw']].head(10)
