# %%

import pandas as pd

# %%

# dict
d = {
    'name': ['Peter','Nienke','Johan','Jebbe'],
    'residence': ['Lhee','Delft','Delft','Delft'],
    }

# dataframe
df = pd.DataFrame(d)

df['wins'] = [4,7,2,3]

# %% EDA - Exporatory Data Analysis

df.info()

print( list(df.columns ) )

nrows, ncols = df.shape

df['residence'].value_counts()

# %%

url = 'https://nl.wikipedia.org/wiki/Provincies_van_Nederland'

df = pd.read_html(url, thousands='.', decimal=',')[0]

# %% ca-500.csv

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

filename = r'Datasets/ca-500.csv'

df = pd.read_csv(filename)

df['city'].value_counts()

selected = df.loc[df['city']=='Montreal', ['first_name','last_name','city','email']]
selected = df.iloc[1:10, [0,1,4,9]]


selected.to_csv('selected.csv')

df2 = df.set_index('email')



























