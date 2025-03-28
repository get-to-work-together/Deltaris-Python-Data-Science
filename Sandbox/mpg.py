import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% Step 1 - load the mpg dataset

df = sns.load_dataset('mpg')

# %% Step 2 - explore the dataset 
    
df.info()

# %% Drop rows with empty cells

df.dropna(inplace=True)
df.info()

# %% Step 3 - Display the distribution of the cars over the origin column

df['origin'].value_counts()

# %% Step 4 - Seperate the car name in a make and a model. 
# Display the distribution of the makes. Fix the typos.

df[['make', 'model']] = df['name'].str.split(n=1, expand=True)

typos = {'chevroelt': 'chevrolet',
         'chevy': 'chevrolet',
         'maxda': 'mazda',
         'mercedes-benz': 'mercedes',
         'toyouta': 'toyota',
         'vokswagen': 'volkswagen',
         'vw': 'volkswagen'}

df['make'] = df['make'].replace(typos)

df['make'].value_counts().sort_index()


# %% Step 5 - Plots

sns.pairplot(data=df, hue='mpg')


# %% Step 6 - MPG

columns_of_interest = ['make','model','origin','model_year','mpg']
df.sort_values('mpg').head(5)[columns_of_interest]

df.sort_values('mpg', ascending=False).head(5)[columns_of_interest]


# %% Step 7 - Convert the weight to a new categorical column

weigth_categories = [0, 2500, 3500, 9999]
labels = ['light', 'medium', 'heavy']
df['weight_category'] = pd.cut(df['weight'], weigth_categories, labels=labels)

df['weight_category'].value_counts()


# %% Step 8 - Create a cross table with origin versus weight

origin_vs_weight = pd.pivot_table(df, 
                                  index = 'origin',
                                  columns = 'weight_category',
                                  aggfunc = 'size',
                                  fill_value = 0)


origin_vs_weight

df[['weight_category','mpg']].groupby('weight_category').agg(['min','mean','max']).round(1)

