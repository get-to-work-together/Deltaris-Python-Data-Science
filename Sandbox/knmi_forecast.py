import pandas as pd
import matplotlib.pyplot as plt


# %%

filename = 'Datasets/knmi.csv'

df = pd.read_csv(filename, skiprows=10, skipinitialspace=True)

df.rename(columns={'# STN': 'STN'}, inplace=True)

# %%

df.info()

# %%

df.head()

# %%

df['TEMPERATURE'] = df['TG']/10
df['DD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
df = df.set_index('DD')

# %%

df_train = df.loc['2020-01-01':'2023-12-31', ['TEMPERATURE']]
df_test = df.loc['2024-01-01':'2024-12-31', ['TEMPERATURE']]

fig, ax = plt.subplots()
ax.plot(df_train['TEMPERATURE'], color='C0')
ax.plot(df_test['TEMPERATURE'], color='C1')
plt.show()

# %%

df_train['day_of_year'] = df_train.index.day_of_year
df_test['day_of_year'] = df_test.index.day_of_year

# %%

features = ['day_of_year']
target = 'TEMPERATURE'

X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

# %%

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)

y_predicted = regressor.predict(X_test)

df_test['predicted'] = y_predicted

# %%

fig, ax = plt.subplots()
ax.plot(df_train['TEMPERATURE'], color='C0')
ax.plot(df_test['TEMPERATURE'], color='C1')
ax.plot(df_test['predicted'], color='red')
plt.show()

# %%

from sklearn.metrics import r2_score, mean_absolute_error

print('R squared:', r2_score(y_test, y_predicted))
print('MAE:', mean_absolute_error(y_test, y_predicted))
