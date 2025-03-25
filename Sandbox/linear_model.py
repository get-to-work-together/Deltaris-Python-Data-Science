import numpy as np
import matplotlib.pyplot as plt


# %%


n = 200

x = np.linspace(0, 10, n)

spread = 3
noise = np.random.randn(n) * spread

a = 4
b = 12
y = a * x + b + noise
# %%


# Linear regresion

coefs = np.polyfit(x, y, deg = 2)
print(coefs)

y_regression = np.polyval(coefs, x)

plt.scatter(x, y)
plt.plot(x, y_regression, color='red')
plt.show()


# %% LinearRegression with scikit learn

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

X = x.reshape(-1, 1)

regressor.fit(X, y)

y_predicted = regressor.predict(X)

plt.scatter(x, y)
plt.plot(x, y_predicted, color='red')
plt.show()

# %% Metrics

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = np.sqrt(mean_squared_error(y, y_predicted))
mae = mean_absolute_error(y, y_predicted)
r_squared = r2_score(y, y_predicted)

print(mse)
print(mae)
print(r_squared)

# %%

data = np.random.normal(10, 2, 1000)
data[421] = 60

plt.violinplot(data, showmeans=True)

plt.show()

# %%

q1, q3 = np.quantile(data, [0.25, 0.75])

iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

is_outlier = (data < lower_limit) | (data > upper_limit)

outliers = data[is_outlier]
data_without_outliers = data[~is_outlier]

plt.boxplot(data_without_outliers)
plt.show()

print(outliers)





