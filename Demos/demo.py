q1, q3 = np.quantile(data, [0.25, 0.75])

iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

is_outlier = (data < lower_limit) | (data > upper_limit)

outliers = data[is_outlier]
data_without_outliers = data[~is_outlier]
