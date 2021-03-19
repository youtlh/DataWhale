- Check basic statistics `df.describe()`. Gives count, mean, std, min, 25, 50, 75 percentiles, and max. Things to learn:
    - Min-Max range
    - Extreme values 999 9999 -1 might be corrupted data, equals to NaN.
    - Empty column to be removed
    - Highly skewed data due to percentiles.
- Check `df.info()` to check data type

### Target Value Distribution

- Check target value count distribution using `df['target'].value_counts()`
- Check general distribution

```python
import scipy.stats as st
import seaborn as sns
y = train_df['target']
plt.figure(1); plt.title('Default')
sns.distplot(y, rug=True, bins=20)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
```

### Skewness

Measure of symmetry or right/left tailed, degree of distortion from the symmetrical bell curve or the normal distribution. Symmetrical distribution has skewness equals 0. `df.skew()`

- Positive skewness: tail on the right side of the distribution is longer, mean and median larger than mode.
- Negative skewness: tail on the left side of the distribution is longer, mean and median smaller than mode.
- If skewness between 0.5 and -0.5, the data are fairly symmetrical.
- If skewness is between -1 and -0.5 (negatively skewed) or between 0.5 and 1 (positively skewed), the data are moderately skewed.
- If skewness is less than -1 or larger than 1, the data are highly skewed.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/df54b5a8-2a84-48ec-9fdd-4799328527d1/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/df54b5a8-2a84-48ec-9fdd-4799328527d1/Untitled.png)

### Kurtosis

Measure of whether the data are heavy-tailed or light-tailed relative to normal distribution. Datasets with high kurtosis tend to have heavy tails or outliers. `df.kurt()`

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ad05f8dd-cfe4-431f-956b-cb804406fbff/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ad05f8dd-cfe4-431f-956b-cb804406fbff/Untitled.png)

- Mesokurtic: similar to normal distribution. A standard normal distribution has a kurtosis of 3.
- Leptokurtic (**Kurtosis > 3):** distribution is longer, tails are fatter. Peak is higher and sharper than Mesokurtic, which means that data are heavy-tailed or profusion of outliers.
- Platykurtic: distribution is shorter, tails are thinner than the normal distribution. Peak is lower and broader than Mesokurtic, wihch means that data are light-tailed or lack of outliers.

### Generate Data Report

Generate data report using pandas_profiling

```python
import pandas_profiling

pfr = pandas_profiling.ProfileReport(train_df)
pfr.to_file('example.html')
```
