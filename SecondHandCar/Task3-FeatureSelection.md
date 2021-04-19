## Anomaly Handling

Note that anomaly can only be deleted for training set, never mistakenly delete something for test set.

### Box plot method

IQR is the box length, whisker extends to both side for the 1.5 times IQR. All data points outside these 1.5*IQR values are flagged as outliers.

```python
def outliers_proc(data, col_name, scale=3):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :return:
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    return data_n
```

### 3-sigma Method

Simply delete all the points that are 3 times std away

### BOX-COX Transform

Transformation of non-normal dependent variables into a normal shape.

The main variable is an exponent, lambda ($\lambda$), which varies from -5 to 5. If the optimal value for lambda is 1, then the data is already normally distributed, since lambda of 1 only shifts the data downward without changing the shape. The transformation for variable x has the form for positive data

$X_t(\lambda) = \begin{cases}
   \frac{x_t^\lambda-1}{\lambda} &\text{if } \lambda \ne 0 \\
   logx_t &\text{if } \lambda = 0
\end{cases}$

For negative data:

$X(\lambda) = \begin{cases}
   \frac{(x+\lambda_2)^{\lambda_1}-1}{\lambda_1} &\text{if } \lambda_1 \ne 0 \\
   log(x+\lambda_2) &\text{if } \lambda_1 = 0
\end{cases}$

Scipy has a boxcox function that choose the optimal value of lambda for us `scipy.stats.boxcox()`. To reverse a transformed function into its original scale, use `scipy.special.inv_boxcox(x, lambda)`.

Limitation: Should not be used if interpretation is the goal.

## Feature Standardization

Note: normalize only applies on each row, not on vertical feature columns

### MinMaxScalar

$\bar{x} = \frac{x-x_{min}}{x_{max}-x{min}}$. Target to scale. Should be used first unless have theoretical reason to need stronger handling. Preserves the shape of the original distribution. Does not reduce the importance of outliers.

```python
from sklearn import preprocessing
normalized = preprocessing.normalize(x)
# OR
normalized = preprocessing.MinMaxScalar()
scaled = scaler.fit_transform(x)
```

### RobustScalar

Target to Standardize. Used if have outliers and don't want them to have much influence. Subtract the median and divide by the IQR. Does not scale data into a predefined interval.

### StardardScalar

Subtract the mean and scale to unit variance. Used when need to transform a feature so that it is close to normally distributed. Return a distribution with std = 1, variance = 1 ⇒ change a distribution to normal distribution.

### Power Law Distribution Handling

Power Law distribution is a special phenomenon that a few items are clustered at the top (or bottom) of a distribution, taken up 95% or more of the resources. Formula to handle that is $log(\frac{1+x}{1+median})$

## Continuous Features
### Data Binning

Statistical data binning is a way to group numbers of more or less continuous values into a smaller number of "bins".

Advantages

- Sparse variable accelerates the computation speed, make the output easier to store
- Decrease the influence of extreme values, original values can easily been grouped with x>30, thus 200 will not greatly deviate the model
- After discretization, each bin comes with weight. This is equal to introduces non-linearity, enhance the fitting result
- Cross-ref the variables such that M+N goes to M*N combinations, once again introduces non-linearity
- Model more stable, will not be easily influence by small variations, as long as fits in bins

Binning method:

1. Binning by frequency: ensure equal amount of items in each bin

    ```python
    df['bin_qcut'] = pd.qcut(df['Cupcake'], q=3, precision=1, labels=labels)
    ```

2. Binning by distance: ensure equal distance between each boundary

    ```python
    # use bin to specify the boundaries, labels are the list of names for each bin
    bin = [min, mean, max]
    data['power_bin'] = pd.cut(data['power'], bin, labels=[])
    ```

3. Best-KS: 

    [https://blog.csdn.net/hxcaifly/article/details/84593770](https://blog.csdn.net/hxcaifly/article/details/84593770)

4. Chi-square: $X^2 = \displaystyle\sum_{i=1}^2\displaystyle\sum_{j=1}^2\frac{(A_{ij}-E_{ij})^2}{E_{ij}}$, where $A_{ij}$is the number of j class in i group, $E_{ij} = \frac{N_i*C_j}{N}$, the expectation frequency of $A_{ij}$ where N is the total number of sample, $N_i$ is the number of sample in ith group and $C_j$ is the percentage of j class percentage over all samples.
    - Treat each instance as a bin
    - Calculate the Chi-square value between each adjacent interval
    - Combine the pair of bin with smallest Chi-square value
    - Continue until we reach the optimal number of bin
