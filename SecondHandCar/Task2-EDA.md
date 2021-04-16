Pretty similar to the previous group learning.

New things:
## Imputate Missing Values

- When checking data type using `df.info()`, if certain field has type object rather than int / float / string, might as well check the `df.column.value_counts()` to check the distribution, and replace abnormal characters to nan before we handle nan.
- See the proportion of missing values by setting dropna to False: `df.column1.value_counts(dropna=False, normalize=True).head()`
- To check the proportion of missing values across all columns

```python
missing_props = df.isna().sum() / len(df)
missing_props[missing_props > 0].sort_values(ascending=False)
```

- Show number of unique values for all features `df.nunique()`
- Show null values for all features `df.isnull().sum()`
- Show null value percentage `pd.DataFrame({"Missing values (%)": df.isnull().sum()/len(df.index)*100})`. And some columns might be removed since they have more than 75% of data missing.
- Show total missing values `df.isnull().sum().values.sum()`
- Check if categorical variables contains only 'Yes/No'
- Using missingno

    ```python
    import missingno as msno
    msno.matrix(train_df.sample(250))
    # OR
    msno.bar(train_df.sample(1000))
    # Visualize the nan values
    ```

## Pairplot
- More hints on how to use pairplot:

[Seaborn-05-Pairplot多变量图](https://www.jianshu.com/p/6e18d21a4cad)

## Violinplot
Check link: 

[Python可视化 | Seaborn5分钟入门(三)--boxplot和violinplot](https://zhuanlan.zhihu.com/p/34059825)

```python
# Single horizontal plot
sns.violinplot(x=train_df['feature column'])

# Vertical plot grouped by categorial variable
sns.violinplot(x='feature column', y='target column', data=train_df)

# Grouped by 2 categorical variable
sns.violinplot(x='feature1', y='target', hue='feature2', data=train_df)

# Grouped by 2, split the violin
sns.violinplot(x='feature1', y='target', hue='feature2', data=train_df, split=True)
```
