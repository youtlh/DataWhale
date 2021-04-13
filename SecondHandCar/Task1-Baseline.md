# Reduce Data Dimension

## FastICA

Fast algorithm for independent component analysis. It is generally not used for reducing dimensionality, but to separate superimposed signals.

## Factor Analysis

Discover latent-factors. Extends PCA with different noise in each dimension. Advantage over PCA: It can model the variance in every direction of the input space independently.

## SparsePCA

## Principal Component Analysis (PCA)

### Variance

If the variance is very low, we are basically getting all similar kinds of data from that feature or feature set. The machine learning usually works on dissimilarities in behavior of Data Points among different classes. So, if it is very low points are expected to cluster around the same points which cause them to be very hard to differentiate. So variance is often regarded as the information in case of ML datasets.

### Scaling

The data we use have several features. Now, these features have different ranges depending upon the features like some features have decimal values between 0 and 1 while others have values between 100–1000.

- **Standard Scaling**: $Scale= (x-mean(x))/std(x))$. Give values between -1 and 1.
- **Min-Max Scaling**: $Scale=(x- min(x))/(max(x)-min(x))$. Give values between 0 and 1.

### PCA

Principle idea: create n composite features which can best represent the information contained in all features of our dataset. These n features are called principal components. The value of n depends on the user. These n features are not original features, but are developed as a combination of different features. (Based on unsupervised algorithms)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scale=scaler.fit(X)
X_scaled=scaler.transform(X)
X_scaled_df=pd.DataFrame(X_scaled,columns=X.columns)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
PC = pca.fit_transform(X_scaled_df)
p_Df = pd.DataFrame(data = PC
             , columns = ['principal component 1', 'principal component 2'])
p_Df.head()
```

- Can use variance to check how much information are lost due to the PCA.

```python
pca.explained_variance_ratio_
```
