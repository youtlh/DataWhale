# Regression Analysis

Regression analysis is a statistical method to investigate whether two or more variables are correlated with each other. Make predictions on new variables based on known variables.

## Five Basic Assumptions

### Linearity & Addable

Suppose $Y=b+a_1X_1+a_2X_2+\epsilon$

- Linearity: for every change in $X_1$, Y changes $a_1$, unrelated to the value of $X_1$
    - Check: simply plot the distribution
    - Overcome: introduce transformation to the independent variabels like log, sqrt, square...
- Addable: influence of Y from $X_1$ is independent to that of other variable $X_2$

### Independent Residual

- Theorem: Residual terms should be independent from one another. Otherwise, this model has 'Autocorrelation', and therefore the model cannot describe the relationships between the variables well enough. And thus, cause generalization error. This is common on time series data as later terms are likely to be influenced by the former ones. Autocorrelation will decrease the std of the our result and thus shorten the confidence interval.
- Check Durbin-Watson Statistic $DW=\frac{\displaystyle\sum_{t=2}^T(e_t-e_{t-1})^2}{\displaystyle\sum_{t=1}^Te_t^2}$. DW should fell between 0 and 4. DW=2 stands for no autocorrelation. 0<DW<2 hints to positive correlation, and 2<DW<4 is negative correlation. Usually, one should be alert for DW<1 or DW>3

### Independent Variables

- Theorem: Independent variables $X_1 and X_2$ must be independent from one another. Otherwise, we call the variables has 'Multicollinearity'. The linkage relationship between the variables will amplify the std of the result and widen the confidence interval.
- Check: VIF - Variance Inflation Factor. Suppose $Y=\beta_0+\beta_1X_1+\beta_2X_2+...++\beta_kX_k+\epsilon$, then for variable $X_j$, the variance for parameter $\beta_j$ is $\hat{var}(\hat{\beta_j})=\frac{s^2}{(n-1)\hat{var}}(X_j)*\frac{1}{1-R_j^2}$. VIF < 3 means no multicollinearrity and VIF > 10 is rather serious

### Constant Residual

- Theorem: Residual should be constant. If one model has constant residual, we call it Homoskedasticcity 同方差性, otherwise it is Heteroskedasticity 异方差性. Heteroskedasticity means that the variance of the residual is not stable. This is common when we have outlier.
- Check: Residual filter graph

### Normal Residual

- Theorem: Residual term should match normal distribution. Skewed distribution can results to unstable confidence interval.
- Check: Q-Q plot or Kolmogorov-Smirnov test or Shapiro-Wilk test
