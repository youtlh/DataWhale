## Tsfresh (Time Series Fresh)

Tsfresh is a powerful package extract characteristics from time series data. The package also includes feature importance evaluation and selection.

### Procedures

1. Feature extraction. The algorithm characterizes time series with comprehensive and well-established feature mappings and considers additional features describing meta-information. The feature calculators used to derive the features are contained in `tsfresh.feature_extraction.feature_calculators`
2. Feature significance testing. Each feature vector is individually and independently evaluated with respect to its significance for predicting the target under investigation. List of tests conducted in `tsfresh.feature_selection.significance_tests` and results to vector of p-values, quantifying the significance of each feature for predicting the label/target.
3. Multiple test procedure. Vector of p-values is evaluated on basis of the Benjamini-Yekutieli procedure to decide which features to keep.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aa2c6ceb-6d1d-49f4-bf29-8be83efc250a/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aa2c6ceb-6d1d-49f4-bf29-8be83efc250a/Untitled.png)

### List of features

[https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)
