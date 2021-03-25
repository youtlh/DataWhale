# Machine Learning Algorithm

## Division

### Common Supervised Learning Algorithm

- Logistic regression
- Linear regression
- Decision Tree
- Naive Bayes
- K-Nearest Neighbor
- Support Vector Machine
- Integrate algorithm (Adaboost, ...)

### Common Unsupervised Learning Algorithm

- Clustering
- Dimension reduce
- Correlation rule
- PageRank,...

## Naive Bayes Classification

### Advantages

- Extremely fast for both training and prediction
- Provide straightforward probabilistic prediction
- Often very easily interpretable
- Very few tunable parameters

### Usage

- When the naive assumptions actually match the data
- For very well-separated categories, when model complexity is less important
- For very high-dimensional data, when model complexity is less important
- For initial quick-and-dirty baseline

As the dimension of a dataset grows, it is much less likely for any two points to be found close together, thus high dimension data tend to be more separated.

### Theorem

Build on Bayes's theorem that describing the relationship of conditional probabilities of statistical quantities. $P(L|features) = \frac{P(features|L)P(L)}{P(features)}$. If we are to decide between two labels $L_1$ and $L_2$, compute the ratio of the posterior probabilities of both labels: $\frac{P(L_1|features)}{P(L_2|features)} = \frac{P(features|L_1)P(L_1)}{P(features|L_2)P(L_2)}$.

What we are missing is how to compute the generative model $P(features|L_i)$ for each label that specifies the hypothetical random process that generates the data.

"Naive" in "Naive Bayes" means that we are making rough assumptions about the generative model of each label.

### Gaussian Naive Bayes

- Assumption: data from each label is drawn from simple Gaussian distribution, treat each feature independently, fit model simply by finding the mean and standard deviation of the points within each label.
- Theory:

[Learning by Implementing: Gaussian Naive Bayes](https://towardsdatascience.com/learning-by-implementing-gaussian-naive-bayes-3f0e3d2c01b2)

- Gaussian conditional probability: $p(x_i|c_j) = \frac{1}{\sqrt(2\pi\sigma^2_{i,j})}\exp^{-\frac{1}{2}(\frac{x_i-\mu_{i,j}}{\sigma_{i,j}})}$ for i is the number of features and j the number of classes. For every specific class, since features are independent, model training equals calculating the conditional $p(x_1,x_2|c) = p(x_1|c) * p(x_2|c)$ as well as $p(c)$.
- When making prediction, probability for every class were calculated and $Prediction(x) = argmax_c p(c|x)$

```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
yprob = model.predict_proba(Xnew)
yprob.round(2)
# predict proba gives the posterior probabilities of the first ad second label respectively.
```

- The Boundary for Gaussian naive Bayes is quadratic.
- The final classification will only be as good as the model assumptions that lead to it, which is why Gaussian naive Bayes often does not produce very good results.

### Multinomial Naive Bayes

- Assumption: features are assumed to be generated from a simple multinomial distribution.
- Usage: Multinomial distribution describes the probability of observing counts among a number of categories, thus this Bayes is most appropriate for features that represent counts or count rates.
- Theory:

[Applying Multinomial Naive Bayes to NLP Problems: A Practical Explanation](https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf)

## Logistic Regression

Fitting data into a logistic function to predict event happen probability. Handles binary classification problems. Sigmoid function $g(z)=\frac{1}{1+e^{-z}}$ will always return a number between 0 and 1. When z = 0, g(z)=0.5. With decreasing z, the function approaches 0, vise versa.

### Advantage:

- Fast training speed, computation size only related to the number of features
- Easy to understand and interpreted, weighting of the features indicates the influence they have toward the final result
- Suitable for binary classification problem
- Small occupation of RAM

### Disadvantage

- Need to preprocess the missing value and anomalies
- Cannot handle non-linear problem
- Cannot handle data with uneven distribution
- Due to simple algorithm, the accuracy is not very high

### Decision Boundary

Since g(z)=0.5 is the boundary for splitting the two decision outcomes, the decision boundary is to find $\theta^TX = 0$.

$h_\theta(x) = g(\theta^TX) = g(\theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_1^2 + \theta_4x_2^2+...)$

To obtain complicated nonlinear decision boundary shape, we just have to complicate the $\theta^TX$ by including high order terms.

### Cost Function

The function to evaluate the offset between the prediction and actual result.

$J(\theta) = \frac{1}{m}\displaystyle\sum_{i=1}^{m}\frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2$ is the cost function for linear regression, generated based on the linear equation. The cost function for logistic regression is: $Cost(h_\theta(x), y)=\begin{cases}
   -log(h_\theta(x)) &\text{if } y =1 \\
 -log(1-h_\theta(x)) &\text{if } y=0
\end{cases}$.

- For true value class 1, prediction y=1, $Cost=-log(1)=0$
- For true value class 1, prediction y=0, $Cost=-log(0)=\infin$
- For true value class 0, prediction y=1, $Cost=-log(1-1)=\infin$
- For true value class 0, prediction y=0, $Cost=-log(1-0)=0$

### Gradient Descent

## Decision Tree

Decision tree is a basic regression and classification method.

### Advantage

- Easy to understand and interpret, can be visualized
- Almost do not need data preprocessing, unlike the other methods that requires regularization, create virtual variable and delete missing value
- The cost of training the tree is the logarithm of total data points number
- Handle numerical variable and classification variable
- Data can be continuous and discrete
- White box mode. Easy to explain.

### Disadvantage

- Might create a tree too complicated (overfitting)
- Might be unstable, even a small change might result to a completely different tree
- Learn the best decision tree is a never completely perfect question. Thus, the traditional decision tree algorithms are based on the greedy algorithm, take the best decision at each step in order to obtain the global maximum
- If a certain class is original inclined, the decision tree tends to create an inclined tree. Thus, samples should be resampled to average the distribution.

### Feature Selection

Feature selection is done based on information gain (decrement in terms of entropy). 

- Entropy is the measure of uncertainty of random variable. If a potential affairs could be grouped into a multiple class, the information carried by x is $I(x_i)=-log_2p(x_i)$, where $p(x_i)$ is the probability for been in the i class. Thus, the information expectation for all classes are $H=-\displaystyle\sum_{i=1}^{n}p(x_i)log_2p(x_i)$, where n is the total number of classes. The higher the entropy, the more uncertain the random variable is. For a specific dataset D, |D| is the sample size, k is the number of class, $C_k$ is the number of element in class K, then the entropy of D is $H(D)=-\displaystyle\sum_{k=1}^{K}\frac{|C_k|}{|D|}log_2\frac{|C_k|}{|D|}$
- Conditional Entropy: H(Y|X) is the uncertainty of random variable Y given random variable X. $H(Y|X) = \displaystyle\sum_{i=1}^np_iH(Y|X=x_i)$, where $p_i=P(X=x_i), i=1,2,...,n$.
- Information gain: in terms of features, the information gain g(D, A) of feature A to the training set D is the difference between the entropy of D and the conditional entropy of D given A. $g(D,A)=H(D)-H(D|A)$. Suppose we have n features to test ${a_1,a_2,...,a_n}$, training set D can be will be divided into n subsets ${D_1,D_2,....,D_n}$ by A. Thus, $g(D,A)=H(D)-H(D|A)=H(D)-\displaystyle\sum_{i=1}^{n}\frac{|D_i|}{|D|}H(D_i) = H(D)+\displaystyle\sum_{i=1}^{n}\frac{|D_i|}{|D|}\displaystyle\sum_{k=1}^{K}\frac{|D_{ik}|}{|D_i|}log_2\frac{|D_{ik}|}{|D_i|}$. We then compare the information gain for different feature and select the best.

### Tree Construction

- At each node, compare the information gain for different features and select the best to construct the next level
- To decide when to step iterate the construction process
    - When items in each class are individual
    - When all the features are used up. The tree cannot provide a single answer for the classification, thus it will return the most frequent class as result.

### Code

- Sklearn.tree.DecisionTreeClassifier
    - criterion: gini (Default) or entropy. Gini is more statistics while entropy is more information related. ID3 algorithm uses entropy while CART uses gini
    - splitter:  best or random, the mechanism to select node. Best selects based on the criterion, while random select the best one in a random subset. Use random when data size is extremely large
    - max_features: default None. Use None if total feature number smaller than 50.
        - Auto: max_features = sqrt(n_features)
        - Sqrt = Auto
        - Log2: max_features = log2(n_features)
        - None: max_features = n_features
    - max_depth: default None. If model size large and has planty of feature, the common setting is between 10-100.
    - min_samples_split: default 2. Limit the minimum sample size for internal node to be further split. Increase this number if the sample size is huge.
    - min_weight_fraction_leaf: default 0. Leaf with weight less than this number will be cut. Useful when the sample has lots of missing values or the sample distribution is not normal.
    - max_leaf_nodes: default None. Limit the maximum number of leaf number to prevent overfitting. No need to worry about this value if total number of features is not huge.
    - class_weight: None or balanced. Evaluate the distribution of different classes inside the sample to prevent certain class has too many sample. Balanced will automatically calculate and average the weighting. No need to pay attention to this if the sample distribution is relatively average
- Graphviz to visualize decision tree
    - Install package pydotplus (pip) and Graphviz ([http://www.graphviz.org/Home.php](http://www.graphviz.org/Home.php))
    - Add Graphviz to environment variable (system variable, Path)
    - Reference:

    [https://blog.csdn.net/c406495762/article/details/76262487](https://blog.csdn.net/c406495762/article/details/76262487)

### Tips

- Decision tree can easily get overfit if the sample size is limited with abundant feature
- For cases with limited sample and abundant sample feature, use PCA, Losso or ICA to restrict the dimension.
- Visualize the decision tree if possible and limit the tree depth. Observe the fitting inside decision tree before modify the tree depth
- Check the sample class distribution. For uneven distribution, limit the model to incline toward the majority using class_weight
- If the original matrix is sparse, recommend to use csc_matrix to sparse the matrix before fitting.

## Ensemble Learning

Ensemble learning is the learning to make prediction by creating multiple weak classification, and use one algorithm to ensemble their result and generate the final result. There are two genres inside ensemble learning: boosting and bagging

1. Boosting is to split the problem to multiple serial subproblems. Problems have to be solved one after another. Common algorithm: Adaboost, GBDT, Xgboost...
2. Bagging is to split a problem to multiple unrelated parallel subproblems and assign them to different person. Common algorithm: Random Forest.

Difference between the two division:

- Sample selection:
    - Bagging is the replaceable extraction from the original set. Each training set selection is independent with one another.
    - Boosting has the same training set for each round, but the weighting of each sample inside the classifier is changing according to the previous round prediction result.
- Sample weighting:
    - Bagging is uniform sampling and the weighting for each sample is identical
    - Boosting is correcting the sample weighting based on the error rate. The larger the error rate, the larger the weighting
- Prediction function:
    - Bagging has equal weighting for all the prediction function
    - Boosting gives equal weighting to all sub-classifier originally, but classifier with smaller class error will get higher weighting
- Parallel computation:
    - Bagging can generate prediction functions in parallel
    - Boosting can only generate prediction function in sequence, as parameters of the latter ones depend on the previous result

## GBDT

GBDT is the foundation of other algorithms like XGBoost and LightGBM. It is an integrate model, can be considered as the linear addition of many other base models.

### CART regression tree

CART tree is a decision tree model. Compared with the tradition ID3 model, CART tree is a binary tree.

- To improve the model prediction accuracy, each model has to be simplified. For decision tree, leafs has to be cut properly
- Loss function for cut leafs is $C_a(T)=C(T)+\alpha|T|$,  where C(T) is the error of the training data and |T| is the number of node in the tree. Alpha is the tuning parameter, deciding whether we are targeting the accuracy or simplicity (underfitting or overfitting)
- Algorithm to cut tree: starting from the entire tree T0, for any internal node t, compare the loss for losing the single node t and that of the tree initiated from t. If single node is better than the tree constructed, cut the following tree.

### GBDT model

GBDT model is an integrated model, linear addition of numerous CART tree. For every tree, the growth of the tree is based on the mean square error of the prediction of all its subnode. One then select the subnode with the minimum square error $min(\frac{(x_i-y)^2}{N})$

Let $f_t(x)$ as the model at t round, $h_t(x)$ as the $t^{th}$ decision tree, the model can be defined as $f_t(x)=\displaystyle\sum_{t=1}^Th_t(x) = f_{t-1}(x)+h_t(x)$. The model at step t is based on the model at t-1 step, learn a decision tree each step to correct the error.

The lost function is defined as $L(f_t(x),y)=L(f_{t-1}+h_t(x),y)=\displaystyle\sum_{x_i\in R_m}(y_i-f(x_i))^2$, the square difference the decision tree has with the training data. Gradient of the function is just simply $y_i-f_{m-1}(x_i)$.

To decide what kind of decision tree should be added at the t step, we can fit a CART tree according to the approximate loss for this step based on the negative gradient of the loss function (potentially we still want to minimize gradient, thus always using the negative gradient). The negative gradient for the loss function at step t with i sample $r_{t, i} = -[\frac{\delta L(y,f(x_i))}{\delta f(x_i)}]=y_i-f_{m-1}(x_i)$.

For step t, the sample region is $R_{tj}$, where j is the number of leap node. Thus, the decision tree function is $h_t(x)=\displaystyle\sum_{j=1}^{J}c_{t,j}I(x\in R_{t,j})$. This iteration continues until the loss function converges.

[https://cloud.tencent.com/developer/article/1496824](https://cloud.tencent.com/developer/article/1496824)

## XGBoost

Extreme Gradient Boosting. 

### Theorem

For each sample, it will be predicted by n weak classifiers, and the final prediction is the addition of the answers from the weak  classifiers. $\hat{y_i}=\displaystyle\sum_{t=1}^kf_t(x_i)$, where $f_k$ is the kth base model, and $\hat{y_i}$ is the prediction value of the ith sample. The loss function between the prediction value $\hat{y_i}$ and the real value $y_i$ is $L=\displaystyle\sum_{i=1}^nl(y_i,\hat{y_i})$ where n is the number of samples.

### Steps

- Initialize the prediction value for every sample
- Define the objective function $Obj=\displaystyle\sum_{i=1}^nl(y_i,\hat{y}_i )+\displaystyle\sum_{t=1}^k\varOmega(f_t)$, where $\varOmega(f_t)$ is the normalization term $\varOmega(f_t)=\gamma T_t+\frac{1}{2}\lambda\displaystyle\sum_{j=1}^{T}\omega_j^2$, used to suppress the model complexity. $T_t$ is the number of leaf node, $\omega_j$ is the weight on the j leaf, and $\gamma$ and $\lambda$ is the preset hyperparameters.
    - How to simplify the objection function? By the definition of boosting, at the t step of the model, the model prediction for the ith sample $x_i$ is $\hat{y}_i^t=\hat{y}_i^{t-1}+f_t(x_i)$, where $f_t(x_i)$ is the weak model.
    - $Obj=\displaystyle\sum_{i=1}^nl(y_i,\hat{y}_i^t)+\displaystyle\sum_{t=1}^k\varOmega(f_t)=\displaystyle\sum_{i=1}^nl(y_i,\hat{y}_i^{t-1}+f_t(x_i))+\displaystyle\sum_{t=1}^k\varOmega(f_t)$
- Taylor simplify the objection function
    - Taylor formula: $f(x+\Delta x)\approx f(x)+f'(x)\Delta x+\frac{1}{2}f''(x)\Delta x^2$
    - Take $\hat{y}_i^{t-1}$ as x and the residual $f_t(x_i)$ as $\Delta x$, then $Obj^{(t)}\approx \displaystyle\sum_{i=1}^{n}[l(y_i,\hat{y}_i^{t-1})+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\displaystyle\sum_{i=1}^t \varOmega(f_i)$, where $g_i$ is the first derivative of lost function l, and $h_i$ is the second derivative.
- Further objection function simplification
- Create decision tree based on best split point
- Predict sample based on the new tree and add the result to the previous value
- Return to second step and repeat the iteration


# Model Validation

## Division of Splitting Training and Test Set

Data set division should satisfy the following two criteria:

- Training set and test set should have the identical distribution to the sample space. They have to be sampled independently.
- Training set and test set should be mutual with one another.

Division methods

1. Set aside method: Divide data set D into two mutual set, one as training, another as testing. To divide the set while maintaining the consistency as much as possible, use stratified sampling to sample the raw data. Usually take 2/3 to 4/5 percent of data as training set, others as test set.
2. Cross-validation method: the k fold method usually divide the dataset D into k set, where k-1 as the training set and the remaining 1 as test. This enables us to get k groups of training/test for k times of training and testing. The final return value is the average of the k testing result. Dataset division still done by stratified sampling. To sustain the stability of the result, k is usually larger than 10. k=1 situation is called 1 remaining method.
3. Self help method: select sample from dataset one at a time and put it back afterward. Repeat the operation m times, so we get a training set with size m. The sample that does not appear in the m data set is the testing set.
4. How to decide when to use which division method
    - When we have plenty of data, use set aside or k fold method
    - When the dataset is small and cannot be effectively separate to training/testing
    - When the dataset is small but can be effectively separate, best use the 1 remaining method.

## Train-test-split

Train_test_split + accuracy_score

```python
from sklearn.matrics import accuracy_score
from sklearn.cross_validation import train_test_split

X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)
model.fit(X1, y1)
y2_model = model.predict(X2)
accuracy_score(y2, y2_model)
```

## Cross-validate-score

```python
from sklearn.cross_validation import cross_val_score

cross_val_score(model, X, y, cv=5)
```

## Bias-variance trade-off

Underfitting vs. Overfitting

- $R^2$ coefficient of determination, for high bias model (underfitting) $R^2$ is small; for high variance model (overfitting) $R^2$ is very high on training set but very low on test set.

## Validation Curve

x-axis is the degree of freedom (complexity), y axis is the training score and the validation score. Training score will continue increase until converge, while validation score will reach top and gradually decrease due to overfitting.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1a38eb26-42a0-4f2e-89e9-a702ea7608f3/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1a38eb26-42a0-4f2e-89e9-a702ea7608f3/Untitled.png)

```python
from sklearn.learning_curve import validation_curve
degree = np.arange(21)
train_score, val_score = validation_curve(PolynomialRegression(),
	X, y, 'polynomialfeatures__degree', degree, cv=7)
```

## Learning Curve

plot of the training/validation score with respect to the size of the training set.

- Larger dataset can support more complicated model.
- With increasing training set size, training score will decrease and converge while validation score will increase and eventually converge.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/546e9ac7-a4c9-4659-9200-1d44ef0f15f9/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/546e9ac7-a4c9-4659-9200-1d44ef0f15f9/Untitled.png)

- When learning curve is already converge, adding more training date will not significantly increase the fit. The only way to increase the converged score is to use a different (usually more complicated) model.

```python
from sklearn.learning_curve import learning_curve

N, train, val = learning_curve(PolynomialRegression(degree), X, y, cv=7,
train_sizes=np.linspace(0.3, 1, 25))
```

## Classification accuracy

### Confuse Matrix

1. True Positive (TP): If one class is originally positive and predicted also as positive
2. False Negative (FN): If one class is originally positive but predicted as negative
3. False Positive (FP): If one class is originally negative but predicted as positive
4. True Negative (TN): If one class is originally negative and predicted also as negative

### Accuracy

Not suitable for unbalanced sample. $Accuracy = \frac{Correct}{Total} = \frac{TP+TN}{TP+TN+FP+FN}$

### Precision

For prediction results, precision is the percentage of real positive samples in all samples that are predicted as positive. $Precision = \frac{TP}{TP+FP}$

### Recall

For original samples, recall is the percentage of all positive samples to be predicted as positive samples. $Recall = \frac{TP}{TP+FN}$

Example: suppose out of 10 essays, 4 are the wanted ones, and based on the algorithm, you found 5, which includes 3 of the 4 wanted ones. The precision of the algorithm is 3/5, while the recall of the algorithm is 3/4.

[Multi-Class Metrics Made Simple, Part I: Precision and Recall](https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2)

### ROC / AUC

### Macro-P

Find the precision of each class and calculate the average $macroP = \frac{1}{n}\displaystyle\sum_1^nP_i$

### Macro-R

Find the recall of each class and calculate the average $macroR = \frac{1}{n}\displaystyle\sum_1^nR_i$

### Macro-F1

In general, we prefer classifiers with higher precision and recall score. However, there is a trade-off between precision and recall. Macro F1 is calculated by compute the precision and recall for each class, and thus, F1 for each class is given by $macroF1 = \frac{2macroPmacroR}{macroP+macroR}$. Favoring minority class.

### Weighted-F1

Instead of averaging the macro-F1 with equal weights for each class, weight the F1 score of each class by the number of samples from that class. Favoring majority class.

### Micro-P

Instead of treating different classes differently, micro methods take the entire dataset into consideration. Thus, $microP = \frac{\overline{TP}}{\overline{TP}*\overline{FP}}$. No favoring.

### Micro-R

For multi classification problem, every FP is equivalently FN for another class. Thus, microP equals microR. $microR = \frac{\overline{TP}}{\overline{TP}*\overline{FN}}$

### Micro-F1

This is also the classifier's overall accuracy: the proportion of correctly classified samples out of all the samples (TP is also the TN for another class)

$microF1 = \frac{2*microP*microR}{microP+microR} = microP = microR$

Note that high F1-score does not necessarily mean good model, as they give equal weight to precision and recall. For situations that relative importance difference between the two are significant, ie, healthy people as sick or sick people as health.

[Multi-Class Metrics Made Simple, Part II: the F1-score](https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1)

### Kappa Score

Used to measure the degree of agreement (inter-rater reliability) between two ratings. High level of agreements stands for higher confidence.

$KappaScore = (Agree-ChanceAgree)/(1-ChanceAgree)$

- Agree calculated by total agreement / total number of samples
- ChanceAgree = ChanceAgree(A agree and B agree on 1) + ChanceAgree(A agree and B agree on 2) + ...
- If,$Agree = 1$ then we have perfect agreement, kappa score equals 1.
- If,$Agree = ChanceAgree$ the agreement is entirely by chance and kappa score equals 0.

```python
from sklearn.metrics import confusion_matrix, cohen_kappa_score
confusion_matrix(y_true, y_pred)
cohen_kappa_score(y_true, y_pred)
```

[Multi-Class Metrics Made Simple, Part III: the Kappa Score (aka Cohen's Kappa Coefficient)](https://towardsdatascience.com/multi-class-metrics-made-simple-the-kappa-score-aka-cohens-kappa-coefficient-bdea137af09c)

# Hyperparameter Selection

## Greedy Search

First select the parameter that has the highest model influence to the maximum, then continue with the second important parameter and so on, until all parameters are set.

### Disadvantage

Solution might not be global minimum.

### Tips

Pay special attention to the parameter adjust sequence for tree models (the importance of the parameters)

1. max_depth, num_leaves
2. min_data_in_leaf, min_child_weight
3. bagging_fraction, feature_fraction, bagging_freq
4. reg_lambda, reg_alpha
5. min_split_gain

### Code

```python
# Provide the list of each parameter needed to tune and calibrate one by one
from sklearn.model_selection import cross_val_score
# objective
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    """预测并计算roc的相关指标"""
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    best_obj[obj] = score

# num_leaves
best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0], num_leaves=leaves)
    """预测并计算roc的相关指标"""
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    best_leaves[leaves] = score

# max_depth
best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x:x[1])[0],
                          max_depth=depth)
    """预测并计算roc的相关指标"""
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    best_depth[depth] = score
```

## Grid Search

Provided by sklearn, will return the best result and parameter with potential model parameter. Provide better result compared to greedy search, but only suitable to small data set.

Take lightgbm as an example, we will set a bigger learning rate, and optimize the parameters one by one. Finally to obtain the best parameter, we will decrease the learning rate to 0.05 and tune the final result.

### Code

```python
"""通过网格搜索确定最优参数"""
from sklearn.model_selection import GridSearchCV

def get_best_cv_params(learning_rate=0.1, n_estimators=581, num_leaves=31, max_depth=-1, bagging_fraction=1.0, 
                       feature_fraction=1.0, bagging_freq=0, min_data_in_leaf=20, min_child_weight=0.001, 
                       min_split_gain=0, reg_lambda=0, reg_alpha=0, param_grid=None):
    # 设置5折交叉验证
    cv_fold = KFold(n_splits=5, shuffle=True, random_state=2021)

    model_lgb = lgb.LGBMClassifier(learning_rate=learning_rate,
                                   n_estimators=n_estimators,
                                   num_leaves=num_leaves,
                                   max_depth=max_depth,
                                   bagging_fraction=bagging_fraction,
                                   feature_fraction=feature_fraction,
                                   bagging_freq=bagging_freq,
                                   min_data_in_leaf=min_data_in_leaf,
                                   min_child_weight=min_child_weight,
                                   min_split_gain=min_split_gain,
                                   reg_lambda=reg_lambda,
                                   reg_alpha=reg_alpha,
                                   n_jobs= 8
                                  )

    f1 = make_scorer(f1_score, average='micro')
    grid_search = GridSearchCV(estimator=model_lgb, 
                               cv=cv_fold,
                               param_grid=param_grid,
                               scoring=f1

                              )
    grid_search.fit(X_train, y_train)

    print('模型当前最优参数为:{}'.format(grid_search.best_params_))
    print('模型当前最优得分为:{}'.format(grid_search.best_score_))
```

```python
"""
需要注意一下的是，除了获取上面的获取num_boost_round时候用的是原生的lightgbm（因为要用自带的cv）
下面配合GridSearchCV时必须使用sklearn接口的lightgbm。
"""
"""设置n_estimators 为581，调整num_leaves和max_depth，这里选择先粗调再细调"""
lgb_params = {'num_leaves': range(10, 80, 5), 'max_depth': range(3,10,2)}
get_best_cv_params(learning_rate=0.1, n_estimators=581, num_leaves=None, max_depth=None, min_data_in_leaf=20, 
                   min_child_weight=0.001,bagging_fraction=1.0, feature_fraction=1.0, bagging_freq=0, 
                   min_split_gain=0, reg_lambda=0, reg_alpha=0, param_grid=lgb_params)

"""num_leaves为30，max_depth为7，进一步细调num_leaves和max_depth"""
lgb_params = {'num_leaves': range(25, 35, 1), 'max_depth': range(5,9,1)}
get_best_cv_params(learning_rate=0.1, n_estimators=85, num_leaves=None, max_depth=None, min_data_in_leaf=20, 
                   min_child_weight=0.001,bagging_fraction=1.0, feature_fraction=1.0, bagging_freq=0, 
                   min_split_gain=0, reg_lambda=0, reg_alpha=0, param_grid=lgb_params)

"""
确定min_data_in_leaf为45，min_child_weight为0.001 ，下面进行bagging_fraction、feature_fraction和bagging_freq的调参
"""
lgb_params = {'bagging_fraction': [i/10 for i in range(5,10,1)], 
              'feature_fraction': [i/10 for i in range(5,10,1)],
              'bagging_freq': range(0,81,10)
             }
get_best_cv_params(learning_rate=0.1, n_estimators=85, num_leaves=29, max_depth=7, min_data_in_leaf=45, 
                   min_child_weight=0.001,bagging_fraction=None, feature_fraction=None, bagging_freq=None, 
                   min_split_gain=0, reg_lambda=0, reg_alpha=0, param_grid=lgb_params)

"""
确定bagging_fraction为0.4、feature_fraction为0.6、bagging_freq为 ，下面进行reg_lambda、reg_alpha的调参
"""
lgb_params = {'reg_lambda': [0,0.001,0.01,0.03,0.08,0.3,0.5], 'reg_alpha': [0,0.001,0.01,0.03,0.08,0.3,0.5]}
get_best_cv_params(learning_rate=0.1, n_estimators=85, num_leaves=29, max_depth=7, min_data_in_leaf=45, 
                   min_child_weight=0.001,bagging_fraction=0.9, feature_fraction=0.9, bagging_freq=40, 
                   min_split_gain=0, reg_lambda=None, reg_alpha=None, param_grid=lgb_params)

"""
确定reg_lambda、reg_alpha都为0，下面进行min_split_gain的调参
"""
lgb_params = {'min_split_gain': [i/10 for i in range(0,11,1)]}
get_best_cv_params(learning_rate=0.1, n_estimators=85, num_leaves=29, max_depth=7, min_data_in_leaf=45, 
                   min_child_weight=0.001,bagging_fraction=0.9, feature_fraction=0.9, bagging_freq=40, 
                   min_split_gain=None, reg_lambda=0, reg_alpha=0, param_grid=lgb_params)
```

```python
"""
参数确定好了以后，我们设置一个比较小的learning_rate 0.005，来确定最终的num_boost_round
"""
# 设置5折交叉验证
# cv_fold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True, )
final_params = {
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'num_leaves': 29,
                'max_depth': 7,
                'objective': 'multiclass',
                'num_class': 4,
                'min_data_in_leaf':45,
                'min_child_weight':0.001,
                'bagging_fraction': 0.9,
                'feature_fraction': 0.9,
                'bagging_freq': 40,
                'min_split_gain': 0,
                'reg_lambda':0,
                'reg_alpha':0,
                'nthread': 6
               }

cv_result = lgb.cv(train_set=lgb_train,
                   early_stopping_rounds=20,
                   num_boost_round=5000,
                   nfold=5,
                   stratified=True,
                   shuffle=True,
                   params=final_params,
                   feval=f1_score_vali,
                   seed=0,
                  )
```

## Bayes Optimization

Define the objective function to be optimized, update the posterior of the target function by adding sample point. Tuning the current parameter according to previous information.

### Steps

- Define the optimization function
- Build the model

```python
from sklearn.model_selection import cross_val_score

"""定义优化函数"""
def rf_cv_lgb(num_leaves, max_depth, bagging_fraction, feature_fraction, bagging_freq, min_data_in_leaf, 
              min_child_weight, min_split_gain, reg_lambda, reg_alpha):
    # 建立模型
    model_lgb = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_class=4,
                                   learning_rate=0.1, n_estimators=5000,
                                   num_leaves=int(num_leaves), max_depth=int(max_depth), 
                                   bagging_fraction=round(bagging_fraction, 2), feature_fraction=round(feature_fraction, 2),
                                   bagging_freq=int(bagging_freq), min_data_in_leaf=int(min_data_in_leaf),
                                   min_child_weight=min_child_weight, min_split_gain=min_split_gain,
                                   reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                                   n_jobs= 8
                                  )
    f1 = make_scorer(f1_score, average='micro')
    val = cross_val_score(model_lgb, X_train_split, y_train_split, cv=5, scoring=f1).mean()

    return val
```

- Define the parameter to be tuning

```python
from bayes_opt import BayesianOptimization
"""定义优化参数"""
bayes_lgb = BayesianOptimization(
    rf_cv_lgb, 
    {
        'num_leaves':(10, 200),
        'max_depth':(3, 20),
        'bagging_fraction':(0.5, 1.0),
        'feature_fraction':(0.5, 1.0),
        'bagging_freq':(0, 100),
        'min_data_in_leaf':(10,100),
        'min_child_weight':(0, 10),
        'min_split_gain':(0.0, 1.0),
        'reg_alpha':(0.0, 10),
        'reg_lambda':(0.0, 10),
    }
)

"""开始优化"""
bayes_lgb.maximize(n_iter=10)
"""显示优化结果"""
bayes_lgb.max
```

- Build a new model according to the parameter constructed, lower the learning rate to find the best iteration number

```python
"""调整一个较小的学习率，并通过cv函数确定当前最优的迭代次数"""
base_params_lgb = {
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                    'num_class': 4,
                    'learning_rate': 0.01,
                    'num_leaves': 138,
                    'max_depth': 11,
                    'min_data_in_leaf': 43,
                    'min_child_weight':6.5,
                    'bagging_fraction': 0.64,
                    'feature_fraction': 0.93,
                    'bagging_freq': 49,
                    'reg_lambda': 7,
                    'reg_alpha': 0.21,
                    'min_split_gain': 0.288,
                    'nthread': 10,
                    'verbose': -1,
}

cv_result_lgb = lgb.cv(
    train_set=train_matrix,
    early_stopping_rounds=1000, 
    num_boost_round=20000,
    nfold=5,
    stratified=True,
    shuffle=True,
    params=base_params_lgb,
    feval=f1_score_vali,
    seed=0
)
print('迭代次数{}'.format(len(cv_result_lgb['f1_score-mean'])))
print('最终模型的f1为{}'.format(max(cv_result_lgb['f1_score-mean'])))
```

- Check the result on validation set

```python
import lightgbm as lgb
"""使用lightgbm 5折交叉验证进行建模预测"""
cv_scores = []
for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
    print('************************************ {} ************************************'.format(str(i+1)))
    X_train_split, y_train_split, X_val, y_val = X_train.iloc[train_index], y_train[train_index], X_train.iloc[valid_index], y_train[valid_index]

    train_matrix = lgb.Dataset(X_train_split, label=y_train_split)
    valid_matrix = lgb.Dataset(X_val, label=y_val)

    params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 4,
                'learning_rate': 0.01,
                'num_leaves': 138,
                'max_depth': 11,
                'min_data_in_leaf': 43,
                'min_child_weight':6.5,
                'bagging_fraction': 0.64,
                'feature_fraction': 0.93,
                'bagging_freq': 49,
                'reg_lambda': 7,
                'reg_alpha': 0.21,
                'min_split_gain': 0.288,
                'nthread': 10,
                'verbose': -1,
    }

    model = lgb.train(params, train_set=train_matrix, num_boost_round=4833, valid_sets=valid_matrix, 
                      verbose_eval=1000, early_stopping_rounds=200, feval=f1_score_vali)
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_pred = np.argmax(val_pred, axis=1)
    cv_scores.append(f1_score(y_true=y_val, y_pred=val_pred, average='macro'))
    print(cv_scores)

print("lgb_scotrainre_list:{}".format(cv_scores))
print("lgb_score_mean:{}".format(np.mean(cv_scores)))
print("lgb_score_std:{}".format(np.std(cv_scores)))
```

## Tips

- The cv function inside the integrate model can be used to tune single parameter. Usually used to determine the iteration number.
- Grid search will be exceptionally slow with large dataset
