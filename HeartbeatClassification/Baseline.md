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
- If, Agree = 1 then we have perfect agreement, kappa score equals 1.
- If, Agree = ChanceAgree the agreement is entirely by chance and kappa score equals 0.

```python
from sklearn.metrics import confusion_matrix, cohen_kappa_score
confusion_matrix(y_true, y_pred)
cohen_kappa_score(y_true, y_pred)
```

[Multi-Class Metrics Made Simple, Part III: the Kappa Score (aka Cohen's Kappa Coefficient)](https://towardsdatascience.com/multi-class-metrics-made-simple-the-kappa-score-aka-cohens-kappa-coefficient-bdea137af09c)
