---

title: "[ML] Classification Evaluation Metrics (분류성능지표)"
excerpt: "정의와 예시"
categories:

- Machine Learning

tags:

- 딥러닝
- Deep Learning
- 머신러닝
- Machine Learning

last_modified_at: 2020-09-03-23:00:00

toc: true
toc_sticky: true

---

# Classification Evaluation Metrics (분류성능지표)

---

### 1. Accuracy (정확도)

- 가장 직관적인 분류성능 평가지표

$$Accuracy =\frac{TP+TN}{All}$$

- $All = (TP+FN+FP+TN)$

### 2. Precision (정밀도)

- `Positive`로 예측된 cases 중 **실제** `Positive` label을 가진 cases의 비율

$$Precision = \frac{TP}{TP+FP}$$

### 3. Recall (재현율)

- TPR (True Positive Rate) 또는 Sensitivity라고도 불린다.
- 실제로 Positive label을 가진 cases 중 Positive로 옳게 예측된 cases의 비율

$$Recall = \frac{TP}{TP+FN}$$

- 암이나 테러범 같이 아주 심각한 케이스를 Positive로 분류해주어야 할 때 Recall이 높아야 함
- Accuracy will not always be the metric.
- Precision and recall are often in tension. That is, improving precision typically reduces recall and vice versa.

### 4. ROC curve & AUC

x축 → $FPR = \frac{FP}{FP+TN}$

- 실제 `Negative` label인 cases 중에 모델이 `Positive`로 예측한 비율
- 게임에서 어뷰징 유저 → Positive
- 만약에 클린한 유저를 어뷰징 유저로 분류할 경우 게임 충성도 타격

y축 → $TPR=Recall = \frac{TP}{TP+FN}$

- AUC-ROC curve is one of the most commonly used metrics to evaluate the performance of machine learning algorithms.
- ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.
- The ROC curve can be used to choose the best operating point.

### 5. P-R curve & PR AUC

Precision Recall Plot (이하 PR 그래프)의 경우도 ROC 와 유사한데, 주로 데이타 라벨의 분포가 심하게 불균등 할때 사용한데, 예를 들어 이상 거래 검출 시나리오의 경우 정상 거래의 비율이 비정상 거래에 비해서 압도적으로 많기 때문에 (98%, 2%) 이런 경우에는 ROC 그래프보다 PR 그래프가 분석에 더 유리하다.

출처:

[https://bcho.tistory.com/1206#recentEntries](https://bcho.tistory.com/1206#recentEntries)

[조대협의 블로그]

What is common between ROC AUC and PR AUC is that they both look at prediction scores of classification models and not thresholded class assignments. What is different however is that **ROC AUC looks at** a true positive rate **TPR** **and** false positive rate **FPR** while **PR AUC looks at** positive predictive value **PPV** and true positive rate **TPR**.

Because of that **if you care more about the positive class, then using PR AUC**, which is more sensitive to the improvements for the positive class, is a better choice. One common scenario is a highly imbalanced dataset where the fraction of positive class, which we want to find (like in fraud detection), is small. I highly recommend taking a look at this kaggle kernel for a longer discussion on the subject of ROC AUC vs PR AUC for imbalanced datasets.

**If you care equally about the positive and negative class** or your dataset is quite balanced, then going with **ROC AUC** is a good idea.

ROC 곡선은 다양한 테스트 셋을 만날 때마다 견고한 결과를 보여줄 수 있다. 곡선 형태 유지

PR 곡선은 반면에 뚜렷한 변화가 생긴다.

### 6. F1 Score

F1 vs Accuracy

- Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial
- Accuracy can be used when the class distribution is similar while F1-score is a better metric when there are imbalanced classes as in the above case.
- In most real-life classification problems, imbalanced class distribution exists and thus F1-score is a better metric to evaluate our model on.

Both of those metrics take class predictions as input so you will have to adjust the threshold regardless of which one you choose.

Remember that **F1 score** is balancing precision and recall on the **positive class** while **accuracy** looks at correctly classified observations **both positive and negative**. That makes a **big difference** especially for the **imbalanced problems** where by default our model will be good at predicting true negatives and hence accuracy will be high. However, if you care equally about true negatives and true positives then accuracy is the metric you should choose.

In our example, both metrics are equally capable of helping us rank models and choose the best one. The class imbalance of 1-10 makes our accuracy really high by default. Because of that, even the worst model has very high accuracy and the improvements as we go to the top of the table are not as clear on accuracy as they are on F1 score.

## Reference

---

[https://medium.com/swlh/recall-precision-f1-roc-auc-and-everything-542aedf322b9](https://medium.com/swlh/recall-precision-f1-roc-auc-and-everything-542aedf322b9)

[1] An Introduction to Statistical Learning [James, Witten, Hastie, and Tibshirani]
