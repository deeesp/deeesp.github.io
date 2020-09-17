---
title:  "[ML] Decision Tree(결정트리)"
excerpt: "결정 트리의 장점과 단점, CART 알고리즘"
categories:
  - Machine Learning
  
tags:
  - Machine Learning
  - 머신러닝
  - Deep Learning
  - 딥러닝
  - Decision Tree
  - 결정트리
  
last_modified_at: 2020-09-17-22:00:00

toc: true
toc_sticky: true
---



## 장점
-   이해하고 해석하기 쉽다 (Interpretable)
-   사용하기 편리하고 다용도로 사용할 수 있는데 성능도 괜찮다.
-   데이터 전처리가 거의 필요하지 않다.
-   특히 특성의 스케일을 맞추거나 평균을 원점에 맞추는 작업이 필요하지 않다.
-   Prediction 시에 root node부터 leaf node까지 탐색을 하는데, 약 $O(\text{log}_2(m))$개의 노드를 거치며 하나의 특성값만 확인하므로 $O(\text{log}_2(m))$의 시간복잡도를 가져 빠른 prediction을 할 수 있다.

## 단점 (불안정성)

-   계단 모양의 decision boundary를 만들게 되는데, 이는 Training set의 회전에 민감하다.
-   이 문제를 해결하기 위해 더 좋은 방향으로 회전 시키는 PCA 기법을 사용한다.
-   `sklearn`에서 training하는 알고리즘은 stochastic(각 노드에서 평가할 후보 특성을 무작위로 선택)하기 때문에 같은 training set에서도 다른 모델을 얻게될 수 있다.
-   랜덤 포레스트는 많은 트리에서 만든 예측을 평균하여 불안정성을 극복한다.

## CART 알고리즘

-   이진트리만 만듬 (리프노드 외의 모든노드는 자식노드를 두개씩 가짐)
-   하나의 feature $k$ 의 임계값 $t_k$ 를 사용해 두개의 subset으로 나눈다.
-   나눌 때에는 가장 순수한 subset으로 나눌 수 있는 $(k, t_k)$ 짝을 찾는다.
-   Training을 중지할 때
    -   `max_depth`로 설정된 깊이가 되었을 때
    -   Impurity를 줄이는 분할을 ㅊ자을 수 없을 때
-   Greedy algorithm이기 때문에, 최적해를 보장하지는 않는다.
    -   즉, 가장 낮은 불순도로 이어질 수 있는지 없는지는 고려하지 않는다. 납득할만한 좋은 솔루션에 만족해야한다.
    -   최적의 tree를 찾는 것은 NP-Complete 문제로 알려져 있지만 $O(exp(m))$의 시간복잡도를 가지며, 매우 작은 training set에도 적용하기 힘들기 때문이다.

## 규제 매개변수

-   Training data에 대한 제약사항이 거의 없다. (전혀 없는 것은 아님)
-   훈련 되기 전에 파라미터 수가 결정되지 않는 비파라미터 모델이다.
-   적어도 최대 깊이를 조절할 수 있도록 `sklearn`에서는 `max_depth` 매개변수로 조절하여 overfitting 위험 감소시킴

### `DecisionTreeClassifier`의 매개변수
-   `min_samples_split`: 분할되기 위해 노드가 가져야 하는 최소 샘플의 개수
-   `min_samples_leaf`: 리프노드가 가지고 있어야 할 최소 샘플의 수
-   `min_weight_fraction_leaf`: 가중치가 부여된 전체 샘플에서의 비율로, `min_samples_leaf`와 같음
-   `max_features` 각 노드에서 분할에 사용할 특성의 최대 수

## Reference
- Hands-on Machine Learning with Scikit-Learn & TensorFlow
