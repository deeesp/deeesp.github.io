---
title:  "[통계] Sample Variance를 구할 때 n-1을 나누어주는 이유"
excerpt: "베셀의 보정과 그 이유에 대해 알아보자"
categories:
  - Statistics
  
tags:
  - 통계
  - 신호처리
  - Statistics
  - Signal Processing
  - Machine Learning
  - 머신러닝
  - Stochastic Gradient Descent
  - Least Mean Square
  
last_modified_at: 2020-09-20-23:00:00

toc: true
toc_sticky: true
---

일단, 수식으로 표본분산이 어떻게 생겨먹었는지 보자.

### Sample Variance

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2$$

Sample과 Sample mean 편차 제곱의 합에 n을 나눠주는 것이 아니라 n-1로 나누어준다. 우선, n으로 나누어 주었을 경우 $E(s^2) = \sigma^2$ 식을 만족하지 않는 Biased Estimator를 낳게 된다.

Unbiased Estimator가 "Efficient" Estimator라고 보장할 수는 없지만, Biased Estimator는 "Bad" Estimator이다. 따라서 Sample Variance를 구할 때 임의로 보정해주어 Unbiased Estimator를 구해준 것이다. 많은 글에서는 단순히 "n-1로 나눠주니깐 Unbiased Estimator가 되었어!" 에 대한 수학적 유도로 설명한 글이 끝이었다. 하지만.. 왜? 라는 의문을 떨칠 수 없다. 우선 Bessel's correction<sup>베셀의 보정<sup></sup>으로 부터 이것이 유래되었다. 아래 글에서 그나마 시원하게 해석을 해주었다. 

[가장 시원한 해석](https://blog.naver.com/sw4r/221021838997)
[https://tamref.com/22](https://tamref.com/22)
### 이유 1. 수학적인 유도가 이렇게 된다.

### 이유 2. The number of Samples  
- sample variance는 population variance를 underestimate 하여 같은상태가 되기 때문에, 이를 보정해주기 위해 sample variance의 분모를 작게 만들어 전체 표본분산을 크게 만든다.
- 현실에서 통계량을 계산할 때에는, 시간과 비용의 제약으로 표본을 추출해서 계산한다. 하지만 적은 표본으로는 정확한 값을 추정하기는 힘들고, 많은 표본을 추출하는 것은 리소스가 많이 든다. 따라서 n-1로 보정을 해주어 정확도를 높인다.

### 이유 3. Sample Variance 의 자유도가 n-1이다.
카이제곱 자유도

### 이유 4. 통계학적 Random Variable 해석 만족
