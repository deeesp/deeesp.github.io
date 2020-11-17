---
title:  "[통계] Estimation (1) - Unbiased Estimator (불편 추정기)"
excerpt: "Detection and Estimation Series - Unbiased Estimator의 정의와 왜 unbiased 해야 하는지"
categories:
  - Statistics
  
tags:
  - Statistics
  - Signal Processing
  - Machine Learning
  - Detection and Estimation
  - Unbiased Estimator
  - MVUE
  - 통계
  - 신호처리
  - 검출 및 추정
  - 불편 추정기
  - 불편 추정량

  
last_modified_at: 2020-11-10-17:00:00

toc: true
toc_sticky: true
---

앞서 Detection and Estimation Series [Estimation (0)](https://deeesp.github.io/statistics/Detection_n_Estimation/) 에서 Detection과 Estimation의 기본적인 개념을 다루어 보았다.

이번 글이서는 Unbiased Estimatior에 대해 다루어 보겠다.

## [1] Unbiased Estimator (불편 추정기)

### 정의
 Estimator가 *On the average <sup>평균적으로</sup>* unknown parameter의 실제 값을 산출해내면 **Unbiased** 하다.
 
 $$s.t. \text{if}\ E(\hat{\theta}) = \theta, \quad a< \theta <b \quad \text{for } \forall \theta \quad \text{, then estimator } \hat{\theta} \text{ is } \textbf{unbiased}$$
 
- $(a,b)$ 범위로 인해, estimator가 on the average $\theta$를 산출해낼 것이다.
- $\hat{\theta}=g(\textbf{x})$라 하면, $E(\hat{\theta}) =\int{g(\textbf{x})p(\textbf{x};\theta)d\textbf{x}}= \theta$
- $E(\hat{\theta})\ne \theta$이면 bias는 $b(\theta) = E(\hat{\theta}) - \theta$ 이다.

### 어떤 Estimator 가 더 좋은 Estimator인가?
- Unbiased estimator는 반드시 좋은 estimator라고 할 수는 없다.
- **하지만,** biased estimator는 안좋은 estimator이다.
- 이에 대한 기준은 뒤에서 더 자세하게 다룰 예정이다. (Hint: MVUE)

### 예시
1. Unbiased estimator for DC level in WGN<sup>White Gaussian Noise</sup>
2. Biased estimator for DC level in WGN


## [2] 평균제곱오차 (MSE - Mean Squarred Error Criterion)

$$\text{MSE}(\hat{\theta}) = E\left[ \left(\hat{\theta} - \theta \right)^2 \right]$$

 MSE<sup></sup>는 위 식과 같이 실제 값으로부터 estimator의 편차 제곱의 평균 값을 측정한다.

 하지만, MMSE(Minimum MSE) criterion은 많은 경우에 데이터에 대한 함수만으로 나타낼 수 없는 실현 불가능한<sup>Unrealizable</sup> estimator를 만들어 낸다.
 
 아래 수식 처럼, Optimal parameter 찾기 위한 criterion을 모델링 할 때, bias에 unknown parameter $\theta$를 가져 unrealizable하기 때문에, 현실적으로 MMSE를 estimator로 고려하면 안된다.
 
$E\left[ \left(\hat{\theta} - \theta \right)^2 \right]$

$=E \left[ \left[\left(\hat{\theta} - E(\hat{\theta}) \right) +\left(E(\hat{\theta}) - \theta \right) \right]^2 \right]$

$= \text{var}(\hat{\theta}) +\left[E(\hat{\theta})-\theta\ \right]^2$

$= \text{var}(\hat{\theta}) +b^2(\theta)$

이에 대한 대안으로 bias가 0인 값을 갖고, 분산<sup>variance</sup>을 최소화 하도록 제한을 주는 *Minimum Variance Unbiased Estimator* (**MVUE**)가 쓰인다. MVUE는 다음 글에서 자세히 다루어보겠다.

## Reference
- [1] S. Kay. Fundamentals of Statistical Signal Processing: Estimation Theory, Prentice-Hall International Editions, Englewood Cliffs, NJ, 1993.
- [2] GIST EC7204 Detection and Estimation Lecture from Prof. 황의석
