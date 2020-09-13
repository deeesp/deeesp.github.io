---
title:  "[통계] Estimation (1) - Unbiased Estimator"
excerpt: "Detection & Estimation의 개념과 Unbiased Estimator"
categories:
  - 통계
  
tags:
  - 통계
  - 신호처리
  - Statistics
  - Signal Processing
  - Detection
  - Estimation
  - 검출 및 추정
  - 
  
last_modified_at: 2020-09-13-17:00:00

toc: true
toc_sticky: true
---


통계학적 신호처리<sup>Statistical Signal Processing</sup>는 DSP<sup>디지털 신호처리 - Digital Signal Processing</sup>의 일종으로, 신호의 Detection<sup>검출</sup> 및 Estimation<sup>추정</sup>, Time-series Analysis<sup>시계열 분석</sup> 등을 다룬다.
특히, 신호의 detection 및 estimation은 신호에서 **정보를 추출**해 내는 것에 목적을 둔 학문 분야이다. Noisy observation 환경에서 정보의 unknown state value를 최적를 통해 추론해낸다.

## Statistics<sup>통계학</sup> vs Machine Learning<sup>기계학습</sup>
- Statistics와 Machine Learning 모두 다음과 같은 질문에 답할 수 있다.
	> How do we learn from data?
- 통계학은 formal한 Statistical **"Inference<sup>추론</sup>"** 에 초점이 맞추어져 있다.
- 반면에 Machine Learning은 고차원 예측 문제를 다룬다.

## Detection & Estimation에 대한 예제
- 레이더<sup>RADAR - RAdio Detection And Ranging</sup>
	- Detection: 항공기의 존재
	- Estimation: 항공기의 위치를 결정
- 디지털 통신<sup>Digital Communications</sup>
	- Detection: '0'과 '1' 중 어떤 것이 전송되었는지
	- Estimation: 신호의 반송주파수<sup>Carrier Frequency</sup>를 추정하여 신호를 복조<sup>Demodulate</sup>할 수 있다.


## Detection과 Estimation의 차이점
### Detection
- Hypotheses<sup>가설</sup>의 discrete set이다.
- Right / Wrong 으로 구분된다.
### Estimation
- 가설의 continuous set이다.
- 거의 항상 wrong이지만, error<sup>오차</sup>를 최소화하도록 한다.
### Classical Approach
- Parameters / Estimators는 deterministic하다고 가정한다.
### Bayesian Approach
- Hypotheses와 Parameters는 확률변수<sup>Random Variable</sup>로 여겨지며, assumed *a priori* distribution 가진다.


## Estimation에 대한 기본 개념

### Parameter Estimation<sup>매개변수 추정</sup>
신호처리에서의 Waveforms<sup>파형</sup> 또는 Data set<sup>데이터 셋</sup>은 "Real-world"에서는 연속시간을 기반으로 하지만 우리는 *디지털 시스템*인 컴퓨터를 이용해 처리를 하기 때문에, Sampling을 통해 이산시간을 기반으로 parameter를 extract하게 된다. 따라서,  Observatons<sup>관측된 값</sup> $\bf{x}$ (즉, Discrete-time waveform<sup>이산시간 신호</sup>나 Data set)로부터 Parameter $\bf{\theta}$를 estimate / Infer<sup>추정</sup>한다.

- 이들은 수학적으로 다음과 같은 벡터<sup>Vectors</sup> 또는 스칼라<sup>Scalars</sup>로 표현한다.
$$\theta=[\theta_1, \theta_2, ..., \theta_p ]^T$$

$$\textbf{x} = [ x[0], x[1], ... , x[N-1] ]^T$$


### Estimator
- Estimator $\hat{\theta}$: $\bf{x}$의 실현에 대해 값을 $\theta$에 할당한다. (????)
- Estimation: $\theta$ 값은 주어진 $\bf{x}$의 실현값으로 부터 얻어지는 값이다.
- Estimator $\hat{\theta}$는 추정을 위해 사용되지만,  $\theta$는 unknown parameter의 실제 값을 표현하기 위해 사용된다.


### Parameterized PDF<sup>매개변수화 된 PDF</sup>
> "좋은" estimator를 결정하는 데에 있어서 가장 첫 번째 단계는 데이터를 수학적으로 모델링 하는 것이다.

$$p(\textbf{x}; \bf{\theta})$$

데이터는 본질적으로 random하기 때문에 다음과 같이 PDF<sup>확률밀도함수 - Probability Density Function</sup>로 나타낸다. 이 PDF는 unknown parameter $\theta$로 매개변수화 한 것으로, $\theta$ 값에 따라 다른 PDF를 가지게 된다. 세미콜론 ";"는 dependency<sup>의존성</sup>을 나타내며, unknown parameter $\theta$에 의존성을 띈 N-point data set $\bf{x}$를 나타낸다. "좋은" estimator를 결정하려면, PDF를 구체화 해야 한다. 현실 문제에서는 PDF가 주어지지 않는다. 어떠한 제한과 prior knowledge에도 일관성 있고, 수학적으로도 다루기 쉬운 PDF를 선택해야 한다. 이러한 PDF 기반의 estimation에는 두가지 접근법이 있다.

1. *Classical* Estimation
	- 우리가 관심있는 parameters는 deterministic하지만 unknown하다고 가정한다.
2. *Bayesian* Estimation
	- $p(x;\theta)$에서 $\theta$가 확률변수라는 것을 명심해야 한다.
	- 우리가 estimate하고자 하는 parameter는 확률변수 $\theta$의 실현 값이다.
	- $p(\textbf{x}, \theta)= p(\textbf{x}| \theta)p(\theta)$
	- $p(\theta)$는 사전에 관측된 어떤 데이터로부터 $\theta$에 대한 우리의 knowledge를 요약하는 **prior PDF**
	- $p(\textbf{x}| \theta)$는 $\theta$를 알고 있다는 조건 하에 주어진 데이터 $\textbf{x}$에서 우리의 knowledge를 요약하는 conditional PDF


## Unbiased Estimator<sup>불편 추정량</sup>
### 정의
Estimator가 *On the average <sup>평균적으로</sup>* unknown parameter의 실제 값을 산출해내면 **Unbiased** 하다. 즉,  $a< \theta <b$의 범위 내 가능한 모든 parameter $\theta$에 대해서 $E(\hat{\theta})=\theta$이면, Estimator $\hat{\theta}$는 **Unbiased** 하다.
- $(a,b)$ 범위로 인해, estimator가 on the average $\theta$를 산출해낼 것이다.
- $\hat{\theta}=g(\textbf{x})$라 하면, $E(\hat{\theta}) =\int{g(\textbf{x})p(\textbf{x};\theta)d\textbf{x}}= \theta$
- $E(\hat{\theta})\ne \theta$이면 bias는 $b(\theta) = E(\hat{\theta}) - \theta$ 이다.

### 주의할점 
- Unbiased Estimator는 반드시 좋은 estimator라고 할 수는 없다.
- **하지만,** Biased estimator는 안좋은 estimator이다.

### 예시
1. Unbiased estimator for DC level in WGN<sup>White Gaussian Noise</sup>
2. Biased estimator for DC level in WGN

## MSE (Mean Squarred Error) Criterion
$$\text{MSE}(\hat{\theta}) = E\left[ \left(\hat{\theta} - \theta \right)^2 \right]$$

MSE<sup>평균제곱오차</sup>는 위 식과 같이 실제 값으로부터 estimator의 편차 제곱의 평균 값을 측정한다. 하지만, 많은 경우에 MMSE(Minimum MSE) criterion은 데이터에 대한 함수만으로 나타낼 수 없는 unrealizable<sup>실현불가능한</sup> estimator를 야기한다. 즉, Optimal parameter를 나타낼 때 unknown parameter로 모델링해야 하기 때문에, unrealizable하다.

$E\left[ \left(\hat{\theta} - \theta \right)^2 \right]$
$=E \left[ \left[\left(\hat{\theta} - E(\hat{\theta}) \right) +\left(E(\hat{\theta}) - \theta \right) \right]^2 \right]$
$= \text{var}(\hat{\theta}) +\left[E(\hat{\theta})-\theta\ \right]^2$
$= \text{var}(\hat{\theta}) +b^2(\theta)$

현실적인 관점에서 Bias를 가진 criterion은 unrealizable한 경향이 있기 때문에, 따라서 현실적으로 MMSE estimator는 고려되면 안된다. 이에 대한 대안으로 bias가 0인 값을 갖고, variance<sup>분산</sup>을 최소화 하도록 제한을 주는 *Minimum Variance Unbiased Estimator* (**MVUE**)가 쓰인다. 
