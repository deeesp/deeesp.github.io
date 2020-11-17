---
title:  "[통계] Estimation (0) - Detection 과 Estimation의 개념"
excerpt: "Detection and Estimation Series - Estimation 이란"
categories:
  - Statistics
  
tags:
  - Statistics
  - Signal Processing
  - Machine Learning
  - Detection and Estimation
  - Parameter Estimation
  - 파라미터 추정
  - 통계적 신호처리
  - 통계
  - 신호처리
  - 검출 및 추정

  
last_modified_at: 2020-11-09-17:00:00

toc: true
toc_sticky: true
---

Detection and Estimation Series 에는 2020년 2학기에 듣고 있는 <검출 및 추정> 과목에서 공부했던 이론을 바탕으로 작성할 예정이다.
과목 이름은 검출 및 추정이지만, 참고서는 Steven. Kay 저자의 통계학적 신호처리 <sup>Statistical Signal Processing</sup> 이다. 거기서 거기란 얘기다.

## [1] 통계학 (Statistics) vs. 기계학습 (Machine Learning)
 통계학적 신호처리 <sup>Statistical Signal Processing</sup> 는 디지털 신호처리 <sup>DSP - Digital Signal Processing</sup> 분야의 한 갈래로, 신호의 검출 <sup>Detection</sup> 및 추정 <sup>Estimation</sup>, 시계열 분석 <sup>Time-series Analysis</sup> 등을 다룬다. 하지만 신호를 데이터로 했을 뿐이지 그냥 통계학이다 ;;;;;
 
 특히, 신호의 detection 및 estimation은 신호에서 **정보를 추출 <sup>Extract</sup>** 해 내는 것에 목적을 둔 학문 분야이다. Noisy observation 환경에서 알려지지 않은 정보에 대해 최적의 state value를 추론해낸다.
 
- Statistics와 Machine Learning 모두 다음과 같은 질문에 답할 수 있다.
	> How do we learn from data?
- Statistics는 formal한 Statistical "**Inference<sup>추론</sup>**" 에 초점이 맞추어져 있다.
- Machine Learning은 고차원 예측 문제를 구체화 한다.

## [2] 검출 및 추정 (Detection & Estimation)의 예시
- 레이더 <sup>RADAR - RAdio Detection And Ranging</sup>
	- Detection: 항공기의 존재
	- Estimation: 항공기의 위치를 결정
- 디지털 통신 <sup>Digital Communications</sup>
	- Detection: '0'과 '1' 중 어떤 것이 전송되었는지
	- Estimation: 신호를 복조<sup>Demodulate</sup>하기 위한 반송주파수<sup>Carrier Frequency</sup>를 추정
- 이미지 분석 <sup>Image Analysis</sup>
	- Detection: 적외선 감시를 통해 특정 물체의 존재 유무
	- Estimation: 카메라 이미지에서 물체의 위치와 방향


## [3] Detection과 Estimation의 차이점
### Detection
- Hypotheses <sup>가설</sup>의 discrete set이다.
- Right / Wrong 으로 구분된다.

### Estimation
- 가설의 continuous set이다.
- 거의 항상 wrong이지만, 오차 <sup>Error</sup>를 최소화하도록 한다.


## [4] Estimation에 대한 기본 개념

### 매개변수 추정(Parameter Estimation)

 "Real-world"에서 우리가 접하는 신호 파형<sup>Waveforms</sup> 또는 데이터 셋 <sup>Data set</sup>은 보통 연속시간을 기반으로 한다. 하지만 우리는 *디지털 시스템*인 컴퓨터를 이용해 처리를 하기 때문에, 연속시간 정보의 Sampling을 통해 이산시간을 기반으로 parameter를 extract하게 된다.
 
 따라서, 이산시간 <sup>Discrete-time</sup> 의 waveforms 또는 Data set으로 이루어진 관측 값 <sup>Observatons</sup> $\bf{x}$로부터 parameter $\bf{\theta}$를 estimate / Infer<sup>추정</sup>한다. 즉, N-point의 data set $\bf{x}$는 알려지지 않은 parameter $\bf{\theta}$의 분포를 따르는데, 이 알려지지 않은 $\bf{\theta}$를 주어진 $\bf{x}$로 밝혀내겠다는 얘기다.

- 이들은 수학적으로 다음과 같은 벡터<sup>Vectors</sup> 또는 스칼라<sup>Scalars</sup>로 표현한다.

$$\theta=[\theta_1, \theta_2, ..., \theta_p ]^T$$

$$\textbf{x} = [ x[0], x[1], ... , x[N-1] ]^T$$

- 여기서는 단일 값을 추정하는 점 추정 <sup>Point Estimation</sup> 을 주로 다룬다. 상반되는 개념으로는 parameter의 구간을 추정해 내는 구간 추정 <sup>Interval Estimation</sup> 이 있다.


### 추정기 (Estimator)

- 한국어로는 추정기 또는 추정량이라고 불린다. 어감이 다소 이상하여 Estimator로 통일하는게 좋은 듯 하다.
- Estimation: 실제 주어진 관측으로 얻어진 실현 값<sup>Realization Value</sup> $\bf{x}$으로 부터 unknown parameter $\theta$ 값을 구하는 것을 말한다.
- Estimator $\hat{\theta}$: 위의 Estimation 하는 방법을 말하며, 일종의 function $$g$$ 이다.

$$\hat{\theta} = g( x[0], x[1], ... , x[N-1] )$$


### 매개변수화 된 확률밀도함수 (Parameterized PDF Probability Density Function)
> "좋은" estimator를 결정하는 데에 있어서 가장 첫 번째 단계는 데이터를 수학적으로 모델링 하는 것이다.

$$p(\textbf{x}; \bf{\theta})$$

 데이터는 본질적으로 random하기 때문에 위 식과 같은 PDF로 나타낸다. 이 PDF는 unknown parameter $\theta$로 매개변수화 한 것으로, $\theta$ 값에 따라 다른 PDF를 가지게 된다. 세미콜론 ";"는 의존성<sup>dependency</sup>을 나타내며, unknown parameter $\theta$에 의존성을 띈 N-point data set $\bf{x}$를 나타낸다.
 
 "좋은" estimator를 결정하려면 PDF를 구체화 해야 하지만, 현실 문제에서는 PDF가 주어지지 않는다. 어떠한 제한과 prior knowledge에도 일관성 있고, 수학적으로도 다루기 쉬운 PDF를 선택해야 한다. 이러한 PDF 기반의 estimation에는 두가지 접근법이 있다.

1. *Classical* Estimation
	- 우리가 관심있는 unknown parameters가 random하지 않고 deterministic하다고 가정한다. 즉, 고정되어 있는 unknown parameter를 estimation 하는 방법이다.
	
2. *Bayesian* Estimation
	- Hypotheses와 Parameters는 *a priori* distributions를 가정한, 확률변수<sup>random variable</sup>로 다뤄진다.
	- $p(x;\theta)$에서 $\theta$ 가 우리가 estimation 하고자 하는 parameter로, random variable 이라는 것을 명심해야 한다.
	- $p(\theta)$는 사전에 관측된 어떤 데이터로부터 $\theta$에 대한 우리의 knowledge를 요약하는 **prior PDF** 이다.
	- $p(\textbf{x} \| \theta)$는 $\theta$를 알고 있다는 조건 하에 주어진 데이터 $\textbf{x}$에서 우리의 knowledge를 요약하는 conditional PDF이다.
	
$$p(\textbf{x}, \theta)= p(\textbf{x} \mid \theta)p(\theta)$$


### Notation
- $$p(x;\theta)$$,  $$p(x,\theta)$$,  $$p(x\mid \theta)$$ 간의 차이점 및 관계

1. $p(x;\theta)$는 Parameterized pdf 이다.

- Random Variable(이하 r.v. : 확률변수) 𝑋 의 한 점 𝑥에서의 Probability Density (이하 pdf : 확률분포) 를 말하는데, 여기서 𝜃 는 어떤 분포에 대한 parameter 이다.

2. $$p(x,\theta)$$는 Joint pdf 이다.

- $$𝑋$$와 $$\Theta$$의 한 점 $$(x,\theta)$$에서의 Joint pdf (결합 확률분포)를 말한다. 이는 $$\Theta$$ 가 r.v.일 때만 성립한다.
- Intersection : 사건의 개념으로 보았을 때에는 두 사건의 교집합, 그래프로 시각화 하여 보았을 때에는 두 분포가 겹치는 부분이 되겠다.

3. $$p(x\mid \theta)$$ 는 Conditional pdf 이다.

- $$\Theta$$가 주어졌을 때 $$𝑋$$의 Conditional pdf (조건부 확률분포)이며, 이는 $$\Theta$$가 r.v. 일 때만 성립한다.
- 다음과 Marginal pdf 와 Joint pdf 로 정의할 수 있다.

$$p(x\mid\theta) = \frac{p(x,\theta)}{p(x)} \text{,     } \ p(x)>0$$

- 이 식은 곧 Bayesian Estimation에 응용된다.


## Reference
- [1] S. Kay. Fundamentals of Statistical Signal Processing: Estimation Theory, Prentice-Hall International Editions, Englewood Cliffs, NJ, 1993.
- [2] GIST EC7204 Detection and Estimation Lecture from Prof. 황의석
