---
title:  "[Random Process 01] Probability(확률론)"
excerpt: "Random Process의 기반 확률에 대한 기본 정의"

categories:
  - Random Process
  
tags:
  - Statistics
  - Random Process
  - Random Variables
  - Probability
  - 확률과 통계
  - 랜덤 프로세스
  
last_modified_at: 2020-03-17

toc: true
toc_sticky: true
---

확률론은 특정 게임에서 가능성을 분석하는데서 기원하여, 특히나 과학 기술분야에서 많이 응용되고 있다. 같은 개념임에도 다른 용어를 쓰기도 하고 표기법도 다른 경우도 많기 때문에, 무엇보다 먼저 용어를 확실히 하고 들어가야 하는 것이 중요하다고 생각한다.   


### 1. Random Experiment(랜덤시행)
- **Experiment(시행)**: 모든 관찰(Observation)의 과정을 말한다.
- **Outcomes(결과)**: Observation에 대한 results를 Experiment에 대한 *Outcomes(결과)* 라고 불린다.
- **Random Experiment(확률시행, 랜덤시행)**: Outcomes를 예측할 수 없는 *Experiment(시행)* 을 말한다.   


### 2. Sample Space(표본 공간): Ω 또는 S
- Sample Space $S$는 Random Experiment의 모든 가능한 outcomes의 전체집합(Universal Set)을 말한다. 
- 즉, 실험 결과 하나하나를 모은 것을 말하며, 모든 가능한 결과(발생할 수 있는 하나의 현상)들을 원소로 갖는 전체집합이다.
- 이 Sample Space $ S $의 원소를 **Sample(표본)** 이라고 하며, Probability Sample(확률표본), Random Sample, Sample Point(표본, 표본점) 이라고도 불린다.
- 하나의 Sample은 뒤에서 나올 Events(사건) 중 **기본사건(Elementary Event)** 을 말한다.
- Sample space는 모든 가능한 Outcomes가 [Mutually Exclusive(상호 배타적)](https://en.wikipedia.org/wiki/Mutual_exclusivity)이며 [Collectively Exhaustive(전체를 이루는)](https://en.wikipedia.org/wiki/Collectively_exhaustive_events) 집합이다. $ P(S) = 1 $ 
- Null Space $ \emptyset $: 가능한 outcomes이 없는 공간 $ P( \emptyset ) = 1 $
- **typical e.g.** <img src="https://image.flaticon.com/icons/svg/1715/1715535.svg" width="30%" height="30%" title="cointoss"></img>

  > 동전을 두 번 튕기는 Random Experiment에서, 가능한 outcomes는 다음과 같다.   
  > { HH, HT, TH, TT } (H: 앞면, T: 뒷면)   
  > 이 네가지 쌍이 Random Experiments의 outcomes이며 각각 Sample point가 되고, 이들의 전체집합이 Sample Space를 형성한다.
  > Sample Space $S = \{ HH,\ HT,\ TH,\ TT \}$


### 3. Events (사건)

- Sample space S의 부분집합이다.
- S에서 하나의 Sample point는 Elementary Event(단순사건, 기본사건)이라고 종종 불린다.
- Sample Space S는 S 자신의 부분집합($ S \subset S $)이며, 특정 Event이다.  
- 사건 A가 일어날 확률 $P(A) \geq 0$
- 사건 A가 일어나지 않을 확률 $P(\bar{A}) = 1-P(A)$
- 사건 A와 사건 B가 Mutually Exclusive(상호 배타적 즉, 독립이 )이면, $P(A \cap B) = 0$
  

### 5. Event Space(사건 공간): F

- Event Class(사건 클래스) 또는 Field(필드)라고도 불린다.
- Sample space S의 부분집합은 하나의 event가 될 수 있다.
- 이 event들의 집합 혹은 모임(collection)을 형성하고 다음 조건을 만족할 경우 Event Space라고 한다.

$$ S \in F $$

$$ if \ A\in F, \ then \ \bar{A} \in F $$

$$ if \ A_i\in F \ for\ it \ge 1,\ then \ \bigcup_{i=1}^{\infty}A_i  \in F $$  


### *Sample Space* vs. *Event Space* ?

- 즉, event란 sample space의 부분집합이고, 모든 가능한 event 들의 집합이 event space 이다.  
  

### Reference
---
본 포스트는 다음 자료를 기반으로 작성되었습니다.
- GIST 황의석 교수님의 Random Process 강의
- Hwei Hsu, Schaum's Outline of Probability, Random Variables, and Random Processes
- Gubenr, Probability and Random Processes for Electrical and Computer Engineers
