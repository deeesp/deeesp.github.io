---
title:  "[Random Process 01] Probability(확률론) "
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

### 0. **들어가기 앞서..**
---
확률론은 특정 게임에서 가능성을 분석하는데서 기원하여, 특히나 과학 기술분야에서 많이 응용되고 있다. 같은 개념임에도 다른 용어를 쓰기도 하고 표기법도 다른 경우도 많기 때문에, 무엇보다 먼저 용어를 확실히 하고 들어가야 하는 것이 중요하다고 생각한다. 특히, 집합에 대한 사전 지식이 있어야 이해 가능하다.   
   
   
### 1. Random Experiment(랜덤시행)
---
- **Experiment(시행)**: 모든 관찰(Observation)의 과정을 말한다.
- **Outcomes(결과)**: Observation에 대한 results를 Experiment에 대한 *Outcomes(결과)* 라고 불린다.
- **Random Experiment(확률시행, 랜덤시행)**: Outcomes를 예측할 수 없는 *Experiment(시행)* 을 말한다.   
   
   
   
### 2. Sample(표본) $\zeta$, Sample Space(표본공간): Ω 또는 $S$
---
- **Sample Space(표본공간) $S$** 는 실험 결과 하나하나를 모은 것을 말하며, 모든 가능한 결과(발생할 수 있는 하나의 현상)들을 포함하는 공간이다.
- 다시 말해, Random Experiment에서 모든 가능한 Outcomes의 *전체집합(Universal Set)* 을 말한다. **$ P(S) = 1 $** 
- 즉, Sample Space는 모든 가능한 Outcomes가 **Mutually Exclusive(상호 배타적)** [^ME] 이며 **Collectively Exhaustive(전체를 이루는)** [^CE] 집합이다.
- 이 Sample Space $ S $의 원소를 **Sample(표본) $\zeta$** 이라고 하며, **Probability Sample(확률표본), Random Sample, Sample Point(표본, 표본점)** 이라고도 불린다.
- **Null Space $ \varnothing $**: 가능한 Outcomes가 없는 공간 $ P( \varnothing ) = 1 $   
- **Typical e.g.** :  **(1)동전 던지기, (2)주사위 던지기**  <img src="https://image.flaticon.com/icons/svg/1715/1715535.svg" width="5%" height="5%" title="cointoss">

  > (1) 동전을 두 번 튕기는 Random Experiment에서, 가능한 outcomes는 다음과 같다.   
  > -> $\{HH,\ HT,\ TH,\ TT\}$ (H: 앞면, T: 뒷면)   
  > 이 네 쌍이 Random Experiments의 outcomes이며 각각 Sample point가 되고, 이들의 전체집합이 Sample Space를 형성한다.   
  > Sample Space $S =${$HH,\ HT,\ TH,\ TT$}   
  
  > (2) 정육면체 주사위를 굴리는 Random Experiment에서, 형성하는 Sample Space는 다음과 같다.   
  > $S$ = {1, 2, 3, 4, 5, 6}
  > 이 때, 각 원소가 Sample point가 된다.
   
[^ME]: **Mutually Exclusive(상호 배타적)** : Two sets $A$ and $B$ are mutually exclusive if $A\cap B=0$   
[^CE]: **Collectively Exhaustive(전체를 이루는)** A collection of sets $A_1,\ldots , A_n$ is collectively exhaustive if and only if $A_1\cup A_2 \cup \ldots \cup A_n$   
   
   
### 3. Events (사건), Event Space(사건 공간): $F$
---
- Sample space $S$의 부분집합은 하나의 **Event(사건)** 이다. 즉, Experiment의 Outcomes의 집합이 Event이다.
- $S$에서 하나의 Sample point $\zeta$는 **Elementary Event(단순사건, 기본사건)** 이라고 종종 불린다.
- Sample Space $S$는 $S$ 자신의 부분집합($ S \subset S $)이며, *Certain Event* 또는 *Sure Event* 라고 불린다.
- Event $A$가 일어날 확률: $P(A) \geq 0$
- Event $A$가 일어나지 않을 확률: $P(\bar{A}) = 1-P(A)$
- Event $A$와 Event $B$가 Mutually Exclusive[^ME]이면, $P(A \cap B) = 0$ 즉, $P(A+B) = P(A) + P(B)$
- Event들이 집합 혹은 모임(collection) $F$를 형성하고 다음 조건을 만족할 경우 **Event Space(사건공간)** 라고 한다.
> (Let Event $A \subset S$)   
> $S \in F $   
> if $\ A\in F$, then $\bar{A} \in F $ (M.E.)   
> if $\ A_i\in F$ for $i \ge 1$, then $\bigcup_{i=1}^{\infty}A_i  \in F $ (C.E.)   
- **Event Space(사건 공간) $F$** 은 *Event Class(사건 클래스)* 또는 *Field(필드)* 라고도 불린다.   
- **Quiz:** Event Space는 Events의 Mutually Exclusive and Collectively Exhaustive한 집합인가?[^Quiz]

[^Quiz]: Answer - True??? False? Sample Space 전체를 커버해야 하지 않나?
   
   
### 4. Probability Axioms
---
1. For the empty set $\varnothing$, called impossible event, $P(\varnothing)=0$
2. For any event $A$, $P(A) \geq 0$
3. For any countable collection $A_1, A_2, \ldots$ of mutually exclusive events, $ P ( \bigcup_{i=1}^{\infty}A_n ) = \sum_{n=1}^{\infty} P(A_n)$
4. $P(S) = 1$

### 5. *Sample Space* vs. *Event Space* ?
---
- Event란 Sample Space의 부분집합이고, 이 Events 중 특정 조건에 맞는 Events의 집합이 Event Space 이다.   
- Sample Space와 Event Space 둘다 각 원소 간 Mutually Exclusive and Collectively Exhaustive한 집합이다.
- **Granularity** : Sample Space의 원소는 Outcomes만 가지고 있어 *Finest-grain* 한 반면, Event Space는 *Compounded Outcomes* 를 원소로 가지고 있다.  
   
### 6. Reference
---
본 포스트는 다음 자료를 기반으로 작성되었습니다.
- GIST 황의석 교수님의 Random Process 강의
- Gubner, Probability and Random Processes for Electrical and Computer Engineers   
- Hwei Hsu, Schaum's Outline of Probability, Random Variables, and Random Processes   
   
   
---
