---
title:  "[Random Process] Random Variables "
excerpt: "Random Process의 기반이 될 확률론에 대한 기본 정의"

categories:
  - Random Process
  
tags:
  - Statistics
  - Random Process
  - Random Variables
  - Probability
  - 확률과 통계
  - 랜덤 프로세스
  
last_modified_at: 2020-05-04

toc: true
toc_sticky: true
---

## 0. Random varialbe (확률변수)
### Definition
> A **random variable** consists of  
> 1) an expriment with a probability measure $P[\cdot ]$ defined on a sample space $S$  
> 2) and a function that assigns a real number to each outcome in the sample space of the experiment.  

- 확률변수는 확률실험의 표본 공간의 원소인 결과들에 숫자를 부여하는 것이다.
- 즉, 확률변수는 다양한 사건들에 대한 확률을 계산하기 쉽도록 하는 `변수`가 아니라 관측 `함수`이다.

### Ex
- 광섬유에 광 검출기를 달아 $10^{-6}$초 동안 도착하는 광전자의 수를 세는 확률실험을 한다고 하자. 이 때, 각각의 관측을 확률변수 X로 볼 수 있다. 그러면 X의 치역은 $S_X = {0,1,2,...}$이고, 표본공간 S와 같게 된다.


## 1. Discrete Random Variables (이산 확률변수)

### 1-1. Definition
> X is a discrete random variable, if the range of X is a countable set.
> 만약 확률변수 X의 치역이 셀 수 있는 집합이면, X는 이산 확률변수이다.

### 1-2. Probability Mass Funciton (PMF - 확률질량함수)
> The PMF of a r.v. $X$ expresses the probability model of an expreiment as a mathematical funciton. The funciton is the probability $P[X=x]$ for every number $x$.  
> 확률변수 $X$의 PMF는 확률실험의 확률 모델을 수학적 함수로 표현한 것이다. 그 함수는 각 숫자 $x$에 대한 확률 $P[X=x]$ 이다.  
> $P_X(x) = P[X=x]$  

### 1-3. Theorem of PMF
> For a discrete r.v. $X$ with PMF $P_X(x)$ and range $S_X$:
> (a) For any x, $P_X(x) \ge 0$  
> (b) $\sum_{x \in S_X} P_X(x) = 1$  
> (c) For any event $B \subset S_X$, the probability that $X$ is in the set $B$ is
$$P[B]=\sum_{x\in B}P_X(x)$$  

### Reference
---
본 포스트는 다음 자료를 기반으로 작성되었습니다.
- GIST 황의석 교수님의 Random Process 강의
- Gubner, Probability and Random Processes for Electrical and Computer Engineers   
- Hwei Hsu, Schaum's Outline of Probability, Random Variables, and Random Processes   
- Roy D. Yates, David J. Goodman - Probability and Stochastic Processes_ A Friendly Introduction for Electrical and Computer Engineers    
   
---
