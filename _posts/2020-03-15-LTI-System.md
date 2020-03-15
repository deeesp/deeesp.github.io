---
title:  "LTI System"
excerpt: "신호처리의 기초 LTI system"

categories:
  - signal
tags:
  - 신호및시스템
  - Signals_and_Systems
last_modified_at: 2020-03-15

toc: true
toc_sticky: true
---


## LTI system 이란?
- Linear Time-Invariant System, 즉 선형 시불변 시스템이다.
~라고 말하면 면접에서 떨어진다.~

### LTI 시스템이 중요한 역할을 하는 이유
1. Many physical processes posses Linearity and Time Invariance properties thus can be modeled as LTI systems
2. LTI systems can be analyzed in considerable detail, providing both insight into their properties and a set of powerful tools that form the core of signal and system analysis.

### 그럼 어떻게 혀?
1. Represent the input to an LTI system in terms of a linear combination of a set of basic signals
2. Use superposition to compute the output of the system in terms of its responses to these basic signals

- Unit impulse → very general signals can be represented  as linear combinations of delayed impulses.
- superposition + time invariance → characterization of any LTI systems in terms of its response to a unit impulse with convolution sum(integral).


---

## 그전에 Convolution 연산을 알아봅시다.

### Convolution 연산은 어따 쓰는겨?
LTI시스템 상에서 이전값과 현재값을 연산하기 위해

시스템이 메모리가 있는 경우, 시스템의 출력이 현재 입력에 의해서만 결정되는 것이 아닌 이전 입력(causal system 이라면)에 의해서도 영향을 받기 때문에 그에 대한 출력을 나타내기 위해 하는 연산이다.

예를 들어 종(鍾)을 LTI(Liner Time Invariant) 시스템이라고 가정한다면 종을 한번 치면 그 소리가 치는 순간만 나는게 아니라 치는 순간에 소리가 크게 났다가 점점 소리가 감쇄되며 작아진다. 그림으로 나타내면 다음 그림의 첫번째 경우와 같다. 종을 한번 탕 치는 것을 impulse 입력이라 하고 한 번 종을 쳤을 때 나는 소리를 삼각형으로 나타내었다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e8849c15-1c79-49a3-9e9b-ccb0a1f64337/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e8849c15-1c79-49a3-9e9b-ccb0a1f64337/Untitled.png)

그런데 종을 한번 치고 다시 치면 어떨까? 그림의 두번째 경우는 처음 종을 치고 잠시 후 이전보다 약하게 친 경우이다. 이 때는 종소리를 Linear system으로 가정했기 때문에 이전의 입력에 의해 나고 있는 소리에 현재 입력에 의해 나는 소리가 더해져 나타난다. 그리고 이것은 impulse 입력과 종소리의 convolution 과 같은 결과가 나올 것이다.

### 따라서 convolution은 한 LTI 시스템에 대해 현재와 이전의 입력에 대한 출력을 계산하기 위해 수행하는 것이다.

그림의 두번째 경우를 보면 왜 컨볼루션을 할 때 system이나 입력 중 하나를 반전시켜야 하는지에 대해 알 수 있다.

한 시스템에 시간적으로 앞선 입력을 먼저 넣고 그 후에 시간적으로 나중에 발생한 입력을 넣어야 올바른 출력을 얻을 수 있기 때문에 시간적으로 앞선 입력을 먼저 넣기 위해 입력을 반전시키거나 시스템을 반전시키는 것이다.

입력을 반전시키지 않으면 처음에 종을 약하게 치고 그 후에 세게 친것과 같은 결과가 나오기 때문에 시간적으로 반대의 결과가 나온다.

Reference: [https://trip2ee.tistory.com/101](https://trip2ee.tistory.com/101)

메모리가 있는 선형 시스템 * 임펄스 입력 = 출력 Response

북채 때리는 순간과 내부 공명

 [https://pulse.embs.org/january-2015/history-convolution-operation/](https://pulse.embs.org/january-2015/history-convolution-operation/)
